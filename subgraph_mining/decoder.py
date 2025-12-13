import argparse
import os
import pickle
import random
import time
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
from torch_geometric.datasets import TUDataset, PPI
import torch_geometric.utils as pyg_utils

from common import data, models, utils, combined_syn
from subgraph_mining.config import parse_decoder
from subgraph_matching.config import parse_encoder
from visualizer.visualizer import visualize_pattern_graph_ext
from subgraph_mining.search_agents import (
    GreedySearchAgent,
    MCTSSearchAgent,
    MemoryEfficientMCTSAgent,
    MemoryEfficientGreedyAgent,
    BeamSearchAgent,
)


# ------------------------- Graph Analysis & Chunking -------------------------
def analyze_graph_for_streaming(graph, args):
    """Analyze graph characteristics to decide whether to use streaming mode."""
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    avg_degree = num_edges / num_nodes if num_nodes else 0

    # Bipartite check
    is_bipartite = nx.is_bipartite(graph)

    # Clustering coefficient (sample if very large)
    if graph.is_directed():
        g = graph.to_undirected()
    else:
        g = graph
    if num_nodes > 10000:
        sample_nodes = random.sample(list(g.nodes()), 1000)
        clustering_coef = nx.average_clustering(g, nodes=sample_nodes)
    else:
        clustering_coef = nx.average_clustering(g)

    # Power-law degree
    degrees = [d for _, d in graph.degree()]
    if degrees:
        max_degree = max(degrees)
        median_degree = sorted(degrees)[len(degrees) // 2]
        is_power_law = (max_degree / (median_degree + 1)) > 10
    else:
        is_power_law = False

    # Connectivity ratio
    components = list(nx.weakly_connected_components(graph)) if graph.is_directed() else list(nx.connected_components(graph))
    n_components = len(components)
    connectivity_ratio = len(max(components, key=len)) / num_nodes if components else 0.0

    # Decision logic
    use_streaming = False
    reason = ""
    if is_bipartite:
        reason = "Bipartite graph structure - BFS chunking ineffective"
    elif connectivity_ratio > 0.9:
        reason = f"Well-connected graph (connectivity={connectivity_ratio:.2f})"
    elif is_power_law:
        reason = "Power-law degree distribution - hub nodes would cause imbalance"
    elif num_nodes > 100000 and 5 <= avg_degree <= 20 and clustering_coef > 0.3 and n_components < 100:
        use_streaming = True
        reason = f"Large modular graph (degree={avg_degree:.2f}, clustering={clustering_coef:.3f})"
    else:
        reason = "Graph characteristics don't benefit from chunking"

    return {
        "use_streaming": use_streaming,
        "reason": reason,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": avg_degree,
        "clustering_coef": clustering_coef,
        "n_components": n_components,
        "connectivity_ratio": connectivity_ratio,
        "is_bipartite": is_bipartite,
        "is_power_law": is_power_law,
        "estimated_memory_mb": (num_nodes * 200 + num_edges * 100) / 1024,
    }


def bfs_chunk(graph, start_node, max_size):
    visited = {start_node}
    queue = [start_node]
    while queue and len(visited) < max_size:
        node = queue.pop(0)
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                if len(visited) >= max_size:
                    break
    return graph.subgraph(visited).copy()


def process_large_graph_in_chunks(graph, chunk_size=10000):
    """Split a large graph into BFS-based chunks."""
    remaining_nodes = set(graph.nodes())
    chunks = []
    while remaining_nodes:
        start_node = next(iter(remaining_nodes))
        chunk = bfs_chunk(graph, start_node, chunk_size)
        chunks.append(chunk)
        remaining_nodes -= set(chunk.nodes())
    return chunks


# ------------------------- Synthetic Plant Dataset -------------------------
def make_plant_dataset(size):
    generator = combined_syn.get_generator([size])
    random.seed(3001)
    np.random.seed(14853)
    pattern = generator.generate(size=10)
    nx.draw(pattern, with_labels=True)
    os.makedirs("plots/cluster", exist_ok=True)
    plt.savefig("plots/cluster/plant-pattern.png")
    plt.close()

    graphs = []
    for _ in range(1000):
        graph = generator.generate()
        n_old = len(graph)
        graph = nx.disjoint_union(graph, pattern)
        for _ in range(2):
            u = random.randint(0, n_old - 1)
            v = random.randint(n_old, len(graph) - 1)
            graph.add_edge(u, v)
        graphs.append(graph)
    return graphs


# ------------------------- Chunk Processing -------------------------
def _process_chunk(args_tuple):
    chunk_dataset, task, args, chunk_index, total_chunks = args_tuple
    start_time = time.time()
    original_n_workers = getattr(args, "n_workers", 4)
    args.n_workers = 0

    print(f"[{time.strftime('%H:%M:%S')}] Worker PID {os.getpid()} started chunk {chunk_index + 1}/{total_chunks}", flush=True)

    try:
        result = pattern_growth(chunk_dataset, task, args)
        elapsed = int(time.time() - start_time)
        print(f"[{time.strftime('%H:%M:%S')}] Worker PID {os.getpid()} finished chunk {chunk_index + 1}/{total_chunks} in {elapsed}s ({len(result)} patterns)", flush=True)
        args.n_workers = original_n_workers
        return result
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ERROR in chunk {chunk_index + 1}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        args.n_workers = original_n_workers
        return []


def pattern_growth_streaming(dataset, task, args):
    """Process large graphs in streaming mode using multiple workers."""
    graph = dataset[0]
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    avg_degree = num_edges / num_nodes if num_nodes else 0

    # Adjust chunk size based on density
    if avg_degree > args.dense_graph_threshold:
        chunk_size = min(args.chunk_size, 5000)
    elif avg_degree > 20:
        chunk_size = min(args.chunk_size, 7500)
    else:
        chunk_size = args.chunk_size

    print(f"Partitioning graph into chunks of ~{chunk_size} nodes...")
    graph_chunks = process_large_graph_in_chunks(graph, chunk_size)

    min_chunk_size = max(args.min_pattern_size, 20 if avg_degree < 2.0 else 5)
    graph_chunks = [c for c in graph_chunks if c.number_of_nodes() >= min_chunk_size]
    print(f"Filtered to {len(graph_chunks)} chunks with >= {min_chunk_size} nodes")

    chunk_args = [( [chunk], task, args, idx, len(graph_chunks) ) for idx, chunk in enumerate(graph_chunks)]

    all_patterns = []
    with mp.Pool(processes=args.streaming_workers) as pool:
        results = pool.map(_process_chunk, chunk_args)

    for chunk_patterns in results:
        all_patterns.extend(chunk_patterns)

    print(f"Total patterns discovered: {len(all_patterns)}")
    return all_patterns


# ------------------------- Main Pattern Growth -------------------------
def pattern_growth(dataset, task, args):
    start_time = time.time()
    model_cls = {
        "end2end": models.End2EndOrder,
        "mlp": models.BaselineMLP
    }.get(args.method_type, models.OrderEmbedder)
    model = model_cls(1, args.hidden_dim, args).to(utils.get_device())
    model.eval()
    model.load_state_dict(torch.load(args.model_path, map_location=utils.get_device()))

    if task == "graph-labeled":
        dataset, labels = dataset

    graphs = []
    for i, graph in enumerate(dataset):
        if task == "graph-labeled" and labels[i] != 0:
            continue
        if task == "graph-truncate" and i >= 1000:
            break
        if not isinstance(graph, (nx.Graph, nx.DiGraph)):
            graph = pyg_utils.to_networkx(graph).to_undirected()
            for n in graph.nodes():
                graph.nodes[n].setdefault('label', str(n))
                graph.nodes[n].setdefault('id', str(n))
        graphs.append(graph)

    # Neighbor sampling
    neighs = []
    anchors = []
    if not args.use_whole_graphs:
        for graph in graphs:
            if args.sample_method == "radial":
                for node in graph.nodes():
                    neigh = list(nx.single_source_shortest_path_length(graph, node, cutoff=args.radius).keys())
                    if args.subgraph_sample_size:
                        neigh = random.sample(neigh, min(len(neigh), args.subgraph_sample_size))
                    if len(neigh) > 1:
                        subgraph = graph.subgraph(neigh)
                        mapping = {old: new for new, old in enumerate(subgraph.nodes())}
                        subgraph = nx.relabel_nodes(subgraph, mapping)
                        neighs.append(subgraph)
                        if args.node_anchored:
                            anchors.append(0)
            elif args.sample_method == "tree":
                for _ in range(args.n_neighborhoods):
                    graph, neigh = utils.sample_neigh(graphs, random.randint(args.min_neighborhood_size, args.max_neighborhood_size), args.graph_type)
                    subgraph = nx.convert_node_labels_to_integers(graph.subgraph(neigh))
                    subgraph.add_edge(0, 0)
                    neighs.append(subgraph)
                    if args.node_anchored:
                        anchors.append(0)

    # Embedding computation
    embs = []
    for i in range(0, len(neighs), args.batch_size):
        batch = utils.batch_nx_graphs(neighs[i:i+args.batch_size], anchors=anchors if args.node_anchored else None)
        with torch.no_grad():
            emb = model.emb_model(batch).to(torch.device("cpu"))
        embs.append(emb)

    # Initialize search agent
    if args.search_strategy == "mcts":
        agent_cls = MemoryEfficientMCTSAgent if args.memory_efficient else MCTSSearchAgent
    elif args.search_strategy == "greedy":
        agent_cls = MemoryEfficientGreedyAgent if args.memory_efficient else GreedySearchAgent
    elif args.search_strategy == "beam":
        agent_cls = BeamSearchAgent
    else:
        raise ValueError(f"Unknown search strategy {args.search_strategy}")

    agent = agent_cls(args.min_pattern_size, args.max_pattern_size, model, graphs, embs, node_anchored=args.node_anchored, analyze=args.analyze, model_type=args.method_type, out_batch_size=args.out_batch_size)
    if hasattr(agent, "args"):
        agent.args = args

    out_graphs = agent.run_search(args.n_trials)

    print(f"TOTAL TIME: {int(time.time() - start_time)}s")

    # Visualization
    count_by_size = defaultdict(int)
    successful_vis = sum(1 for pattern in out_graphs if visualize_pattern_graph_ext(pattern, args, count_by_size))
    print(f"Successfully visualized {successful_vis}/{len(out_graphs)} patterns")

    os.makedirs("results", exist_ok=True)
    with open(args.out_path, "wb") as f:
        pickle.dump(out_graphs, f)

    return out_graphs


# ------------------------- Main Function -------------------------
def main():
    os.makedirs("plots/cluster", exist_ok=True)
    parser = argparse.ArgumentParser(description='Decoder arguments')
    parse_encoder(parser)
    parse_decoder(parser)
    args = parser.parse_args()

    # Load dataset
    dataset, task = load_dataset(args)

    # Adaptive streaming decision
    if len(dataset) == 1 and isinstance(dataset[0], (nx.Graph, nx.DiGraph)):
        stats = analyze_graph_for_streaming(dataset[0], args)
        print_graph_analysis(stats)
        if stats["use_streaming"]:
            pattern_growth_streaming(dataset, task, args)
        else:
            pattern_growth(dataset, task, args)
    else:
        pattern_growth(dataset, task, args)


# ------------------------- Dataset Helpers -------------------------
def load_dataset(args):
    """Load dataset based on argument name."""
    if args.dataset.endswith(".pkl"):
        with open(args.dataset, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, (nx.Graph, nx.DiGraph)):
            graph = data
            if args.graph_type == "directed" and not graph.is_directed():
                graph = graph.to_directed()
            elif args.graph_type == "undirected" and graph.is_directed():
                graph = graph.to_undirected()
        elif isinstance(data, dict) and "nodes" in data and "edges" in data:
            graph = nx.DiGraph() if args.graph_type == "directed" else nx.Graph()
            graph.add_nodes_from(data["nodes"])
            graph.add_edges_from(data["edges"])
        else:
            raise ValueError("Unknown pickle format")
        return [graph], "graph"

    elif args.dataset.startswith("plant-"):
        size = int(args.dataset.split("-")[-1])
        return make_plant_dataset(size), "graph"

    # Add other datasets
    dataset_map = {
        "enzymes": TUDataset(root="/tmp/ENZYMES", name="ENZYMES"),
        "cox2": TUDataset(root="/tmp/cox2", name="COX2"),
        "reddit-binary": TUDataset(root="/tmp/REDDIT-BINARY", name="REDDIT-BINARY"),
        "dblp": TUDataset(root="/tmp/dblp", name="DBLP_v1"),
        "coil": TUDataset(root="/tmp/coil", name="COIL-DEL"),
        "ppi": PPI(root="/tmp/PPI"),
    }
    if args.dataset in dataset_map:
        return dataset_map[args.dataset], "graph-truncate" if args.dataset == "dblp" else "graph"

    raise ValueError(f"Unknown dataset {args.dataset}")


def print_graph_analysis(stats):
    print("=" * 60)
    print("GRAPH ANALYSIS")
    print("=" * 60)
    print(f"Nodes: {stats['num_nodes']:,}")
    print(f"Edges: {stats['num_edges']:,}")
    print(f"Average degree: {stats['avg_degree']:.2f}")
    print(f"Clustering coefficient: {stats['clustering_coef']:.3f}")
    print(f"Connected components: {stats['n_components']}")
    print(f"Connectivity ratio: {stats['connectivity_ratio']:.2f}")
    print(f"Estimated memory: {int(stats['estimated_memory_mb'])}MB")
    print(f"Decision: {'STREAMING MODE' if stats['use_streaming'] else 'STANDARD MODE'}")
    print(f"Reason: {stats['reason']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
