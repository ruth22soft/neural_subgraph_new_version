import argparse
import os
import pickle
import random
import time
from collections import defaultdict
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.multiprocessing as mp

from common import models, utils, combined_syn
from subgraph_mining.config import parse_decoder
from subgraph_matching.config import parse_encoder
from subgraph_mining.search_agents import (
    GreedySearchAgent,
    MCTSSearchAgent,
    MemoryEfficientMCTSAgent,
    MemoryEfficientGreedyAgent,
    BeamSearchAgent,
)

# =================== Graph Analysis ===================
def analyze_graph_for_streaming(graph, args):
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    avg_degree = num_edges / num_nodes if num_nodes else 0

    clustering_coef = nx.average_clustering(graph.to_undirected())
    degrees = [d for _, d in graph.degree()]
    max_degree = max(degrees) if degrees else 0
    median_degree = sorted(degrees)[len(degrees) // 2] if degrees else 0
    is_power_law = (max_degree / (median_degree + 1)) > 10 if degrees else False

    components = list(nx.connected_components(graph.to_undirected()))
    n_components = len(components)
    largest_cc_size = len(max(components, key=len)) if components else 0
    connectivity_ratio = largest_cc_size / num_nodes if num_nodes else 0

    use_streaming = False
    reason = "default"

    if connectivity_ratio > 0.9:
        use_streaming = False
        reason = "well-connected graph, standard mode"
    elif num_nodes > 100000:
        use_streaming = True
        reason = "large graph, streaming mode"

    return {
        "use_streaming": use_streaming,
        "reason": reason,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": avg_degree,
        "clustering_coef": clustering_coef,
        "n_components": n_components,
        "connectivity_ratio": connectivity_ratio,
        "is_power_law": is_power_law,
        "estimated_memory_mb": (num_nodes * 200 + num_edges * 100) / 1024,
    }

# =================== Chunking for Streaming ===================
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
    return graph.subgraph(visited)

def process_large_graph_in_chunks(graph, chunk_size=10000):
    all_nodes = set(graph.nodes())
    graph_chunks = []
    while all_nodes:
        start_node = next(iter(all_nodes))
        chunk = bfs_chunk(graph, start_node, chunk_size)
        graph_chunks.append(chunk)
        all_nodes -= set(chunk.nodes())
    return graph_chunks

# =================== Plant Dataset Generator ===================
def make_plant_dataset(size):
    generator = combined_syn.get_generator([size])
    random.seed(3001)
    np.random.seed(14853)
    pattern = generator.generate(size=10)
    nx.draw(pattern, with_labels=True)
    plt.savefig("plots/cluster/plant-pattern.png")
    plt.close()

    graphs = []
    for _ in range(1000):
        graph = generator.generate()
        graph = nx.disjoint_union(graph, pattern)
        for _ in range(2):
            u = random.randint(0, len(graph.nodes()) - 1)
            v = random.randint(0, len(graph.nodes()) - 1)
            graph.add_edge(u, v)
        graphs.append(graph)
    return graphs

# =================== Visualization ===================
def visualize_pattern_graph(pattern, count_by_size):
    """Optimized lightweight plotting"""
    num_nodes = pattern.number_of_nodes()
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(pattern, seed=42, iterations=20)
    nx.draw(
        pattern,
        pos,
        with_labels=True,
        node_size=300,
        node_color='lightblue',
        edge_color='gray',
        font_size=8,
    )
    size = len(pattern)
    count_by_size[size] += 1
    filename = f"{size}_{count_by_size[size]}"
    plt.savefig(f"plots/cluster/{filename}.png", dpi=150)
    plt.close()

# =================== Pattern Growth ===================
def pattern_growth(dataset, task, args):
    start_time = time.time()

    # Load model
    model = models.OrderEmbedder(1, args.hidden_dim, args)
    model.to(utils.get_device())
    model.eval()
    model.load_state_dict(torch.load(args.model_path, map_location=utils.get_device()))

    # Convert dataset to NetworkX graphs if needed
    graphs = []
    for graph in dataset:
        if not isinstance(graph, (nx.Graph, nx.DiGraph)):
            graph = nx.convert_node_labels_to_integers(graph)
        graphs.append(graph)

    # Initialize agent
    agent = GreedySearchAgent(
        args.min_pattern_size,
        args.max_pattern_size,
        model,
        graphs,
        [],
        node_anchored=args.node_anchored,
        analyze=args.analyze,
    )

    out_graphs = agent.run_search(args.n_trials)

    # Visualize patterns
    count_by_size = defaultdict(int)
    for pattern in out_graphs:
        visualize_pattern_graph(pattern, count_by_size)

    # Save results
    os.makedirs("results", exist_ok=True)
    with open(args.out_path, "wb") as f:
        pickle.dump(out_graphs, f)

    print(f"Total time: {time.time() - start_time:.2f}s")
    return out_graphs

# =================== Streaming Growth ===================
def _process_chunk(args_tuple):
    chunk_dataset, task, args, chunk_index, total_chunks = args_tuple
    return pattern_growth(chunk_dataset, task, args)

def pattern_growth_streaming(dataset, task, args):
    graph = dataset[0]
    chunk_size = args.chunk_size
    graph_chunks = process_large_graph_in_chunks(graph, chunk_size)
    chunk_args = [( [chunk], task, args, i, len(graph_chunks)) for i, chunk in enumerate(graph_chunks)]

    with mp.Pool(processes=args.streaming_workers) as pool:
        results = pool.map(_process_chunk, chunk_args)

    all_patterns = []
    for r in results:
        all_patterns.extend(r)
    return all_patterns

# =================== Main ===================
def main():
    os.makedirs("plots/cluster", exist_ok=True)
    parser = argparse.ArgumentParser()
    parse_encoder(parser)
    parse_decoder(parser)
    args = parser.parse_args()

    # Load dataset
    if args.dataset.endswith('.pkl'):
        with open(args.dataset, 'rb') as f:
            data = pickle.load(f)
        graph = data if isinstance(data, (nx.Graph, nx.DiGraph)) else nx.Graph()
        dataset = [graph]
        task = 'graph'
    elif args.dataset.startswith('plant-'):
        size = int(args.dataset.split("-")[-1])
        dataset = make_plant_dataset(size)
        task = 'graph'
    else:
        raise ValueError("Dataset not supported in this optimized version")

    # Check if streaming needed
    graph_stats = analyze_graph_for_streaming(dataset[0], args)
    if graph_stats["use_streaming"]:
        out_graphs = pattern_growth_streaming(dataset, task, args)
    else:
        out_graphs = pattern_growth(dataset, task, args)

if __name__ == '__main__':
    main()
