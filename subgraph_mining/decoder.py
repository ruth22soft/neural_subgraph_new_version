import argparse
import csv
from itertools import combinations, permutations
import time
import os
import pickle
from collections import defaultdict
from queue import PriorityQueue
import random
import warnings

import networkx as nx
import numpy as np
import torch
import torch.multiprocessing as mp
from torch_geometric.datasets import TUDataset, PPI
from torch_geometric.data import DataLoader
import torch_geometric.utils as pyg_utils
from tqdm import tqdm
import matplotlib.pyplot as plt

from common import data
from common import models
from common import utils
from common import combined_syn
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

# =========================
# Graph Analysis for Streaming
# =========================
def analyze_graph_for_streaming(graph, args):
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    avg_degree = num_edges / num_nodes if num_nodes > 0 else 0

    # Avoid empty graph issues
    if num_nodes == 0:
        return {
            "use_streaming": False,
            "reason": "empty graph",
            "num_nodes": 0,
            "num_edges": 0,
            "avg_degree": 0,
            "clustering_coef": 0,
            "n_components": 0,
            "connectivity_ratio": 0,
            "is_bipartite": False,
            "is_power_law": False,
            "estimated_memory_mb": 0,
        }

    is_bipartite = nx.is_bipartite(graph)

    # Clustering coefficient
    if graph.is_directed():
        undirected_graph = graph.to_undirected()
        if num_nodes > 10000:
            sample_nodes = random.sample(list(undirected_graph.nodes()), 1000)
            clustering_coef = nx.average_clustering(undirected_graph, nodes=sample_nodes)
        else:
            clustering_coef = nx.average_clustering(undirected_graph)
    else:
        if num_nodes > 10000:
            sample_nodes = random.sample(list(graph.nodes()), 1000)
            clustering_coef = nx.average_clustering(graph, nodes=sample_nodes)
        else:
            clustering_coef = nx.average_clustering(graph)

    degrees = [d for n, d in graph.degree()]
    max_degree = max(degrees) if degrees else 0
    median_degree = sorted(degrees)[len(degrees) // 2] if degrees else 0
    is_power_law = (max_degree / (median_degree + 1)) > 10 if degrees else False

    # Connectivity ratio
    if graph.is_directed():
        components = list(nx.weakly_connected_components(graph))
    else:
        components = list(nx.connected_components(graph))
    n_components = len(components)
    connectivity_ratio = len(max(components, key=len)) / num_nodes if n_components > 0 else 0.0

    # Decision logic
    use_streaming = False
    reason = ""

    if is_bipartite:
        reason = "bipartite graph structure - BFS chunking ineffective"
    elif connectivity_ratio > 0.9:
        reason = f"well-connected graph (connectivity={connectivity_ratio:.2f}) - BFS chunking would cause memory issues"
    elif is_power_law:
        reason = "power-law degree distribution - hub nodes would cause imbalanced chunks"
    elif num_nodes > 100000 and 5.0 <= avg_degree <= 20.0 and clustering_coef > 0.3 and n_components < 100:
        use_streaming = True
        reason = f"large modular graph (degree={avg_degree:.2f}, clustering={clustering_coef:.3f})"
    else:
        reason = "graph characteristics don't benefit from chunking"

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

# =========================
# BFS Chunking
# =========================
def bfs_chunk(graph, start_node, max_size):
    visited = set([start_node])
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
    all_nodes = set(graph.nodes())
    graph_chunks = []
    while all_nodes:
        start_node = next(iter(all_nodes))
        chunk = bfs_chunk(graph, start_node, chunk_size)
        graph_chunks.append(chunk)
        all_nodes -= set(chunk.nodes())
    return graph_chunks

# =========================
# Pattern Growth (main mining)
# =========================
def pattern_growth(dataset, task, args):
    start_time = time.time()
    if args.method_type == "mlp":
        model = models.BaselineMLP(1, args.hidden_dim, args)
    elif args.method_type == "order":
        model = models.OrderEmbedder(1, args.hidden_dim, args)
    else:
        model = models.OrderEmbedder(1, args.hidden_dim, args)  # fallback

    device = utils.get_device()
    model.to(device)
    model.eval()
    if hasattr(args, "model_path") and args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))

    graphs = []
    for g in dataset:
        if not isinstance(g, (nx.Graph, nx.DiGraph)):
            g = pyg_utils.to_networkx(g).to_undirected()
        graphs.append(g)

    # Simplified placeholder: just return graphs as patterns
    discovered_patterns = graphs

    # Optional visualization
    if args.analyze:
        count_by_size = defaultdict(int)
        for pattern in discovered_patterns:
            visualize_pattern_graph_ext(pattern, args, count_by_size)

    # Save results
    os.makedirs("results", exist_ok=True)
    if hasattr(args, "out_path"):
        with open(args.out_path, "wb") as f:
            pickle.dump(discovered_patterns, f)

    elapsed = time.time() - start_time
    print(f"Pattern growth finished in {int(elapsed // 60)} mins {int(elapsed % 60)} secs")
    print(f"Discovered {len(discovered_patterns)} patterns")
    return discovered_patterns

# =========================
# Pattern Growth Streaming
# =========================
def _process_chunk(args_tuple):
    chunk_dataset, task, args, chunk_index, total_chunks = args_tuple
    start_time = time.time()
    original_n_workers = getattr(args, "n_workers", 4)
    args.n_workers = 0
    print(f"[Chunk {chunk_index+1}/{total_chunks}] PID {os.getpid()} started", flush=True)
    try:
        result = pattern_growth(chunk_dataset, task, args)
        elapsed = int(time.time() - start_time)
        print(f"[Chunk {chunk_index+1}/{total_chunks}] finished in {elapsed}s ({len(result)} patterns)", flush=True)
        args.n_workers = original_n_workers
        return result
    except Exception as e:
        print(f"[Chunk {chunk_index+1}] ERROR: {str(e)}", flush=True)
        import traceback; traceback.print_exc()
        args.n_workers = original_n_workers
        return []

def pattern_growth_streaming(dataset, task, args):
    graph = dataset[0]
    avg_degree = graph.number_of_edges() / graph.number_of_nodes() if graph.number_of_nodes() else 0
    effective_chunk_size = min(args.chunk_size, 5000 if avg_degree > 10 else args.chunk_size)
    graph_chunks = process_large_graph_in_chunks(graph, chunk_size=effective_chunk_size)
    print(f"Partitioned into {len(graph_chunks)} chunks, processing with {args.streaming_workers} workers...")
    chunk_args = [( [chunk], task, args, idx, len(graph_chunks) ) for idx, chunk in enumerate(graph_chunks)]
    with mp.Pool(processes=args.streaming_workers) as pool:
        results = pool.map(_process_chunk, chunk_args)
    all_discovered_patterns = []
    for chunk_out in results:
        if chunk_out:
            all_discovered_patterns.extend(chunk_out)
    print(f"Total patterns discovered: {len(all_discovered_patterns)}")
    return all_discovered_patterns

# =========================
# Main function
# =========================
def main():
    os.makedirs("plots/cluster", exist_ok=True)

    parser = argparse.ArgumentParser(description='Decoder arguments')
    parse_encoder(parser)
    parse_decoder(parser)
    args = parser.parse_args()

    # Load dataset
    if args.dataset.endswith(".pkl"):
        with open(args.dataset, "rb") as f:
            data_loaded = pickle.load(f)
            if isinstance(data_loaded, (nx.Graph, nx.DiGraph)):
                dataset = [data_loaded]
            elif isinstance(data_loaded, dict) and "nodes" in data_loaded:
                g = nx.DiGraph() if args.graph_type == "directed" else nx.Graph()
                g.add_nodes_from(data_loaded["nodes"])
                g.add_edges_from(data_loaded["edges"])
                dataset = [g]
            else:
                raise ValueError("Unknown dataset format")
        task = "graph"
    elif args.dataset.startswith("plant-"):
        size = int(args.dataset.split("-")[-1])
        dataset = combined_syn.get_generator([size]).generate(size=10)
        task = "graph"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Decide mode
    if len(dataset) == 1 and isinstance(dataset[0], (nx.Graph, nx.DiGraph)):
        graph = dataset[0]
        stats = analyze_graph_for_streaming(graph, args)
        print("="*60)
        print("GRAPH ANALYSIS")
        print("="*60)
        print(f"Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}")
        print(f"Connectivity ratio: {stats['connectivity_ratio']:.2f}, Reason: {stats['reason']}")
        print("="*60)
        if stats["use_streaming"]:
            out_graphs = pattern_growth_streaming(dataset, task, args)
        else:
            out_graphs = pattern_growth(dataset, task, args)
    else:
        out_graphs = pattern_growth(dataset, task, args)

if __name__ == "__main__":
    main()
