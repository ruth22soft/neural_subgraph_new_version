import argparse
import csv
from itertools import combinations
import time
import os
import pickle

from deepsnap.batch import Batch
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.datasets import TUDataset, PPI
from torch_geometric.datasets import Planetoid, KarateClub, QM7b
from torch_geometric.data import DataLoader
import torch_geometric.utils as pyg_utils

import torch_geometric.nn as pyg_nn
from matplotlib import cm

from common import data
from common import models
from common import utils
from common import combined_syn
from subgraph_mining.config import parse_decoder
from subgraph_matching.config import parse_encoder
from subgraph_mining.search_agents import GreedySearchAgent, MCTSSearchAgent, MemoryEfficientMCTSAgent, MemoryEfficientGreedyAgent, BeamSearchAgent

import matplotlib.pyplot as plt
import random
from scipy.io import mmread
import scipy.stats as stats
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from collections import defaultdict
from itertools import permutations
from queue import PriorityQueue
import matplotlib.colors as mcolors
import networkx as nx
import pickle
import torch.multiprocessing as mp
from sklearn.decomposition import PCA
import warnings

# --------------------------- Graph Analysis ---------------------------

def analyze_graph_for_streaming(graph, args):  
    import random  
      
    num_nodes = graph.number_of_nodes()  
    num_edges = graph.number_of_edges()  
    avg_degree = num_edges / num_nodes if num_nodes > 0 else 0  
      
    # Bipartite detection 
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
      
    # Power-law degree distribution check  
    degrees = [d for n, d in graph.degree()]  
    if degrees:  
        max_degree = max(degrees)  
        median_degree = sorted(degrees)[len(degrees) // 2]  
        is_power_law = (max_degree / (median_degree + 1)) > 10  
    else:  
        is_power_law = False  
      
    # Calculate connectivity ratio
    if graph.is_directed():  
        components = list(nx.weakly_connected_components(graph))  
    else:  
        components = list(nx.connected_components(graph))  
      
    n_components = len(components)  
    if n_components > 0:  
        largest_cc_size = len(max(components, key=len))  
        connectivity_ratio = largest_cc_size / num_nodes  
    else:  
        connectivity_ratio = 0.0  
      
    # Decision logic  
    use_streaming = False  
    reason = ""  
      
    if is_bipartite:  
        use_streaming = False  
        reason = "bipartite graph structure - BFS chunking ineffective"  
    elif connectivity_ratio > 0.9:  
        use_streaming = False  
        reason = f"well-connected graph (connectivity={connectivity_ratio:.2f}) - BFS chunking would cause memory issues"  
    elif is_power_law:
        use_streaming = False
        reason = "power-law degree distribution - hub nodes would cause imbalanced chunks"
    elif (num_nodes > 100000 and 5.0 <= avg_degree <= 20.0 and clustering_coef > 0.3 and n_components < 100):
        use_streaming = True
        reason = f"large modular graph (degree={avg_degree:.2f}, clustering={clustering_coef:.3f})"
    else:
        use_streaming = False
        reason = f"graph characteristics don't benefit from chunking"

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

# --------------------------- Chunk Processing ---------------------------

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

# --------------------------- Pattern Growth ---------------------------

def _process_chunk(args_tuple):
    chunk_dataset, task, args, chunk_index, total_chunks = args_tuple
    start_time = time.time()
    original_n_workers = getattr(args, "n_workers", 4)
    args.n_workers = 0  # single-threaded
    print(f"[{time.strftime('%H:%M:%S')}] Worker PID {os.getpid()} started chunk {chunk_index+1}/{total_chunks}", flush=True)
    try:
        result = pattern_growth(chunk_dataset, task, args)
        elapsed = int(time.time() - start_time)
        print(f"[{time.strftime('%H:%M:%S')}] Worker PID {os.getpid()} finished chunk {chunk_index+1}/{total_chunks} in {elapsed}s ({len(result)} patterns)", flush=True)
        args.n_workers = original_n_workers
        return result
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ERROR in chunk {chunk_index+1}: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        args.n_workers = original_n_workers
        return []

def pattern_growth_streaming(dataset, task, args):
    graph = dataset[0]
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    avg_degree = num_edges / num_nodes if num_nodes > 0 else 0
    print(f"Graph statistics: {num_nodes} nodes, {num_edges} edges, avg degree: {avg_degree:.2f}", flush=True)

    if avg_degree > args.dense_graph_threshold:
        effective_chunk_size = min(args.chunk_size, 5000)
    elif avg_degree > 20:
        effective_chunk_size = min(args.chunk_size, 7500)
    else:
        effective_chunk_size = args.chunk_size

    graph_chunks = process_large_graph_in_chunks(graph, chunk_size=effective_chunk_size)
    min_chunk_size = max(args.min_pattern_size, 5)
    graph_chunks = [chunk for chunk in graph_chunks if chunk.number_of_nodes() >= min_chunk_size]
    print(f"Created {len(graph_chunks)} chunks", flush=True)

    all_discovered_patterns = []
    total_chunks = len(graph_chunks)
    chunk_args = [([chunk], task, args, idx, total_chunks) for idx, chunk in enumerate(graph_chunks)]
    with mp.Pool(processes=args.streaming_workers) as pool:
        results = pool.map(_process_chunk, chunk_args)
    for chunk_out_graphs in results:
        if chunk_out_graphs:
            all_discovered_patterns.extend(chunk_out_graphs)
    print(f"Total patterns discovered: {len(all_discovered_patterns)}", flush=True)
    return all_discovered_patterns

# --------------------------- Visualization ---------------------------

def visualize_pattern_graph_ext(pattern, args, count_by_size):
    """
    Save pattern visualization into plots/cluster
    """
    try:
        output_folder = "plots/cluster"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        num_nodes = len(pattern)
        num_edges = pattern.number_of_edges()
        edge_density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0

        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(pattern, seed=42, k=0.5, iterations=50)

        labels = nx.get_node_attributes(pattern, 'label')
        unique_labels = sorted(set(labels.values())) if labels else [str(i) for i in pattern.nodes()]
        color_map = {lbl: cm.tab20(i % 20) for i, lbl in enumerate(unique_labels)}
        node_colors = [color_map.get(labels.get(n, str(n)), 'grey') for n in pattern.nodes()]
        node_sizes = [3000 if pattern.nodes[n].get('anchor', 0) == 1 else 1500 for n in pattern.nodes()]

        nx.draw_networkx_nodes(pattern, pos, node_color=node_colors, node_size=node_sizes)
        nx.draw_networkx_edges(pattern, pos, alpha=0.7)
        nx.draw_networkx_labels(pattern, pos, labels=labels, font_size=10)

        count_by_size[num_nodes] += 1
        filename = f"{num_nodes}_{count_by_size[num_nodes]}.png"
        filepath = os.path.join(output_folder, filename)

        plt.title(f"Pattern: {num_nodes} nodes, {num_edges} edges")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"[INFO] Saved pattern visualization: {filepath}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to visualize pattern: {e}")
        return False

# --------------------------- Main Pattern Growth Function ---------------------------

def pattern_growth(dataset, task, args):
    start_time = time.time()
    if args.method_type == "end2end":
        model = models.End2EndOrder(1, args.hidden_dim, args)
    elif args.method_type == "mlp":
        model = models.BaselineMLP(1, args.hidden_dim, args)
    else:
        model = models.OrderEmbedder(1, args.hidden_dim, args)
    model.to(utils.get_device())
    model.eval()
    model.load_state_dict(torch.load(args.model_path, map_location=utils.get_device()))

    if task == "graph-labeled":
        dataset, labels = dataset

    graphs = []
    for i, graph in enumerate(dataset):
        if task == "graph-labeled" and labels[i] != 0: continue
        if task == "graph-truncate" and i >= 1000: break
        if not isinstance(graph, (nx.Graph, nx.DiGraph)):
            graph = pyg_utils.to_networkx(graph).to_undirected()
            for node in graph.nodes():
                graph.nodes[node].setdefault('label', str(node))
                graph.nodes[node].setdefault('id', str(node))
        graphs.append(graph)

    # For simplicity, just use the graphs directly
    embs = []
    anchors = [0]*len(graphs)
    count_by_size = defaultdict(int)
    out_graphs = graphs  # In original code, agent generates patterns; here we use graphs

    # Visualize patterns
    for pattern in out_graphs:
        visualize_pattern_graph_ext(pattern, args, count_by_size)

    # Save results
    if not os.path.exists("results"):
        os.makedirs("results")
    with open(args.out_path, "wb") as f:
        pickle.dump(out_graphs, f)
    return out_graphs

# --------------------------- Main Entry Point ---------------------------

def main():
    if not os.path.exists("plots/cluster"):
        os.makedirs("plots/cluster")

    parser = argparse.ArgumentParser(description='Decoder arguments')
    parse_encoder(parser)
    parse_decoder(parser)
    args = parser.parse_args()

    print("Using dataset {}".format(args.dataset))
    print("Graph type: {}".format(args.graph_type))

    # Load dataset (pickle or TUDataset)
    if args.dataset.endswith('.pkl'):
        with open(args.dataset, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, (nx.Graph, nx.DiGraph)):
            graph = data
            if args.graph_type == "directed" and not graph.is_directed():
                graph = graph.to_directed()
            elif args.graph_type == "undirected" and graph.is_directed():
                graph = graph.to_undirected()
        elif isinstance(data, dict) and 'nodes' in data and 'edges' in data:
            graph = nx.DiGraph() if args.graph_type == "directed" else nx.Graph()
            graph.add_nodes_from(data['nodes'])
            graph.add_edges_from(data['edges'])
        else:
            raise ValueError("Unknown dataset format")
        dataset = [graph]
        task = 'graph'

    # Decide streaming or standard
    if len(dataset) == 1 and isinstance(dataset[0], (nx.Graph, nx.DiGraph)):
        graph = dataset[0]
        stats = analyze_graph_for_streaming(graph, args)
        use_streaming = stats['use_streaming']
        print("="*60)
        print(f"Decision: {'STREAMING MODE' if use_streaming else 'STANDARD MODE'}")
        print(f"Reason: {stats['reason']}")
        print("="*60)
        if use_streaming:
            out_graphs = pattern_growth_streaming(dataset, task, args)
        else:
            out_graphs = pattern_growth(dataset, task, args)
    else:
        out_graphs = pattern_growth(dataset, task, args)

if __name__ == '__main__':
    main()
