import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch_geometric as pyg
from torch_geometric.data import Data
from itertools import combinations
from torch_geometric.loader import DataLoader
import os
import time
import random
import copy
import pickle

def draw_graph(g):
    nx.draw(g,pos=nx.spring_layout(g, seed=2022),with_labels=True)
    plt.show

def draw_pyg_graph(g):
    g=pyg.utils.to_networkx(g,node_attrs=['x'])
    pos=nx.spring_layout(g, seed=2022)
    nx.draw(g, pos=pos,with_labels=False)
    node_labels=nx.get_node_attributes(g, 'x')
    nx.draw_networkx_labels(g,pos=pos,labels=node_labels)
    plt.show()

def create_folder(filename):
    filename = filename.strip()
    filename = filename.rstrip("\\")
    isExists = os.path.exists(filename)

    if not isExists:
        os.makedirs(filename)
        print(filename+" Create Successful")
        return  True
    else:
        print(filename+" already exists")
        return False

def bfs_all_paths(G, source, target_list, length):
    all_paths = []
    queue = [[source]]
    while queue:
        path = queue.pop(0)
        if len(path) == length + 1:
            if path[-1] in target_list:
                all_paths.append(path)
        elif len(path) < length + 1:
            for neighbor in G[path[-1]]:
                if neighbor not in path:
                    new_path = path + [neighbor]
                    queue.append(new_path)
    return all_paths

def sample_subgraph(G, num_nodes):
    subgraph_nodes=[]
    start_node = random.choice(list(G.nodes()))
    subgraph = nx.Graph()
    subgraph.add_node(start_node)
    subgraph_nodes.append(start_node)
    while len(subgraph) < num_nodes:
        neighbors = set()
        for node in subgraph.nodes():
            neighbors.update(list(G.neighbors(node)))
        candidate_nodes = list(neighbors - set(subgraph.nodes()))
        if not candidate_nodes:
            break
        new_node = random.choice(candidate_nodes)
        subgraph.add_node(new_node)
        subgraph_nodes.append(new_node)
        for neighbor in G.neighbors(new_node):
            if neighbor in subgraph:
                subgraph.add_edge(new_node, neighbor)
    if len(subgraph) < num_nodes:
        return sample_subgraph(G, num_nodes)
    else:
        return subgraph_nodes
