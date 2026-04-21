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
import pymetis

from utils import *
from gnn_model import GAT_NET
from partition import Partition

seed=2022
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# zx-数据集存储（实验环境下）
path_dataset=r'D:\test-code\GNN-PE-main\GNN-PE-main\offline\tangniaobin\\'
print(path_dataset)

num_partition=3
# zx-将图划分为5个部分
num_model=2
#zx-每个部分三个模型
path_length=2
#zx-路径长度为2
embedding_dim=2
#zx-嵌入维度为2

warmup_num=3

query_num=5
nodes_num=4
avg_degree=2
edges_num=int(nodes_num*avg_degree/2)

# zx-nx读取已存储的图数据
data_graph=nx.read_gpickle(path_dataset+'data_graph.gpickle.gz')
print(data_graph)

print("Graph Partition")
#zx-METIS 进行 图划分，划分为5个部分
adjacency_list = []
for node in data_graph.nodes():
    adjacency_list.append(np.array(list(data_graph.neighbors(node))))
n_cuts, membership = pymetis.part_graph(num_partition, adjacency=adjacency_list)
print('Edge Cuts: ',n_cuts)

# zx-partitions类对每个子图进行处理，提取并扩展子图后存储
partitions=[]
for i in range(num_partition):
    partition=Partition(path_dataset+'Partition-{0}'.format(i),num_model)
    nodes=np.argwhere(np.array(membership)==i).ravel()
    part=nx.subgraph(data_graph,nodes)
    print(part)
    print(nx.is_connected(part))
    nodes=set(part.nodes)
    for node in part.nodes:
        extend_nodes=list(nx.bfs_tree(data_graph,node,depth_limit=path_length))
        nodes=nodes|set(extend_nodes)
    nodes=list(nodes)
    part_extend=nx.subgraph(data_graph,nodes)
    print(part_extend)
    print(nx.is_connected(part_extend))
    partition.set_graph(part,part_extend)
    partitions.append(partition)
print('Partition Complete\n')

print('Datasets Generation')
# zx-将networkx格式的data_graph转化为pytorch_Geometric格式的图数据
G=pyg.utils.from_networkx(data_graph,group_node_attrs=['name'])

# zx-不同层次的子图存储
one_hop_subgraph=[]
zero_hop_subgraph=[]
one_hop_substructure=[[]for _ in range(G.num_nodes)]
substructure_num=0

# zx-生成0-hop子图和1-hop子图以及k-hop子图（pyg.utils.k_hop_subgraph）
for i in range(G.num_nodes):
    nodes,edges,node_map,_=pyg.utils.k_hop_subgraph(i,0,G.edge_index,relabel_nodes=True)
    subgraph_edge=[[]for _ in range(2)]

    subgraph_edge[0].append(0)
    subgraph_edge[1].append(0) 
    subgraph_edge=torch.tensor(subgraph_edge)
    subgraph=Data(x=G.x[nodes],edge_index=subgraph_edge,y=i)
    zero_hop_subgraph.append(subgraph)

    nodes,edges,node_map,_=pyg.utils.k_hop_subgraph(i,1,G.edge_index,relabel_nodes=True)
    anchor=node_map
    subgraph_edge=[[]for _ in range(2)]
    for j in range(len(edges[0])):
        u=edges[0][j]
        v=edges[1][j]
        if u==node_map or v==node_map:
            subgraph_edge[0].append(u)
            subgraph_edge[1].append(v) 
    subgraph_edge=torch.tensor(subgraph_edge)
    subgraph=Data(x=G.x[nodes],edge_index=subgraph_edge,y=i)
    one_hop_subgraph.append(subgraph)

    if data_graph.degree(i)>10:
        continue

    g=subgraph
    g_nx=pyg.utils.to_networkx(subgraph,node_attrs=['x'],to_undirected=True)
    nodes=torch.tensor(np.arange(0,g.num_nodes))
    neighbor=torch.gather(nodes,0,(nodes!=anchor).nonzero(as_tuple=True)[0])
    # zx-进一步生成不同大小的子结构用于图嵌入
    for j in range(1,len(neighbor)):
        substucture_nodes=torch.tensor(list(combinations(neighbor,j)))
        for k in range(len(substucture_nodes)):
            nodes=torch.cat((anchor,substucture_nodes[k]),0)
            substructure_nx=g_nx.subgraph(nodes.tolist()).copy()
            substructure=pyg.utils.from_networkx(substructure_nx,group_node_attrs=['x'])
            substructure.y=i
            one_hop_substructure[i].append(substructure)
            substructure_num+=1

print(len(one_hop_subgraph))
print(substructure_num)
print(len(zero_hop_subgraph))

for i in range(num_partition):
    one_hop_subgraphs=[]
    zero_hop_subgraphs=[]
    dataset=[]
    for j in list(partitions[i].graph_extend.nodes):
        one_hop_subgraphs.append(one_hop_subgraph[j])
        zero_hop_subgraphs.append(zero_hop_subgraph[j])
        for substructure in one_hop_substructure[j]:
            dataset.append((one_hop_subgraph[j],substructure))
    
    print('Partition-{0}: '.format(i))
    print(len(one_hop_subgraphs))
    print(len(zero_hop_subgraphs))
    print(len(dataset))
    partitions[i].set_dataset(one_hop_subgraphs,zero_hop_subgraphs,dataset)
    partitions[i].save_dataset()
    partitions[i].save_graph()
print('Datasets Generation Complete\n')


# zx-训练GNN（partition类）
# print('Start Training')
# for i in range(num_partition):
#     print('Partition-{0}: '.format(i))
#     partitions[i].train(batch_size=32)
# print('Training Complete\n')



print("Data Paths Generation")
all_paths=[]
target_list=list(data_graph.nodes)
for source in list(data_graph.nodes):
    path=bfs_all_paths(data_graph,source,target_list,path_length)
    all_paths.append(path)

for i in range(num_partition):
    part=Partition(path_dataset+'Partition-{0}'.format(i),num_model)
    part.load_graph()
    path=[]
    for node in list(part.graph.nodes):
        path.extend(all_paths[node])
    torch.save(path,path_dataset+'Partition-{0}'.format(i)+'/paths.pt')
print("Data Paths Generation Complete\n")


# zx-生成查询图：采样子图作为查询图，要求保持连接性和控制边数
print("Query Graph Generation")
query_graphs=[]
while len(query_graphs)!=query_num:
    subgraph_nodes=sample_subgraph(data_graph, nodes_num)
    subgraph=nx.subgraph(data_graph,subgraph_nodes).copy()
    
    if subgraph.number_of_edges()>=edges_num:
        while subgraph.number_of_edges()>edges_num:
            edges_list=list(subgraph.edges())
            (u,v)=random.sample(edges_list,1)[0]
            G=subgraph.copy()
            G.remove_edge(u,v)
            if nx.is_connected(G):
                subgraph.remove_edge(u,v)

        if nx.is_connected(subgraph):
            query_graphs.append(subgraph)
            print(len(query_graphs))

torch.save(query_graphs,path_dataset+'query_graphs.pt')

query_datasets=[]
query_zero_datasets=[]
for k in range(query_num):
    query_dataset=[]
    query_zero_dataset=[]
    query_graph_pyg=pyg.utils.from_networkx(query_graphs[k],group_node_attrs=['name'])

    for i in range(query_graph_pyg.num_nodes):
        nodes,edges,node_map,_=pyg.utils.k_hop_subgraph(i,1,query_graph_pyg.edge_index,relabel_nodes=True)
        subgraph_edge=[[]for i in range(2)]
        for j in range(len(edges[0])):
            u=edges[0][j]
            v=edges[1][j]
            if u==node_map or v==node_map:
                subgraph_edge[0].append(u)
                subgraph_edge[1].append(v) 
        subgraph_edge=torch.tensor(subgraph_edge)
        subgraph=Data(x=query_graph_pyg.x[nodes],edge_index=subgraph_edge,y=i)
        query_dataset.append(subgraph)

        nodes,edges,node_map,_=pyg.utils.k_hop_subgraph(i,0,query_graph_pyg.edge_index,relabel_nodes=True)
        subgraph_edge=[[]for i in range(2)]
        subgraph_edge[0].append(0)
        subgraph_edge[1].append(0) 
        subgraph_edge=torch.tensor(subgraph_edge)
        subgraph=Data(x=query_graph_pyg.x[nodes],edge_index=subgraph_edge,y=i)
        query_zero_dataset.append(subgraph)
    
    query_datasets.append(query_dataset)
    query_zero_datasets.append(query_zero_dataset)

# zx-计算查询图的嵌入：使用训练好的每个partition的GNN模型进行推理其嵌入
for i in range(num_partition):
    part=Partition(path_dataset+'Partition-{0}'.format(i),num_model)
    part.load_model()
    part.load_graph()
    for j in range(num_model):
        model=part.models[j]
        model.eval()
        model.cuda()

        if j!=0:
            hash_min=int(1)
            hash_max=int((part.graph.number_of_nodes()/(num_model-1)*j)+1)
            print('The hash range is {0}-{1}'.format(hash_min,hash_max))

        eval_time=[]
        zero_eval_time=[]

        one_hop_embeddings=[]
        zero_hop_embeddings=[]
        for k in range(query_num):
            query_dataset=query_datasets[k]
            query_zero_dataset=query_zero_datasets[k]

            query_dataloader=DataLoader(query_dataset,batch_size=len(query_dataset),shuffle=False)
            query_zero_dataloader=DataLoader(query_zero_dataset,batch_size=len(query_zero_dataset),shuffle=False)

            with torch.no_grad():
                for idx,data in enumerate(query_dataloader):
                    data1=copy.deepcopy(data)
                    if j!=0:
                        for l in range(len(data1.x)):
                            random.seed(data1.x[l].item())
                            data1.x[l]=random.randint(hash_min,hash_max)

                    data1.x=data1.x.float()
                    data1=data1.cuda()

                    for _ in range(warmup_num):
                        embedding=model(data1.x.float(),data1.edge_index,data1.batch)

                    stime=time.time()
                    embedding=model(data1.x.float(),data1.edge_index,data1.batch)
                    etime=time.time()
                    eval_time.append((etime-stime)*1000)

                    one_hop_embeddings.append(embedding)
                
                for idx,data in enumerate(query_zero_dataloader):
                    data1=copy.deepcopy(data)
                    if j!=0:
                        for l in range(len(data1.x)):
                            random.seed(data1.x[l].item())
                            data1.x[l]=random.randint(hash_min,hash_max)

                    data1.x=data1.x.float()
                    data1=data1.cuda()

                    for _ in range(warmup_num):
                        embedding=model(data1.x.float(),data1.edge_index,data1.batch)

                    stime=time.time()
                    embedding=model(data1.x.float(),data1.edge_index,data1.batch)
                    etime=time.time()
                    zero_eval_time.append((etime-stime)*1000)

                    zero_hop_embeddings.append(embedding)

        torch.save(one_hop_embeddings,path_dataset+'Partition-{0}'.format(i)+'/Model-{0}'.format(j)+'/query_embeddings.pt')
        torch.save(eval_time,path_dataset+'Partition-{0}'.format(i)+'/Model-{0}'.format(j)+'/query_embedding_times.pt')    
        torch.save(zero_hop_embeddings,path_dataset+'Partition-{0}'.format(i)+'/Model-{0}'.format(j)+'/query_zero_embeddings.pt')
        torch.save(zero_eval_time,path_dataset+'Partition-{0}'.format(i)+'/Model-{0}'.format(j)+'/query_zero_embedding_times.pt')

print("Query Graph Generation Complete\n")



print("Output Online Data")
path_output=path_dataset+'Output'
create_folder(path_output)

data_graph_pyg=pyg.utils.from_networkx(data_graph,group_node_attrs=['name'])
filename=path_output+'/data_graph.txt'
with open(filename,'w') as file_object:
    file_object.write('t '+str(data_graph.number_of_nodes())+' '+str(data_graph.number_of_edges())+'\n')
    for i in range(data_graph.number_of_nodes()):
        file_object.write('v '+str(i)+' '+str(data_graph_pyg.x[i].item())+' '+str(data_graph.degree[i])+'\n')
    for (u,v) in data_graph.edges:
        file_object.write('e '+str(u)+' '+str(v)+'\n')

create_folder(path_output+'/query_graphs')
query_graphs=torch.load(path_dataset+'query_graphs.pt')
for i in range(query_num):
    query_graph=query_graphs[i]
    query_graph_pyg=pyg.utils.from_networkx(query_graph,group_node_attrs=['name'])
    filename=path_output+'/query_graphs'+'/query_graph-{0}.txt'.format(i)
    filename_map=path_output+'/query_graphs'+'/query_graph_map-{0}.txt'.format(i)
    map={}
    original_nodes=list(query_graph.nodes)
    with open(filename_map,'w') as file_object:
        file_object.write(str(query_graph.number_of_nodes())+'\n')
        for j in range(query_graph.number_of_nodes()):
            file_object.write(str(j)+' '+str(original_nodes[j])+'\n')
            map[original_nodes[j]]=j
    with open(filename,'w') as file_object:
        file_object.write('t '+str(query_graph.number_of_nodes())+' '+str(query_graph.number_of_edges())+'\n')
        for j in range(query_graph.number_of_nodes()):
            node=original_nodes[j]
            file_object.write('v '+str(j)+' '+str(query_graph_pyg.x[j].item())+' '+str(query_graph.degree[node])+'\n')
        for (u,v) in query_graph.edges:
            u_=map[u]
            v_=map[v]
            file_object.write('e '+str(u_)+' '+str(v_)+'\n')

# zx-输出最终的查询数据、嵌入、以及图划分信息
for i in range(num_partition):
    print('Partition-{0}'.format(i))
    create_folder(path_output+'/Partition-{0}'.format(i))
    part=Partition(path_dataset+'Partition-{0}'.format(i),num_model)
    part.output_data(data_graph,path_output+'/Partition-{0}'.format(i),embedding_dim,path_length)
print("Output Online Data Complete\n")



