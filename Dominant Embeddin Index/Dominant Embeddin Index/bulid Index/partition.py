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

seed=2022
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class Partition():
    def __init__(self,path_part,num_model):
        self.path=path_part
        self.num_model=num_model
        self.models=[]

        create_folder(self.path)
        for i in range(self.num_model):
            create_folder(self.path+'/Model-{0}'.format(i))

        for _ in range(self.num_model):
            self.models.append(GAT_NET(seed=1,heads=3,input_dim=1,hidden_dim=32,output_dim=2))

    def set_graph(self,graph,graph_extend):
        self.graph=graph
        self.graph_extend=graph_extend

    def save_graph(self):
        nx.write_gpickle(self.graph,self.path+'/graph.gpickle.gz')
        nx.write_gpickle(self.graph_extend,self.path+'/graph_extend.gpickle.gz')

    def load_graph(self):
        self.graph=nx.read_gpickle(self.path+'/graph.gpickle.gz')
        self.graph_extend=nx.read_gpickle(self.path+'/graph_extend.gpickle.gz')

    def set_dataset(self,one_subgraphs,zero_subgraphs,dataset):
        self.one_subgraphs=one_subgraphs
        self.zero_subgraphs=zero_subgraphs
        self.dataset=dataset

    def save_dataset(self):
        torch.save(self.one_subgraphs,self.path+'/one_subgraphs.pt')
        torch.save(self.zero_subgraphs,self.path+'/zero_subgraphs.pt')
        torch.save(self.dataset,self.path+'/dataset.pt')

    def load_dataset(self):
        self.one_subgraphs=torch.load(self.path+'/one_subgraphs.pt')
        self.zero_subgraphs=torch.load(self.path+'/zero_subgraphs.pt')
        self.dataset=torch.load(self.path+'/dataset.pt')

    def load_model(self):
        self.models=[]
        for i in range(self.num_model):
            model=torch.load(self.path+'/Model-{0}'.format(i)+'/model.pth')
            self.models.append(model)

    def save_model(self):
        for i in range(self.num_model):
            torch.save(self.models[i],self.path+'/Model-{0}'.format(i)+'model.pth')

    def output_data(self,data_graph,path_output,embedding_dim,path_length):
        self.load_graph()
        graph_pyg=pyg.utils.from_networkx(self.graph,group_node_attrs=['name'])
        graph_extend_pyg=pyg.utils.from_networkx(self.graph_extend,group_node_attrs=['name'])
        map={}
        extend_map={}
        original_nodes=list(self.graph.nodes)
        original_extend_nodes=list(self.graph_extend.nodes)

        part_filename=path_output+'/graph.txt'
        part_extend_filename=path_output+'/graph_extend.txt'
        part_filename_map=path_output+'/graph_map.txt'
        part_extend_filename_map=path_output+'/graph_extend_map.txt'

        with open(part_filename_map,'w') as f:
            f.write(str(self.graph.number_of_nodes())+'\n')
            for j in range(self.graph.number_of_nodes()):
                f.write(str(j)+' '+str(original_nodes[j])+'\n')
                map[original_nodes[j]]=j
        with open(part_extend_filename_map,'w') as f:
            f.write(str(self.graph_extend.number_of_nodes())+'\n')
            for j in range(self.graph_extend.number_of_nodes()):
                f.write(str(j)+' '+str(original_extend_nodes[j])+'\n')
                extend_map[original_extend_nodes[j]]=j

        with open(part_filename,'w') as f:
            f.write('t '+str(self.graph.number_of_nodes())+' '+str(self.graph.number_of_edges())+'\n')
            for j in range(self.graph.number_of_nodes()):
                node=original_nodes[j]
                f.write('v '+str(j)+' '+str(graph_pyg.x[j].item())+' '+str(self.graph.degree[node])+'\n')
            for (u,v) in self.graph.edges:
                u_=map[u]
                v_=map[v]
                f.write('e '+str(u_)+' '+str(v_)+'\n')

        with open(part_extend_filename,'w') as f:
            f.write('t '+str(self.graph_extend.number_of_nodes())+' '+str(self.graph_extend.number_of_edges())+'\n')
            for j in range(self.graph_extend.number_of_nodes()):
                node=original_extend_nodes[j]
                f.write('v '+str(j)+' '+str(graph_extend_pyg.x[j].item())+' '+str(self.graph_extend.degree[node])+'\n')
            for (u,v) in self.graph_extend.edges:
                u_=extend_map[u]
                v_=extend_map[v]
                f.write('e '+str(u_)+' '+str(v_)+'\n')

        paths=torch.load(self.path+'/paths.pt')
        with open(path_output+'/paths.txt','w') as f:
            f.write(str(len(paths))+' '+str(path_length+1)+'\n')
            for i in range(len(paths)):
                for j in range(path_length+1):
                    f.write(str(extend_map[paths[i][j]])+' ')
                f.write('\n')

        for i in range(self.num_model):
            create_folder(path_output+'/Model-{0}'.format(i))
            input_path=self.path+'/Model-{0}/'.format(i)
            output_path=path_output+'/Model-{0}/'.format(i)

            one_embeddings=torch.load(input_path+'one_embeddings.pt')
            with open(output_path+'one_embeddings.txt','w') as f:
                f.write(str(len(one_embeddings))+' '+str(embedding_dim)+'\n')
                for j in range(len(one_embeddings)):
                    if data_graph.degree(extend_map[original_extend_nodes[j]])>10:
                        for k in range(embedding_dim):
                            f.write(str(1.0)+' ')
                    else:
                        for k in range(embedding_dim):
                            f.write(str(one_embeddings[j][k].item())+' ')
                    f.write('\n')

            zero_embeddings=torch.load(input_path+'zero_embeddings.pt')
            with open(output_path+'zero_embeddings.txt','w') as f:
                f.write(str(len(zero_embeddings))+' '+str(embedding_dim)+'\n')
                for j in range(len(zero_embeddings)):
                    for k in range(embedding_dim):
                        f.write(str(zero_embeddings[j][k].item())+' ')
                    f.write('\n')

            query_embedding_times=torch.load(input_path+'query_embedding_times.pt')
            with open(output_path+'query_embedding_times.txt','w') as f:
                f.write(str(len(query_embedding_times))+' '+str(embedding_dim)+'\n')
                for j in range(len(query_embedding_times)):
                    f.write(str(query_embedding_times[j])+'\n')

            query_zero_embedding_times=torch.load(input_path+'query_zero_embedding_times.pt')
            with open(output_path+'query_zero_embedding_times.txt','w') as f:
                f.write(str(len(query_zero_embedding_times))+' '+str(embedding_dim)+'\n')
                for j in range(len(query_zero_embedding_times)):
                    f.write(str(query_zero_embedding_times[j])+'\n')

            create_folder(output_path+'query_embeddings')
            output_path=output_path+'query_embeddings/'

            query_embeddings=torch.load(input_path+'query_embeddings.pt')
            for j in range(len(query_embeddings)):
                with open(output_path+'query_embeddings-{0}.txt'.format(j),'w') as f:
                    f.write(str(len(query_embeddings[j]))+' '+str(embedding_dim)+'\n')
                    for k in range(len(query_embeddings[j])):
                        for l in range(embedding_dim):
                            f.write(str(query_embeddings[j][k][l].item())+' ')
                        f.write('\n')

            query_zero_embeddings=torch.load(input_path+'query_zero_embeddings.pt')
            for j in range(len(query_zero_embeddings)):
                with open(output_path+'query_zero_embeddings-{0}.txt'.format(j),'w') as f:
                    f.write(str(len(query_zero_embeddings[j]))+' '+str(embedding_dim)+'\n')
                    for k in range(len(query_zero_embeddings[j])):
                        for l in range(embedding_dim):
                            f.write(str(query_zero_embeddings[j][k][l].item())+' ')
                        f.write('\n')

    def train(self,batch_size):
        dataloader=DataLoader(self.dataset,batch_size=batch_size,shuffle=True)

        for i in range(self.num_model):
            print('model ',i,' start training')
            if i!=0:
                hash_min=int(1)
                hash_max=int((self.graph.number_of_nodes()/(self.num_model-1)*i)+1)
                print('The hash range is {0}-{1}'.format(hash_min,hash_max))

            model=self.models[i]
            model.cuda()
            model.train()
            optimizer=optim.Adam(model.parameters(),lr=0.001)
            start_time=time.time()
            last_time=start_time

            for epoch in range(1,2):
                loss_all=0
                model.train()
                for idx,data in enumerate(dataloader):
                    optimizer.zero_grad()
                    data1=copy.deepcopy(data)

                    if i!=0:
                        for j in range(len(data1[0].x)):
                            random.seed(data1[0].x[j].item())
                            data1[0].x[j]=random.randint(hash_min,hash_max)

                        for j in range(len(data1[1].x)):
                            random.seed(data1[1].x[j].item())
                            data1[1].x[j]=random.randint(hash_min,hash_max)

                    data1[0]=data1[0].cuda()
                    data1[1]=data1[1].cuda()

                    pos_a1=model(data1[0].x.float(),data1[0].edge_index.long(),data1[0].batch)
                    pos_b1=model(data1[1].x.float(),data1[1].edge_index.long(),data1[1].batch)

                    loss1=model.pos_loss(pos_a1,pos_b1)

                    loss1.backward()
                    optimizer.step()

                with torch.no_grad():
                    model.eval()
                    for idx,data in enumerate(dataloader):
                        data1=copy.deepcopy(data)

                        if i!=0:
                            for j in range(len(data1[0].x)):
                                random.seed(data1[0].x[j].item())
                                data1[0].x[j]=random.randint(hash_min,hash_max)

                            for j in range(len(data1[1].x)):
                                random.seed(data1[1].x[j].item())
                                data1[1].x[j]=random.randint(hash_min,hash_max)

                        data1[0]=data1[0].cuda()
                        data1[1]=data1[1].cuda()

                        pos_a1=model(data1[0].x.float(),data1[0].edge_index.long(),data1[0].batch)
                        pos_b1=model(data1[1].x.float(),data1[1].edge_index.long(),data1[1].batch)

                        loss1=model.pos_loss(pos_a1,pos_b1)
                        loss_all+=loss1.item()

                end_time=time.time()
                print(
                    'Epoch: ',epoch,
                    ' Loss: ',loss_all,
                    ' Time (s): ',(end_time-last_time))
                last_time=end_time

                if loss_all==0:
                    break

            print('Time (hour): ',(end_time-start_time)/3600)
            torch.save(model,self.path+'/Model-{0}'.format(i)+'/model.pth')

            model.eval()
            model.cuda()
            print('model ',i,' generate embeddings')
            one_dataloader=DataLoader(self.one_subgraphs,batch_size=batch_size,shuffle=False)
            zero_dataloader=DataLoader(self.zero_subgraphs,batch_size=batch_size,shuffle=False)

            start_time=time.time()
            for idx,data in enumerate(one_dataloader):
                data1=copy.deepcopy(data)

                if i!=0:
                    for j in range(len(data1.x)):
                        random.seed(data1.x[j].item())
                        data1.x[j]=random.randint(hash_min,hash_max)

                data1=data1.cuda()
                emb1=model(data1.x.float(),data1.edge_index.long(),data1.batch)
                if idx==0:
                    one_embeddings=emb1.cpu().detach()
                else:
                    one_embeddings=torch.cat([one_embeddings,emb1.cpu().detach()],0)

            for idx,data in enumerate(zero_dataloader):
                data1=copy.deepcopy(data)

                if i!=0:
                    for j in range(len(data1.x)):
                        random.seed(data1.x[j].item())
                        data1.x[j]=random.randint(hash_min,hash_max)

                data1=data1.cuda()
                emb1=model(data1.x.float(),data1.edge_index.long(),data1.batch)
                if idx==0:
                    zero_embeddings=emb1.cpu().detach()
                else:
                    zero_embeddings=torch.cat([zero_embeddings,emb1.cpu().detach()],0)
            end_time=time.time()
            print(' Time (s): ',(end_time-last_time))

            torch.save(one_embeddings,self.path+'/Model-{0}'.format(i)+'/one_embeddings.pt')
            torch.save(zero_embeddings,self.path+'/Model-{0}'.format(i)+'/zero_embeddings.pt')

    def test(self,batch_size):
        one_dataloader=DataLoader(self.one_subgraphs,batch_size=batch_size,shuffle=False)
        zero_dataloader=DataLoader(self.zero_subgraphs,batch_size=batch_size,shuffle=False)
        for i in range(self.num_model):
            model=self.models[i]
            model.eval()
            model.cuda()
            print('model ',i,' generate embeddings')
            
            if i!=0:
                hash_min=int(1)
                hash_max=int((self.graph.number_of_nodes()/(self.num_model-1)*i)+1)
                print('The hash range is {0}-{1}'.format(hash_min,hash_max))

            with torch.no_grad():
                for idx,data in enumerate(one_dataloader):
                    data1=copy.deepcopy(data)

                    if i!=0:
                        for j in range(len(data1.x)):
                            random.seed(data1.x[j].item())
                            data1.x[j]=random.randint(hash_min,hash_max)

                    data1=data1.cuda()
                    emb1=model(data1.x.float(),data1.edge_index,data1.batch)
                    if idx==0:
                        one_embeddings=emb1
                    else:
                        one_embeddings=torch.cat([one_embeddings,emb1],0)

                for idx,data in enumerate(zero_dataloader):
                    data1=copy.deepcopy(data)

                    if i!=0:
                        for j in range(len(data1.x)):
                            random.seed(data1.x[j].item())
                            data1.x[j]=random.randint(hash_min,hash_max)

                    data1=data1.cuda()
                    emb1=model(data1.x.float(),data1.edge_index,data1.batch)
                    if idx==0:
                        zero_embeddings=emb1
                    else:
                        zero_embeddings=torch.cat([zero_embeddings,emb1],0)

            torch.save(one_embeddings,self.path+'/Model-{0}'.format(i)+'/one_embeddings.pt')
            torch.save(zero_embeddings,self.path+'/Model-{0}'.format(i)+'/zero_embeddings.pt')
            

