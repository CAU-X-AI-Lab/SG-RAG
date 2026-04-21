import torch
import torch.optim as optim
import torch.nn as nn
import torch_geometric as pyg

#GAT网络 单层
class GAT_NET(nn.Module):
    def __init__(self,seed,heads,input_dim,hidden_dim,output_dim):
        super(GAT_NET, self).__init__()
        torch.cuda.manual_seed_all(seed)
        self.conv1=pyg.nn.GATConv(input_dim,hidden_dim,heads=heads,concat=True)
        self.lin = nn.Linear(hidden_dim*heads, output_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim*heads)

    def forward(self,x,edge_index,batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = pyg.nn.global_add_pool(x, batch)
        x = torch.sigmoid(self.lin(x))
        return x

    def pos_loss(self,pos_a,pos_b):
        pos_e= torch.sum(torch.max(torch.zeros_like(pos_a), pos_b - pos_a)**2, dim=1)
        return torch.sum(pos_e)

#GIN网络 单层
class GIN_NET(nn.Module):
    def __init__(self,seed,input_dim,hidden_dim,output_dim):
        super(GIN_NET, self).__init__()
        nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
            )
        self.conv1 = pyg.nn.GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(output_dim)

    def forward(self,x,edge_index,batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = pyg.nn.global_add_pool(x, batch)
        return x
    
    def pos_loss(self,pos_a,pos_b):
        pos_e= torch.sum(torch.max(torch.zeros_like(pos_a), pos_b - pos_a)**2, dim=1)
        return torch.sum(pos_e)