============================================================

MODELS

============================================================

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

============================================================

1️⃣ GAT ENCODER (Graph Attention Network)

============================================================

class GATEncoder(torch.nn.Module):
"""Three-layer Graph Attention Network encoder with multi-head attention."""

def __init__(self, in_dim, hidden, latent_dim, heads=4):  
    super().__init__()  
    self.gat1 = GATConv(in_dim, hidden, heads=heads, concat=True)  
    self.gat2 = GATConv(hidden * heads, hidden, heads=1, concat=False)  
    self.gat3 = GATConv(hidden, latent_dim, heads=1, concat=False)  
    self.dropout = torch.nn.Dropout(0.3)  

def forward(self, data):  
    x, edge_index = data.x, data.edge_index  
    x, _ = self.gat1(x, edge_index, return_attention_weights=True)  
    x = F.elu(x)  
    x = self.dropout(x)  
    x, _ = self.gat2(x, edge_index, return_attention_weights=True)  
    x = F.elu(x)  
    x = self.dropout(x)  
    x = self.gat3(x, edge_index)  
    return x

============================================================

2️⃣ GCN ENCODER (Graph Convolutional Network)

============================================================

class GCNEncoder(torch.nn.Module):
"""Three-layer Graph Convolutional Network encoder."""

def __init__(self, in_dim, hidden, latent_dim):  
    super().__init__()  
    self.conv1 = GCNConv(in_dim, hidden)  
    self.conv2 = GCNConv(hidden, hidden)  
    self.conv3 = GCNConv(hidden, latent_dim)  
    self.dropout = torch.nn.Dropout(0.3)  

def forward(self, data):  
    x, edge_index = data.x, data.edge_index  
    x = F.relu(self.conv1(x, edge_index))  
    x = self.dropout(x)  
    x = F.relu(self.conv2(x, edge_index))  
    x = self.dropout(x)  
    x = self.conv3(x, edge_index)  
    return x

============================================================

3️⃣ GRAPHSAGE ENCODER (Inductive Graph Network)

============================================================

class GraphSAGEEncoder(torch.nn.Module):
"""Three-layer GraphSAGE encoder with neighbor sampling aggregation."""

def __init__(self, in_dim, hidden, latent_dim):  
    super().__init__()  
    self.conv1 = SAGEConv(in_dim, hidden)  
    self.conv2 = SAGEConv(hidden, hidden)  
    self.conv3 = SAGEConv(hidden, latent_dim)  
    self.dropout = torch.nn.Dropout(0.3)  

def forward(self, data):  
    x, edge_index = data.x, data.edge_index  
    x = F.relu(self.conv1(x, edge_index))  
    x = self.dropout(x)  
    x = F.relu(self.conv2(x, edge_index))  
    x = self.dropout(x)  
    x = self.conv3(x, edge_index)  
    return x

============================================================

4️⃣ GRAPH CLASSIFIER (Wrapper for all encoders)

============================================================

class GraphClassifier(torch.nn.Module):
"""Wrapper classifier that combines an encoder with a linear classification head."""

def __init__(self, encoder, latent_dim, out_dim):  
    super().__init__()  
    self.encoder = encoder  
    self.classifier = torch.nn.Linear(latent_dim, out_dim)  

def forward(self, data):  
    latent = self.encoder(data)  
    logits = self.classifier(latent)  
    return F.log_softmax(logits, dim=1), latent
