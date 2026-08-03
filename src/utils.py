# ============================================================
# UTILITIES
# ============================================================

import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import to_undirected

def build_graph(X, k=20):
    """
    Construct an undirected, symmetric kNN graph from a feature matrix.
    
    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features)
        k (int): Number of neighbors for kNN graph construction.
    
    Returns:
        torch.Tensor: Edge index of shape (2, n_edges) for the undirected graph.
    """
    adj = kneighbors_graph(X, n_neighbors=k, mode="connectivity", include_self=False)
    adj = adj.maximum(adj.T)  # make symmetric
    row, col = adj.nonzero()
    edge_index = torch.tensor(np.vstack([row, col]), dtype=torch.long)
    edge_index = to_undirected(edge_index)
    return edge_index
