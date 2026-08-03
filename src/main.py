# ============================================================
# CELLGRAPH - MAIN PIPELINE
# ============================================================

import scanpy as sc
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from src.data_loader import load_pbmc3k
from src.models import GATEncoder, GraphClassifier
from src.utils import build_graph
from src.train import train_model
from src.evaluate import evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# 1️⃣ LOAD DATA
# ============================================================
adata, X = load_pbmc3k()
print(f"Data loaded: {X.shape[0]} cells, {X.shape[1]} features")

# ============================================================
# 2️⃣ GENERATE LABELS
# ============================================================
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.leiden(adata, resolution=0.5)
le = LabelEncoder()
y = le.fit_transform(adata.obs["leiden"].astype(str))
n_classes = len(np.unique(y))
print(f"Number of classes: {n_classes}")

# ============================================================
# 3️⃣ TRAIN / VAL / TEST SPLIT
# ============================================================
idx = np.arange(len(y))
train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
train_idx, val_idx = train_test_split(train_idx, test_size=0.125, stratify=y[train_idx], random_state=42)

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

# Convert to tensors
train_idx_t = torch.tensor(train_idx, dtype=torch.long).to(device)
val_idx_t = torch.tensor(val_idx, dtype=torch.long).to(device)
test_idx_t = torch.tensor(test_idx, dtype=torch.long).to(device)

# ============================================================
# 4️⃣ BUILD GRAPH (ONLY FROM TRAINING DATA)
# ============================================================
edge_index = build_graph(X[train_idx], k=20)
print(f"Graph edges: {edge_index.shape[1]}")

# ============================================================
# 5️⃣ CREATE PYTORCH GEOMETRIC DATA
# ============================================================
data = Data(
    x=torch.tensor(X, dtype=torch.float),
    edge_index=edge_index,
    y=torch.tensor(y, dtype=torch.long)
).to(device)

# ============================================================
# 6️⃣ INITIALIZE MODEL
# ============================================================
encoder = GATEncoder(in_dim=X.shape[1], hidden=32, latent_dim=32, heads=4).to(device)
model = GraphClassifier(encoder, latent_dim=32, out_dim=n_classes).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# 7️⃣ TRAIN MODEL
# ============================================================
model, losses = train_model(model, data, train_idx_t, val_idx_t)
print("Training complete!")

# ============================================================
# 8️⃣ EVALUATE MODEL
# ============================================================
acc, f1 = evaluate_model(model, data, test_idx_t)
print(f"\n✅ Final Results:")
print(f"   Accuracy:  {acc:.4f}")
print(f"   Macro F1:  {f1:.4f}")
