python
# ============================================================
# TRAIN
# ============================================================

import torch
import torch.nn.functional as F

def train_model(model, data, train_idx, val_idx, epochs=150, lr=0.005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out, _ = model(data)
        loss = F.nll_loss(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return model, losses
