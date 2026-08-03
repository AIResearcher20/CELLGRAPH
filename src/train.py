# ============================================================
# TRAIN
# ============================================================

import torch
import torch.nn.functional as F

def train_model(model, data, train_idx, val_idx, epochs=150, lr=0.005, patience=20):
    """
    Train the GNN model with early stopping based on validation loss.
    
    Args:
        model: GNN model
        data: PyTorch Geometric Data object
        train_idx: Training indices
        val_idx: Validation indices
        epochs: Maximum number of epochs
        lr: Learning rate
        patience: Early stopping patience
    
    Returns:
        model: Trained model (best checkpoint)
        losses: List of training losses
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # ========== Training ==========
        model.train()
        optimizer.zero_grad()
        out, _ = model(data)
        loss = F.nll_loss(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # ========== Validation ==========
        model.eval()
        with torch.no_grad():
            val_out, _ = model(data)
            val_loss = F.nll_loss(val_out[val_idx], data.y[val_idx])
            val_losses.append(val_loss.item())

        # ========== Early Stopping ==========
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                model.load_state_dict(best_model_state)
                print(f"⏹️ Early stopping at epoch {epoch}")
                break

        # ========== Logging ==========
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

    # Load best model before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"✅ Training complete. Best val loss: {best_val_loss:.4f}")
    return model, train_losses
