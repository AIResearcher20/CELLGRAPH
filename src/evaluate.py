
# ============================================================
# EVALUATE
# ============================================================

import torch
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, data, test_idx):
    model.eval()
    with torch.no_grad():
        out, latent = model(data)
        preds = out.argmax(dim=1)
        y_true = data.y[test_idx].cpu().numpy()
        y_pred = preds[test_idx].cpu().numpy()
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1
