# ============================================================
# EVALUATE
# ============================================================

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

def evaluate_model(model, data, test_idx, class_names=None, return_details=False):
    """
    Comprehensive evaluation of GNN model on test data.
    
    Args:
        model: Trained GNN model
        data: PyTorch Geometric Data object
        test_idx: Test indices
        class_names: List of class names for report (optional)
        return_details: If True, return additional metrics and predictions
    
    Returns:
        dict or tuple: Evaluation results
    """
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        out, latent = model(data)
        preds = out.argmax(dim=1)
        
        # Extract test predictions
        y_true = data.y[test_idx].cpu().numpy()
        y_pred = preds[test_idx].cpu().numpy()
        latent_np = latent[test_idx].cpu().numpy()
        
        # ========== Core Metrics ==========
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        
        # ========== Classification Report ==========
        if class_names is not None:
            report = classification_report(
                y_true, 
                y_pred, 
                target_names=class_names,
                output_dict=True
            )
            report_str = classification_report(
                y_true, 
                y_pred, 
                target_names=class_names
            )
        else:
            report = classification_report(y_true, y_pred, output_dict=True)
            report_str = classification_report(y_true, y_pred)
        
        # ========== Confusion Matrix ==========
        cm = confusion_matrix(y_true, y_pred)
        
        # ========== Print Results ==========
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:          {acc:.4f}")
        print(f"Macro F1:          {f1_macro:.4f}")
        print(f"Weighted F1:       {f1_weighted:.4f}")
        print(f"Balanced Accuracy: {bal_acc:.4f}")
        print("\nClassification Report:")
        print(report_str)
        
        # ========== Return ==========
        results = {
            'accuracy': acc,
            'macro_f1': f1_macro,
            'weighted_f1': f1_weighted,
            'balanced_accuracy': bal_acc,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': {'y_true': y_true, 'y_pred': y_pred},
            'latent': latent_np
        }
        
        if return_details:
            return results
        return acc, f1_macro
