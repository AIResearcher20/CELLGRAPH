
# ============================================================
# DATA LOADER
# ============================================================

import scanpy as sc
import numpy as np

def load_pbmc3k():
    """Load and preprocess PBMC 3k dataset."""
    adata = sc.datasets.pbmc3k()
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable].copy()
    sc.tl.pca(adata, n_comps=50)
    
    X = adata.obsm["X_pca"]
    return adata, X
```
