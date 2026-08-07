import scanpy as sc
import numpy as np

def load_pbmc3k():
    """Load and preprocess PBMC 3k dataset with QC."""
    adata = sc.datasets.pbmc3k()
    
    # ========== Quality Control ==========
    # Filter cells
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_genes=1)
    
    # Filter genes
    sc.pp.filter_genes(adata, min_cells=3)
    
    # ========== Normalization ==========
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # ========== Feature Selection ==========
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable].copy()
    
    # ========== Dimensionality Reduction ==========
    sc.tl.pca(adata, n_comps=50)
    
    X = adata.obsm["X_pca"]
    return adata, X
