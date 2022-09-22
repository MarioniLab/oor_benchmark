import numpy as np
import scanpy as sc
from anndata import AnnData


def check_dataset(adata: AnnData):
    """Check that dataset output fits expected API."""


def check_method(adata: AnnData):
    """Check that method output fits expected API."""


def sample_dataset():
    """Create a simple dataset to use for testing methods in this task."""
    np.random.seed(3456)
    adata = sc.datasets.pbmc3k_processed()
    adata.obs["dataset_group"] = np.random.choice(["atlas", "ctrl", "query"], p=[0.6, 0.2, 0.2], size=adata.shape[0])
    adata.obs["OOR_state"] = np.where(adata.obs["louvain"] == "B cells", 1, 0)
    return adata
