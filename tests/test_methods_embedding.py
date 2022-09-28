import pytest

from oor_benchmark.api import sample_dataset
from oor_benchmark.methods._latent_embedding import embedding_scArches, embedding_scvi


@pytest.fixture
def anndata_trained(seed=42):
    adata = sample_dataset()
    embedding_scvi(adata, n_hvgs=50, train_params={"max_epochs": 1})
    return adata


@pytest.fixture
def anndata_trained_scarches(seed=42):
    adata = sample_dataset()
    embedding_scArches(adata, ref_dataset="atlas", n_hvgs=50, train_params={"max_epochs": 1})
    return adata


# @pytest.mark.skip(reason="This decorator should be removed when test passes.")
def test_embedding_scvi(anndata_trained):
    adata_merge = anndata_trained.copy()
    assert "X_scVI" in adata_merge.obsm


# @pytest.mark.skip(reason="This decorator should be removed when test passes.")
def test_embedding_scarches(anndata_trained_scarches):
    adata_merge = anndata_trained_scarches.copy()
    assert "X_scVI" in adata_merge.obsm


# @pytest.mark.skip(reason="This decorator should be removed when test passes.")
def test_full_vars(anndata_trained):
    adata_merge = anndata_trained.copy()
    assert adata_merge.n_vars > 50
