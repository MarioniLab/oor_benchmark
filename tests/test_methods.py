import pytest

from oor_benchmark.api import check_method, sample_dataset
from oor_benchmark.methods import scArches_milo
from oor_benchmark.methods._latent_embedding import embedding_scArches, embedding_scvi


@pytest.fixture
def anndata_trained(seed=42):
    adata = sample_dataset()
    adata_ref = adata[adata.obs["dataset_group"] == "atlas"].copy()
    adata_query = adata[adata.obs["dataset_group"] == "query"].copy()
    adata_merge = embedding_scvi(adata_ref, adata_query, n_hvgs=50, train_params={"max_epochs": 1})
    return adata_merge


@pytest.fixture
def anndata_trained_scarches(seed=42):
    adata = sample_dataset()
    adata_ref = adata[adata.obs["dataset_group"] == "atlas"].copy()
    adata_query = adata[adata.obs["dataset_group"] == "query"].copy()
    adata_merge = embedding_scArches(adata_ref, adata_query, n_hvgs=50, train_params={"max_epochs": 1})
    return adata_merge


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


def test_method_output():
    adata = sample_dataset()
    adata.obsm["X_scVI"] = adata.obsm["X_pca"].copy()
    assert check_method(scArches_milo.scArches_atlas_milo_ctrl(adata, annotation_col="louvain"))
    assert check_method(scArches_milo.scArches_atlas_milo_atlas(adata, annotation_col="louvain"))
    assert check_method(scArches_milo.scArches_ctrl_milo_ctrl(adata, annotation_col="louvain"))
