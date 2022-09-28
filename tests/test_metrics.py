import pytest

from oor_benchmark.api import sample_dataset
from oor_benchmark.methods.scArches_milo import scArches_atlas_milo_ctrl
from oor_benchmark.metrics.auprc import auprc
from oor_benchmark.metrics.FDR_TPR_FPR import FDR_TPR_FPR


@pytest.fixture
def anndata_trained(seed=42):
    adata = sample_dataset()
    adata.obsm["X_scVI"] = adata.obsm["X_pca"].copy()
    adata_merge = scArches_atlas_milo_ctrl(adata, annotation_col="louvain")
    return adata_merge


# @pytest.mark.skip(reason="This decorator should be removed when test passes.")
def test_auprc_outputs(anndata_trained):
    adata_merge = anndata_trained.copy()
    assert auprc(adata_merge).shape[0] == 1
    assert auprc(adata_merge, return_curve=False).shape[0] == 1
    assert auprc(adata_merge, return_curve=True).shape[0] > 1


# @pytest.mark.skip(reason="This decorator should be removed when test passes.")
def test_TPR_outputs(anndata_trained):
    adata_merge = anndata_trained.copy()
    tpr_df = FDR_TPR_FPR(adata_merge)
    assert tpr_df.shape[0] == 1
    assert (tpr_df.loc[0, "TPR"] >= 0.0) & (tpr_df.loc[0, "TPR"] <= 1.0)
    assert (tpr_df.loc[0, "FPR"] >= 0.0) & (tpr_df.loc[0, "FPR"] <= 1.0)
    assert (tpr_df.loc[0, "FDR"] >= 0.0) & (tpr_df.loc[0, "FDR"] <= 1.0)


# @pytest.mark.skip(reason="This decorator should be removed when test passes.")


def test_positives(anndata_trained):
    """test that sum of TP and FP makes sense and TN and FN"""
    adata_merge = anndata_trained.copy()
    tpr_df = FDR_TPR_FPR(adata_merge)
    assert tpr_df["TP"][0] + tpr_df["FP"][0] == adata_merge.uns["sample_adata"].var["OOR_signif"].sum()
    assert tpr_df["TP"][0] + tpr_df["FN"][0] == adata_merge.uns["sample_adata"].var["OOR_state_group"].sum()
