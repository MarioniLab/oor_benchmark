import os

from oor_benchmark.api import check_method, sample_dataset
from oor_benchmark.methods import scArches_milo


def test_method_output():
    adata = sample_dataset()
    adata.obsm["X_scVI"] = adata.obsm["X_pca"].copy()
    assert check_method(scArches_milo.scArches_atlas_milo_ctrl(adata, annotation_col="louvain"))
    assert check_method(scArches_milo.scArches_atlas_milo_atlas(adata, annotation_col="louvain"))
    assert check_method(scArches_milo.scArches_ctrl_milo_ctrl(adata, annotation_col="louvain"))


def test_saved_output(tmp_path):
    outdir = str(tmp_path)
    adata = sample_dataset()
    scArches_milo.scArches_atlas_milo_ctrl(adata, train_params={"max_epochs": 2}, outdir=outdir)
    assert os.path.exists(os.path.join(outdir, "model_atlas"))
    assert os.path.exists(os.path.join(outdir, "model_fit_query2atlas"))
    # assert check_method(scArches_milo.scArches_atlas_milo_ctrl(adata, , train_params={'max_epochs': 2}, outdir=outdir))


def test_milo_output():
    adata = sample_dataset()
    adata.obsm["X_scVI"] = adata.obsm["X_pca"].copy()
    adata = scArches_milo.scArches_atlas_milo_ctrl(adata, annotation_col="louvain")
    assert "logFC" in adata.uns["sample_adata"].var
    assert "SpatialFDR" in adata.uns["sample_adata"].var
    assert "PValue" in adata.uns["sample_adata"].var
