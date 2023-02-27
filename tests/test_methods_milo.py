import logging
import os

import pytest

from oor_benchmark.api import check_method, sample_dataset
from oor_benchmark.methods import scArches_milo, scVI_milo


@pytest.mark.skip(reason="This decorator should be removed when test passes.")
def test_method_output():
    adata = sample_dataset()
    adata.obsm["X_scVI"] = adata.obsm["X_pca"].copy()
    assert check_method(scArches_milo.scArches_atlas_milo_ctrl(adata, annotation_col="louvain"))
    assert check_method(scArches_milo.scArches_atlas_milo_atlas(adata, annotation_col="louvain"))
    assert check_method(scArches_milo.scArches_ctrl_milo_ctrl(adata, annotation_col="louvain"))
    assert check_method(scVI_milo.scVI_atlas_milo_ctrl(adata, annotation_col="louvain"))
    assert check_method(scVI_milo.scVI_atlas_milo_atlas(adata, annotation_col="louvain"))
    assert check_method(scVI_milo.scVI_ctrl_milo_ctrl(adata, annotation_col="louvain"))


# @pytest.mark.skip(reason="This decorator should be removed when test passes.")
def test_saved_output_scVI(tmp_path):
    outdir = str(tmp_path)
    adata = sample_dataset()
    scVI_milo.scVI_atlas_milo_ctrl(adata, train_params={"max_epochs": 2}, outdir=outdir)
    assert os.path.exists(os.path.join(outdir, "model_atlasctrlquery"))


@pytest.mark.skip(reason="This decorator should be removed when test passes.")
def test_saved_output_scArches(tmp_path):
    outdir = str(tmp_path)
    adata = sample_dataset()
    scArches_milo.scArches_atlas_milo_ctrl(adata, train_params={"max_epochs": 2}, outdir=outdir)
    assert os.path.exists(os.path.join(outdir, "model_atlas"))
    assert os.path.exists(os.path.join(outdir, "model_fit_query2atlas"))


@pytest.mark.skip(reason="This decorator should be removed when test passes.")
def test_no_overwrite(tmp_path, caplog):
    outdir = str(tmp_path)
    caplog.set_level(logging.INFO)
    adata = sample_dataset()
    scArches_milo.scArches_atlas_milo_ctrl(adata, train_params={"max_epochs": 2}, outdir=outdir)
    assert "Saved scVI models not found, running scVI and scArches embedding" in caplog.text
    scArches_milo.scArches_atlas_milo_ctrl(adata, train_params={"max_epochs": 2}, outdir=outdir)
    assert "Saved scVI models found - loading latent embedding" in caplog.text


@pytest.mark.skip(reason="This decorator should be removed when test passes.")
def test_milo_output():
    adata = sample_dataset()
    adata.obsm["X_scVI"] = adata.obsm["X_pca"].copy()
    adata = scArches_milo.scArches_atlas_milo_ctrl(adata, annotation_col="louvain")
    assert "logFC" in adata.uns["sample_adata"].var
    assert "SpatialFDR" in adata.uns["sample_adata"].var
    assert "PValue" in adata.uns["sample_adata"].var
    adata = sample_dataset()
    adata.obsm["X_scVI"] = adata.obsm["X_pca"].copy()
    adata = scVI_milo.scVI_atlas_milo_ctrl(adata, annotation_col="louvain")
    assert "logFC" in adata.uns["sample_adata"].var
    assert "SpatialFDR" in adata.uns["sample_adata"].var
    assert "PValue" in adata.uns["sample_adata"].var
