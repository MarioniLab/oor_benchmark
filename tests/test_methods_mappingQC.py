from oor_benchmark.api import check_method, sample_dataset
from oor_benchmark.methods import scArches_mappingQC


def test_method_output(tmp_path):
    outdir = str(tmp_path)
    adata = sample_dataset()
    adata.obsm["X_scVI"] = adata.obsm["X_pca"].copy()
    # assert check_method(scArches_mappingQC.scArches_ctrl_mappingQCreconstruction(adata, outdir='./tmp/'))
    # assert check_method(scArches_mappingQC.scArches_atlas_mappingQCreconstruction(adata, outdir='./tmp/'))
    assert check_method(scArches_mappingQC.scArches_ctrl_mappingQClabels(adata, outdir=outdir))
    assert check_method(scArches_mappingQC.scArches_atlas_mappingQClabels(adata, outdir=outdir))


def test_mappingQClabel_output(tmp_path):
    outdir = str(tmp_path)
    adata = sample_dataset()
    adata.obsm["X_scVI"] = adata.obsm["X_pca"].copy()
    adata = scArches_mappingQC.scArches_ctrl_mappingQClabels(adata, outdir=outdir)
    assert "mappingQC_labels" in adata.uns["sample_adata"].var
