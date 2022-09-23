import anndata
import milopy
import scanpy as sc
from anndata import AnnData

from ._latent_embedding import embedding_scArches


def run_milo(
    adata_design: AnnData,
    query_group: str,
    reference_group: str,
    sample_col: str = "sample_id",
    annotation_col: str = "cell_type",
    design: str = "~ is_query",
):
    """Test differential abundance analysis on neighbourhoods with Milo.

    Parameters:
    ------------
    adata_design : AnnData
        AnnData object of disease and reference cells to compare
    query_group : str
        Name of query group in adata_design.obs['dataset_group']
    reference_group : str
        Name of reference group in adata_design.obs['dataset_group']
    sample_col : str
        Name of column in adata_design.obs to use as sample ID
    annotation_cols : str
        Name of column in adata_design.obs to use as annotation
    design : str
        Design formula for differential abundance analysis
    """
    milopy.core.make_nhoods(adata_design, prop=0.1)
    milopy.core.count_nhoods(adata_design, sample_col=sample_col)
    milopy.utils.annotate_nhoods(adata_design[adata_design.obs["dataset_group"] == reference_group], annotation_col)
    adata_design.obs["is_query"] = adata_design.obs["dataset_group"] == query_group
    milopy.core.DA_nhoods(adata_design, design=design)


def scArches_milo(
    adata: AnnData,
    embedding_reference: str = "atlas",
    diff_reference: str = "ctrl",
    sample_col: str = "sample_id",
    annotation_col: str = "cell_type",
    signif_alpha: float = 0.1,
    **kwargs,
):
    r"""Worflow for OOR state detection with scArches embedding and Milo differential analysis.

    Parameters:
    ------------
    adata: AnnData
        AnnData object of disease and reference cells to compare
    embedding_reference: str
        Name of reference group in adata.obs['dataset_group'] to use for latent embedding
    diff_reference: str
        Name of reference group in adata.obs['dataset_group'] to use for differential abundance analysis
    sample_col: str
        Name of column in adata.obs to use as sample ID
    annotation_col: str
        Name of column in adata.obs to use as annotation
    signif_alpha: float
        FDR threshold for differential abundance analysi (default: 0.1)
    \**kwargs:
        extra arguments to embedding_scArches
    """
    adata_ref = adata[adata.obs["dataset_group"] == embedding_reference].copy()
    if diff_reference == embedding_reference:  # for AR design
        adata_query = adata[adata.obs["dataset_group"] == "query"].copy()
    else:
        adata_query = adata[adata.obs["dataset_group"] != embedding_reference].copy()

    # for testing (remove later?)
    if "X_scVI" in adata_ref.obsm and "X_scVI" in adata_query.obsm:
        adata_merge = anndata.concat([adata_query, adata_ref])
    else:
        adata_merge = embedding_scArches(adata_ref, adata_query, **kwargs)

    # remove embedding_reference from anndata if not needed anymore
    if diff_reference != embedding_reference:
        adata_merge = adata_merge[adata_merge.obs["dataset_group"] != embedding_reference].copy()

    # Make KNN graph for Milo neigbourhoods
    n_controls = adata_merge[adata_merge.obs["dataset_group"] == diff_reference].obs[sample_col].unique().shape[0]
    n_querys = adata_merge[adata_merge.obs["dataset_group"] == "query"].obs[sample_col].unique().shape[0]
    sc.pp.neighbors(adata_merge, use_rep="X_scVI", n_neighbors=(n_controls + n_querys) * 5)

    run_milo(adata_merge, "query", diff_reference, sample_col=sample_col, annotation_col=annotation_col)

    # Harmonize output
    sample_adata = adata_merge.uns["nhood_adata"].T.copy()
    sample_adata.var["OOR_score"] = sample_adata.var["logFC"].copy()
    sample_adata.var["OOR_signif"] = (
        ((sample_adata.var["SpatialFDR"] < signif_alpha) & (sample_adata.var["logFC"] > 0)).astype(int).copy()
    )
    sample_adata.varm["groups"] = adata_merge.obsm["nhoods"].T
    adata_merge.uns["sample_adata"] = sample_adata.copy()
    return adata_merge


def scArches_atlas_milo_ctrl(adata: AnnData, **kwargs):
    """Worflow for OOR state detection with scArches embedding and Milo differential analysis - ACR design."""
    return scArches_milo(adata, embedding_reference="atlas", diff_reference="ctrl", **kwargs)


def scArches_atlas_milo_atlas(adata: AnnData, **kwargs):
    """Worflow for OOR state detection with scArches embedding and Milo differential analysis - AR design."""
    return scArches_milo(adata, embedding_reference="atlas", diff_reference="atlas", **kwargs)


def scArches_ctrl_milo_ctrl(adata: AnnData, **kwargs):
    """Worflow for OOR state detection with scArches embedding and Milo differential analysis - CR design."""
    return scArches_milo(adata, embedding_reference="ctrl", diff_reference="ctrl", **kwargs)
