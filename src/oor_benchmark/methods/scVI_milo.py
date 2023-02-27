import logging
import warnings

import pandas as pd
import scanpy as sc
import scvi
from anndata import AnnData

from ._latent_embedding import embedding_scvi
from .scArches_milo import run_milo

# logger = logging.getLogger(__name__)
#  Turn off deprecation warnings in scvi-tools
warnings.filterwarnings("ignore", category=DeprecationWarning)


def scVI_milo(
    adata: AnnData,
    embedding_reference: str = "atlas",
    diff_reference: str = "ctrl",
    sample_col: str = "sample_id",
    annotation_col: str = "cell_annotation",
    signif_alpha: float = 0.1,
    outdir: str = None,
    harmonize_output: bool = True,
    milo_design: str = "~ is_query",
    **kwargs,
):
    r"""Worflow for OOR state detection with scVI embedding and Milo differential analysis.

    Parameters:
    ------------
    adata: AnnData
        AnnData object of disease and reference cells to compare.
        If `adata.obsm['X_scVI']` is already present, the embedding step is skipped
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
    outdir: str
        path to output directory (default: None)
    milo_design: str
        design formula for differential abundance analysis (the test variable is always 'is_query')
    \**kwargs:
        extra arguments to embedding_scArches
    """
    # Subset to datasets of interest
    try:
        assert embedding_reference in adata.obs["dataset_group"].unique()
    except AssertionError:
        raise ValueError(f"Embedding reference '{embedding_reference}' not found in adata.obs['dataset_group']")
    try:
        assert diff_reference in adata.obs["dataset_group"].unique()
    except AssertionError:
        raise ValueError(f"Differential analysis reference '{diff_reference}' not found in adata.obs['dataset_group']")

    adata = adata[adata.obs["dataset_group"].isin([embedding_reference, diff_reference, "query"])]
    adata = adata[adata.obs.sort_values("dataset_group").index].copy()

    # for testing (remove later?)
    if "X_scVI" not in adata.obsm:
        if outdir is not None:
            try:
                dataset_groups = adata.obs["dataset_group"].unique().tolist()
                dataset_groups.sort()
                ref_dataset = "".join(dataset_groups)
                vae_ref = scvi.model.SCVI.load(outdir + f"/model_{ref_dataset}/")
                assert vae_ref.adata.obs_names.isin(
                    adata[adata.obs["dataset_group"] == embedding_reference].obs_names
                ).all()
                X_scVI = pd.DataFrame(vae_ref.get_latent_representation(), index=vae_ref.adata.obs_names)
                adata.obsm["X_scVI"] = X_scVI.loc[adata.obs_names].values
                logging.info("Loading saved scVI models")
                del vae_ref
            except (ValueError, FileNotFoundError):
                logging.info("Saved scVI models not found, running scVI and scArches embedding")
                embedding_scvi(adata, outdir=outdir, batch_key="sample_id", **kwargs)
            except AssertionError:
                logging.info(
                    "Saved scVI model doesn't match cells in reference dataset, running scVI and scArches embedding"
                )
                embedding_scvi(adata, outdir=outdir, batch_key="sample_id", **kwargs)
        else:
            embedding_scvi(adata, outdir=outdir, batch_key="sample_id", **kwargs)

    # remove embedding_reference from anndata if not needed anymore
    if diff_reference != embedding_reference:
        adata = adata[adata.obs["dataset_group"] != embedding_reference].copy()

    # Make KNN graph for Milo neigbourhoods
    n_controls = adata[adata.obs["dataset_group"] == diff_reference].obs[sample_col].unique().shape[0]
    n_querys = adata[adata.obs["dataset_group"] == "query"].obs[sample_col].unique().shape[0]
    #  Set max to 200 or memory explodes for large datasets
    k = min([(n_controls + n_querys) * 5, 200])
    sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=k)

    run_milo(adata, "query", diff_reference, sample_col=sample_col, annotation_col=annotation_col, design=milo_design)

    # Harmonize output
    if harmonize_output:
        sample_adata = adata.uns["nhood_adata"].T.copy()
        sample_adata.var["OOR_score"] = sample_adata.var["logFC"].copy()
        sample_adata.var["OOR_signif"] = (
            ((sample_adata.var["SpatialFDR"] < signif_alpha) & (sample_adata.var["logFC"] > 0)).astype(int).copy()
        )
        sample_adata.varm["groups"] = adata.obsm["nhoods"].T
        adata.uns["sample_adata"] = sample_adata.copy()
    return adata


def scVI_atlas_milo_ctrl(adata: AnnData, **kwargs):
    """Worflow for OOR state detection with scVI embedding and Milo differential analysis - ACR design."""
    return scVI_milo(adata, embedding_reference="atlas", diff_reference="ctrl", **kwargs)


def scVI_atlas_milo_atlas(adata: AnnData, **kwargs):
    """Worflow for OOR state detection with scVI embedding and Milo differential analysis - AR design."""
    return scVI_milo(adata, embedding_reference="atlas", diff_reference="atlas", **kwargs)


def scVI_ctrl_milo_ctrl(adata: AnnData, **kwargs):
    """Worflow for OOR state detection with scVI embedding and Milo differential analysis - CR design."""
    return scVI_milo(adata, embedding_reference="ctrl", diff_reference="ctrl", **kwargs)
