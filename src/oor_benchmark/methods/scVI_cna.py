import logging
import warnings

import cna
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from anndata import AnnData
from multianndata import MultiAnnData

from ._latent_embedding import embedding_scvi

# logger = logging.getLogger(__name__)
#  Turn off deprecation warnings in scvi-tools
warnings.filterwarnings("ignore", category=DeprecationWarning)


def run_cna(adata: AnnData, query_group: str, reference_group: str, sample_col: str = "sample_id"):
    """
    Run CNA to compute probability estimate per condition.

    Following tutorial in https://nbviewer.org/github/yakirr/cna/blob/master/demo/demo.ipynb

    Parameters:
    ------------
    adata : AnnData
        AnnData object of disease and reference cells to compare
    query_group : str
        Name of query group in adata_design.obs['dataset_group']
    reference_group : str
        Name of reference group in adata_design.obs['dataset_group']
    sample_col : str
        Name of column in adata_design.obs to use as sample ID
    """
    adata_design = MultiAnnData(adata, sampleid=sample_col)
    adata_design.obs["dataset_group"] = adata_design.obs["dataset_group"].astype("category")
    adata_design.obs["dataset_group_code"] = (
        adata_design.obs["dataset_group"].cat.reorder_categories([reference_group, query_group]).cat.codes
    )
    adata_design.obs_to_sample(["dataset_group_code"])
    res = cna.tl.association(adata_design, adata_design.samplem.dataset_group_code, ks=[20])
    adata.obs["CNA_ncorrs"] = res.ncorrs
    return None


def scArches_cna(
    adata: AnnData,
    embedding_reference: str = "atlas",
    diff_reference: str = "ctrl",
    sample_col: str = "sample_id",
    signif_quantile: float = 0.9,
    outdir: str = None,
    harmonize_output: bool = True,
    **kwargs,
):
    r"""Worflow for OOR state detection with scArches embedding and CNA differential analysis.

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
    signif_quantile: float
        quantile threshold for CNA ncorr (default: 0.9, top 10% probabilities are considered significant)
    outdir: str
        path to output directory (default: None)
    harmonize_output: bool
        whether to harmonize output to match other methods (default: True)
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

    # Pick K for CNA KNN graph
    n_controls = adata[adata.obs["dataset_group"] == diff_reference].obs[sample_col].unique().shape[0]
    n_querys = adata[adata.obs["dataset_group"] == "query"].obs[sample_col].unique().shape[0]
    #  Set max to 200 or memory explodes for large datasets
    k = min([(n_controls + n_querys) * 5, 200])
    sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=k)

    run_cna(adata, "query", diff_reference, sample_col=sample_col)
    assert "CNA_ncorrs" in adata.obs

    # Harmonize output
    if harmonize_output:
        sample_adata = AnnData(var=adata.obs)
        sample_adata.var["OOR_score"] = sample_adata.var["CNA_ncorrs"]
        quant_10perc = np.quantile(sample_adata.var["OOR_score"], signif_quantile)
        sample_adata.var["OOR_signif"] = sample_adata.var["OOR_score"] >= quant_10perc
        # sample_adata.varm["groups"] = csc_matrix(np.identity(sample_adata.n_vars))
        adata.uns["sample_adata"] = sample_adata.copy()

    return adata


# def scArches_atlas_meld_ctrl(adata: AnnData, **kwargs):
#     """Worflow for OOR state detection with scArches embedding and MELD differential analysis - ACR design."""
#     return scArches_meld(adata, embedding_reference="atlas", diff_reference="ctrl", **kwargs)


# def scArches_atlas_meld_atlas(adata: AnnData, **kwargs):
#     """Worflow for OOR state detection with scArches embedding and MELD differential analysis - AR design."""
#     return scArches_meld(adata, embedding_reference="atlas", diff_reference="atlas", **kwargs)


# def scArches_ctrl_meld_ctrl(adata: AnnData, **kwargs):
#     """Worflow for OOR state detection with scArches embedding and MELD differential analysis - CR design."""
#     return scArches_meld(adata, embedding_reference="ctrl", diff_reference="ctrl", **kwargs)
