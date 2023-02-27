import logging
import warnings

import numpy as np
import pandas as pd
import scipy.stats
import scvi
from anndata import AnnData

from ._latent_embedding import embedding_scvi
from ._meld import run_meld

# logger = logging.getLogger(__name__)
#  Turn off deprecation warnings in scvi-tools
warnings.filterwarnings("ignore", category=DeprecationWarning)


# def _run_wilcoxon(adata, OOR_state, diff_reference: str = "ctrl"):
#     query_densities = adata[(adata['OOR_state'] == OOR_state) & (adata['dataset_group'] == 'query')]['sample_density'].values
#     ctrl_densities = adata[(adata['OOR_state'] == OOR_state) & (adata['dataset_group'] == 'ctrl')]['sample_density'].values
#     return(scipy.stats.ranksums(query_densities, ctrl_densities))


def scVI_meld(
    adata: AnnData,
    embedding_reference: str = "atlas",
    diff_reference: str = "ctrl",
    sample_col: str = "sample_id",
    # signif_quantile: float = 0.9,
    signif_alpha: float = 0.1,
    outdir: str = None,
    harmonize_output: bool = True,
    **kwargs,
):
    r"""Worflow for OOR state detection with scVI embedding and MELD differential analysis.

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
    signif_alpha: float
        Significance threshold for wilcoxon rank-sum test (default: 0.1)
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

    # Pick K for MELD KNN graph
    n_controls = adata[adata.obs["dataset_group"] == diff_reference].obs[sample_col].unique().shape[0]
    n_querys = adata[adata.obs["dataset_group"] == "query"].obs[sample_col].unique().shape[0]
    #  Set max to 200 or memory explodes for large datasets
    k = min([(n_controls + n_querys) * 3, 200])

    run_meld(adata, "query", diff_reference, sample_col=sample_col, n_neighbors=k)

    # Run wilcoxon test
    query_samples = adata.obs[sample_col][adata.obs["dataset_group"] == "query"].unique().tolist()
    reference_samples = adata.obs[sample_col][adata.obs["dataset_group"] == diff_reference].unique().tolist()

    pvals = []
    statistics = []
    for _, c in adata.obsm["sample_densities"].iterrows():
        statistic, pval = scipy.stats.ranksums(c[query_samples].values, c[reference_samples].values)
        pvals.append(pval)
        statistics.append(statistic)

    wilcox_df = pd.DataFrame(
        np.vstack([np.array(pvals), np.array(statistics)]).T, columns=["wilcox_pval", "wilcox_stat"]
    )
    wilcox_df.index = adata.obs_names
    adata.obs = pd.concat([adata.obs, wilcox_df], 1)

    # Harmonize output
    if harmonize_output:
        sample_adata = AnnData(var=adata.obs, varm=adata.obsm)
        sample_adata.var["OOR_score"] = sample_adata.var["wilcox_stat"]
        # quant_10perc = np.quantile(sample_adata.var["OOR_score"], signif_quantile)
        sample_adata.var["OOR_signif"] = sample_adata.var["wilcox_pval"] < signif_alpha
        # sample_adata.varm["groups"] = csc_matrix(np.identity(sample_adata.n_vars))
        adata.uns["sample_adata"] = sample_adata.copy()

    return adata


# def scVI_atlas_meld_ctrl(adata: AnnData, **kwargs):
#     """Worflow for OOR state detection with scArches embedding and MELD differential analysis - ACR design."""
#     return scArches_meld(adata, embedding_reference="atlas", diff_reference="ctrl", **kwargs)


# def scArches_atlas_meld_atlas(adata: AnnData, **kwargs):
#     """Worflow for OOR state detection with scArches embedding and MELD differential analysis - AR design."""
#     return scArches_meld(adata, embedding_reference="atlas", diff_reference="atlas", **kwargs)


# def scArches_ctrl_meld_ctrl(adata: AnnData, **kwargs):
#     """Worflow for OOR state detection with scArches embedding and MELD differential analysis - CR design."""
#     return scArches_meld(adata, embedding_reference="ctrl", diff_reference="ctrl", **kwargs)
