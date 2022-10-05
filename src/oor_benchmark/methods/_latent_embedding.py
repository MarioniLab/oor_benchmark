import pandas as pd
import scanpy as sc
import scvi
from anndata import AnnData


def embedding_scvi(adata_merge: AnnData, n_hvgs: int = 5000, outdir: str = None, train_params=None, **kwargs):
    r"""Latent embedding with scVI.

    Parameters:
    ------------
    adata_merge: AnnData
        object of reference data
    n_hvgs: int
        number of highly variable genes to use for latent embedding
    outdir: str
        path to dir to save trained models
    train_params : dict, optional
        dictionary of training parameters to add, by default None
    \**kwargs:
        extra arguments to _train_scVI

    Returns:
    -----------
    None, adds latent embedding in `.obsm` in place
    """
    dataset_groups = adata_merge.obs["dataset_group"].unique().tolist()
    dataset_groups.sort()
    ref_dataset = "".join(dataset_groups)
    # adata_merge = anndata.concat([adata_query, adata_ref])
    # adata_merge.layers["counts"] = adata_merge.X.copy()
    adata_merge_train = adata_merge.copy()

    # Filter genes
    adata_merge_train.layers["counts"] = adata_merge_train.X.copy()
    _filter_genes_scvi(adata_merge_train)

    # Train scVI model
    if outdir is not None:
        outdir = outdir + f"/model_{ref_dataset}/"
    model_scvi = _train_scVI(adata_merge_train, outfile=outdir, train_params=train_params, **kwargs)

    # Get latent embeddings
    adata_merge.obsm["X_scVI"] = model_scvi.get_latent_representation()
    # return adata_merge


def embedding_scArches(
    adata_merge: AnnData, ref_dataset: str, n_hvgs: int = 5000, outdir: str = None, train_params=None, **kwargs
):
    r"""Latent embedding with scVI + scArches.

    Parameters:
    ------------
    adata_merge: AnnData
        object of reference and query data
    ref_dataset: str
        ID of reference dataset (must be in adata_merge.obs['dataset_group'])
    n_hvgs: int
        number of highly variable genes to use for latent embedding
    outdir: str
        path to dir to save trained models
    train_params : dict, optional
        dictionary of training parameters to add, by default None
    \**kwargs:
        extra arguments to _train_scVI

    Returns:
    -----------
    None, adds latent embedding in `.obsm` in place
    """
    # ref_dataset = adata_ref.obs["dataset_group"]
    # adata_merge = anndata.concat([adata_query, adata_ref])
    # adata_merge.layers["counts"] = adata_merge.X.copy()
    # adata_query.layers["counts"] = adata_query.X.copy()
    # adata_ref.layers["counts"] = adata_ref.X.copy()

    # Filter genes
    assert ref_dataset in adata_merge.obs["dataset_group"].unique().tolist()
    adata_merge.layers["counts"] = adata_merge.X.copy()
    adata_ref_train = adata_merge[adata_merge.obs["dataset_group"] == ref_dataset].copy()
    _filter_genes_scvi(adata_ref_train)

    # Train scVI model
    if outdir is not None:
        ref_outdir = outdir + f"/model_{ref_dataset}/"
    else:
        ref_outdir = None
    vae_ref = _train_scVI(adata_ref_train, outfile=ref_outdir, train_params=train_params, **kwargs)

    # Fit query data to scVI model
    adata_query_fit = adata_merge[adata_merge.obs["dataset_group"] != ref_dataset].copy()
    if outdir is not None:
        q_outdir = outdir + f"/model_fit_query2{ref_dataset}/"
    else:
        q_outdir = None
    vae_q = _fit_scVI(vae_ref, adata_query_fit, train_params=train_params, outfile=q_outdir)

    # Get latent embeddings
    # adata_merge = adata_merge[np.hstack([adata_ref_train.obs_names.values, adata_query_fit.obs_names.values])].copy()
    X_scVI_ref = pd.DataFrame(vae_ref.get_latent_representation(), index=vae_ref.adata.obs_names)
    X_scVI_q = pd.DataFrame(vae_q.get_latent_representation(), index=vae_q.adata.obs_names)
    X_scVI = pd.concat([X_scVI_q, X_scVI_ref], axis=0)
    adata_merge.obsm["X_scVI"] = X_scVI.loc[adata_merge.obs_names].values


# --- Model wrappers --- #


def _train_scVI(train_adata: AnnData, train_params: dict = None, outfile: str = None, **kwargs) -> scvi.model.SCVI:
    r"""Train scVI model.

    Parameters:
    ------------
    train_adata : AnnData
        training data (already subset to highly variable genes)
        counts should be stored in train_adata.layers['counts']
    outfile : str
        path to dir to save trained model
    train_params : dict, optional
        dictionary of training parameters to add, by default None
    \**kwargs : dict, optional
        Extra arguments to `scvi.model.SCVI.setup_anndata` (specifying batch etc)
    """
    scvi.model.SCVI.setup_anndata(train_adata, layer="counts", **kwargs)

    arches_params = {
        "use_layer_norm": "both",
        "use_batch_norm": "none",
        "encode_covariates": True,
        "dropout_rate": 0.2,
        "n_layers": 2,
    }

    model_train = scvi.model.SCVI(train_adata, **arches_params)

    model_train.train(**train_params)
    if outfile is not None:
        model_train.save(outfile, save_anndata=True, overwrite=True)
    return model_train


def _fit_scVI(
    model_train: scvi.model.SCVI, query_adata: AnnData, train_params: dict = None, outfile: str = None
) -> scvi.model.SCVI:
    r"""Fit query data to scVI model with scArches.

    Parameters:
    ----------------
    model_train :  scvi.model.SCVI
        trained scVI model
    query_adata : AnnData
        AnnData object of query data (already subset to highly variable genes)
        counts should be stored in query_adata.layers['counts']
    outfile : str
        path to dir to save fitted model (if None, don't save)
    train_params : dict, optional
        dictionary of training parameters to add, by default None

    Returns:
    ----------
    model_fitted : scvi.model.SCVI
        model updated with scArches mapping
    """
    vae_q = scvi.model.SCVI.load_query_data(query_adata, model_train, inplace_subset_query_vars=True)

    vae_q.train(**train_params, plan_kwargs={"weight_decay": 0.0})
    if outfile is not None:
        vae_q.save(outfile, save_anndata=True, overwrite=True)
    return vae_q


# --- Latent embedding utils --- #


def _filter_genes_scvi(adata: AnnData):
    """Filter genes for latent embedding."""
    # Filter genes not expressed anywhere
    sc.pp.filter_genes(adata, min_cells=1)

    # Select HVGs
    if "log1p" not in adata.uns.keys():
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=5000, subset=True)
