import scvi
from anndata import AnnData


def train_scVI(train_adata: AnnData, outfile: str = None, **kwargs) -> scvi.model.SCVI:
    r"""Train scVI model.

    Parameters:
    ------------
    train_adata : AnnData
        training data (already subset to highly variable genes)
        counts should be stored in train_adata.layers['counts']
    outfile : str
        path to dir to save trained model
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

    model_train.train()
    if outfile is not None:
        model_train.save(outfile, save_anndata=True, overwrite=True)
    return model_train


def fit_scVI(model_train: scvi.model.SCVI, query_adata: AnnData, outfile: str) -> scvi.model.SCVI:
    """Fit query data to scVI model with scArches.

    Parameters:
    ----------------
    - model_train: trained scVI model
    - query_adata: AnnData object of query data (already subset to highly variable genes)
        counts should be stored in query_adata.layers['counts']
    - outfile: path to dir to save fitted model (if None, don't save)

    Returns:
    ----------
    - model_fitted: scVI model updated with scArches mapping
    """
    vae_q = scvi.model.SCVI.load_query_data(query_adata, model_train, inplace_subset_query_vars=True)

    vae_q.train(max_epochs=200, plan_kwargs={"weight_decay": 0.0})
    if outfile is not None:
        vae_q.save(outfile, save_anndata=True, overwrite=True)
    return vae_q
