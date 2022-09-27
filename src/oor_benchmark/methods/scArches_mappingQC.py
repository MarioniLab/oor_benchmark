import pickle as pkl
from collections import Counter
from typing import List, Union

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import scvi
import sklearn.metrics
from anndata import AnnData
from pynndescent import NNDescent
from scipy.sparse import csc_matrix, identity

from ..api import check_dataset
from ._latent_embedding import embedding_scArches

# --- KNN classifier label transfer --- #
# Adapting implementation in https://github.com/LungCellAtlas/mapping_data_to_the_HLCA/blob/main/scripts/scarches_label_transfer.py


def scArches_mappingQClabels(
    adata: AnnData,
    embedding_reference: str,
    outdir: str,
    annotation_col: str = "cell_annotation",
    k_neighbors: int = 100,
    **kwargs,
):
    r"""Worflow for OOR state detection with scArches embedding and VAE reconstruction error.

    Parameters:
    ------------
    adata: AnnData
        AnnData object of disease and reference cells to compare
    embedding_reference: str
        Name of reference group in adata.obs['dataset_group'] to use for latent embedding
    outdir: str
        Path to save output models (required)
    \**kwargs:
        extra arguments to embedding_scArches
    """
    assert check_dataset(adata)

    adata_ref = adata[adata.obs["dataset_group"] == embedding_reference].copy()
    adata_query = adata[adata.obs["dataset_group"] != embedding_reference].copy()

    # for testing (remove later?)
    if "X_scVI" in adata_ref.obsm and "X_scVI" in adata_query.obsm:
        adata_merge = anndata.concat([adata_query, adata_ref])
    else:
        try:
            vae_ref = scvi.model.SCVI.load(outdir + f"/model_{embedding_reference}/")
            vae_q = scvi.model.SCVI.load(outdir + f"/model_fit_query2{embedding_reference}/")
            adata_merge = anndata.concat([adata_query, adata_ref])
            adata_merge.obsm["X_scVI"] = np.vstack(
                [vae_q.get_latent_representation(), vae_ref.get_latent_representation()]
            )
        except (FileNotFoundError, ValueError):
            adata_merge = embedding_scArches(adata_ref, adata_query, outdir=outdir, **kwargs)

    # Train KNN classifier
    _train_weighted_knn(
        adata_merge[adata_merge.obs["dataset_group"] == embedding_reference],
        outfile=outdir + "weighted_KNN_classifier.pkl",
        n_neighbors=k_neighbors,
    )

    # Compute label transfer probability
    mappingQC_labels = _weighted_knn_transfer_uncertainty(
        outdir + "weighted_KNN_classifier.pkl",
        query_adata=adata_merge[adata_merge.obs["dataset_group"] != embedding_reference],
        train_labels=adata_merge[adata_merge.obs["dataset_group"] == embedding_reference].obs[annotation_col],
    )["pred_uncertainty"]

    adata_merge.obs["mappingQC_labels"] = np.nan
    adata_merge.obs.loc[adata_merge.obs["dataset_group"] != embedding_reference, "mappingQC_labels"] = mappingQC_labels
    adata_merge.obs["mappingQC_labels"] = adata_merge.obs["mappingQC_labels"].fillna(0)

    # Harmonize output (here groups are cells)
    sample_adata = anndata.AnnData(var=adata_merge.obs[["mappingQC_labels"]])
    sample_adata.var["OOR_score"] = sample_adata.var["mappingQC_labels"].copy()
    sample_adata.var["OOR_signif"] = 0
    sample_adata.varm["groups"] = csc_matrix(identity(n=sample_adata.n_vars))
    adata_merge.uns["sample_adata"] = sample_adata.copy()
    return adata_merge


def _train_weighted_knn(train_adata: AnnData, outfile: str = None, use_rep: str = "X_scVI", n_neighbors: int = 50):
    """Trains a weighted KNN classifier on ``train_adata.obsm[use_rep]``.

    Parameters
    ----------
    train_adata: AnnData
        Annotated dataset to be used to train KNN classifier with ``label_key`` as the target variable.
    outfile: str
        path to pkl file to save trained model
    use_rep: str
        Name of the obsm layer to be used for calculation of neighbors. If set to "X", anndata.X will be
        used (default: X_scVI)
    n_neighbors: int
        Number of nearest neighbors in KNN classifier.
    """
    print(
        f"Weighted KNN with n_neighbors = {n_neighbors} ... ",
        end="",
    )
    if use_rep == "X":
        train_emb = train_adata.X
    elif use_rep in train_adata.obsm.keys():
        train_emb = train_adata.obsm[use_rep]
    else:
        raise ValueError("use_rep should be set to either 'X' or the name of the obsm layer to be used!")
    k_neighbors_transformer = NNDescent(
        train_emb,
        n_neighbors=n_neighbors,
        metric="euclidean",
        n_jobs=-1,
    )
    k_neighbors_transformer.prepare()

    if outfile is not None:
        with open(outfile, "wb") as f:
            pkl.dump(k_neighbors_transformer, f)
    return k_neighbors_transformer


def _weighted_knn_transfer_uncertainty(
    model: Union[NNDescent, str],
    query_adata: AnnData,
    train_labels: List[str],
    use_rep: str = "X_scVI",
    return_labels: bool = False,
):
    """Annotates ``query_adata`` cells with an input trained weighted KNN classifier.

    Parameters
    ----------
    model: pynndescent._neighbors.NNDescent
        knn model trained on reference adata with models.weighted_knn_trainer function
    query_adata: `AnnData`
        Annotated dataset to be used to queryate KNN classifier. Embedding to be used
    train_labels: List[str]
        list or series or array of labels from training data
    use_rep: str
        Name of the obsm layer to be used for label transfer. If set to "X",
        query_adata.X will be used
    ref_adata_obs: `pd.DataFrame`
        obs of ref Anndata
    label_keys: str
        Names of the columns to be used as target variables (e.g. cell_type) in ``query_adata``.
    """
    if type(model) == NNDescent:
        knn_model = model
    elif type(model) == str:
        try:
            with open(model, "rb") as f:
                knn_model = pkl.load(f)
        except (FileNotFoundError, ValueError):
            raise FileNotFoundError(f"{model} should be either a trained NNDescent object or a path to a pickle file")

    if isinstance(train_labels, pd.Series):
        y_train_labels = train_labels.values
    else:
        y_train_labels = train_labels

    if use_rep == "X":
        query_emb = query_adata.X
    elif use_rep in query_adata.obsm.keys():
        query_emb = query_adata.obsm[use_rep]
    else:
        raise ValueError("use_rep should be set to either 'X' or the name of the obsm layer to be used!")
    top_k_indices, top_k_distances = knn_model.query(query_emb, k=knn_model.n_neighbors)

    stds = np.std(top_k_distances, axis=1)
    stds = (2.0 / stds) ** 2
    stds = stds.reshape(-1, 1)

    top_k_distances_tilda = np.exp(-np.true_divide(top_k_distances, stds))

    weights = top_k_distances_tilda / np.sum(top_k_distances_tilda, axis=1, keepdims=True)
    uncertainties = pd.DataFrame(columns=["pred_uncertainty"], index=query_adata.obs_names)
    pred_labels = pd.DataFrame(columns=["pred_label"], index=query_adata.obs_names)

    for i in range(len(weights)):
        counter = Counter(y_train_labels[top_k_indices[i]])
        # Here I assume the highest no of neighbors also has the highest probability
        best_label = max(counter, key=counter.get)
        best_prob = weights[i, y_train_labels[top_k_indices[i]] == best_label].sum()

        uncertainties.iloc[i] = max(1 - best_prob, 0)
        pred_labels.iloc[i] = best_label

    if return_labels:
        return (uncertainties, pred_labels)
    else:
        return uncertainties


# --- Reconstruction error --- #

# def run_mappingQCreconstruction(
#     adata: AnnData,
#     reference_group: str,
#     annotation_col: str = "cell_type",
#     k_neighbors: int = 100,
#     outdir: str = None,
#     **kwargs
# ):
#     '''Compute reconstruction error from scArches model
#     '''
#     query_adata = adata[adata.obs['dataset_group'] != reference_group].copy()


def scArches_mappingQCreconstruction(
    adata: AnnData,
    embedding_reference: str,
    outdir: str,
    **kwargs,
):
    r"""Worflow for OOR state detection with scArches embedding and VAE reconstruction error.

    Parameters:
    ------------
    adata: AnnData
        AnnData object of disease and reference cells to compare
    embedding_reference: str
        Name of reference group in adata.obs['dataset_group'] to use for latent embedding
    outdir: str
        Path to save output models (required)
    \**kwargs:
        extra arguments to embedding_scArches
    """
    assert check_dataset(adata)

    adata_ref = adata[adata.obs["dataset_group"] == embedding_reference].copy()
    adata_query = adata[adata.obs["dataset_group"] != embedding_reference].copy()

    try:
        vae_ref = scvi.model.SCVI.load(outdir + f"/model_{embedding_reference}/")
        vae_q = scvi.model.SCVI.load(outdir + f"/model_fit_query2{embedding_reference}/")
        adata_merge = anndata.concat([adata_query, adata_ref])
        adata_merge.obsm["X_scVI"] = np.vstack([vae_q.get_latent_representation(), vae_ref.get_latent_representation()])
    except (FileNotFoundError, ValueError):
        adata_merge = embedding_scArches(adata_ref, adata_query, outdir=outdir, **kwargs)

    vae_q = scvi.model.SCVI.load(outdir + f"/model_fit_query2{embedding_reference}/")

    mappingQC_reconstruction = _reconstruction_dist_cosine(vae_q, adata_query)
    adata.obs["mappingQC_reconstruction"] = np.nan
    adata.obs.loc[
        adata.obs["dataset_group"] != embedding_reference, "mappingQC_reconstruction"
    ] = mappingQC_reconstruction

    # Harmonize output (here groups are cells)
    sample_adata = anndata.AnnData(var=adata_merge.obs[["mappingQC_reconstruction"]])
    sample_adata.var["OOR_score"] = sample_adata.var["mappingQC_reconstruction"].copy()
    sample_adata.var["OOR_signif"] = 0
    sample_adata.varm["groups"] = csc_matrix(identity(n=sample_adata.n_vars))
    adata_merge.uns["sample_adata"] = sample_adata.copy()
    return adata_merge


def _reconstruction_dist_cosine(
    model: Union[scvi.model.SCVI, str], query_adata: AnnData, n_samples=50, seed: int = 42, scale: bool = False
):
    """
    Compute cosine distance between true and reconstructed gene expression profile from scVI/scArches model.

    Parameters:
    ------------
    model:
        scVI/scArches model object or path to saved model file
    adata: AnnData object
        AnnData object or query data
    n_samples: int
        number of samples from posterior to use for reconstruction
    seed: int
        random seed for sampling from posterior
    scale: boolean
        should gex profiles be scaled before computing the distance (== calculating correlation)

    Returns:
    ------------
    None, modifies adata in place adding `adata.obs['trueVSpred_gex_cosine']`
    """
    if type(model) == scvi.model.SCVI:
        vae = model
    elif type(model) == str:
        try:
            vae = scvi.model.SCVI.load(model)
        except (FileNotFoundError, ValueError):
            raise FileNotFoundError(
                f"{model} should be either a trained scvi.model.SCVI object or a path to a model dir"
            )

    # Check the model is correct
    assert vae.adata.n_obs == query_adata.n_obs, "The model was trained on a different set of cells"
    assert all(vae.adata.obs_names == query_adata.obs_names), "The model was trained on cells in a different order"

    # Compute true log-normalized gene expression profile
    if "log1p" not in query_adata.uns.keys():
        sc.pp.normalize_per_cell(query_adata)
        sc.pp.log1p(query_adata)
    X_true = query_adata[:, vae.adata.var_names].X.copy()

    scvi.settings.seed = seed
    post_sample = vae.posterior_predictive_sample(n_samples=n_samples)
    post_sample = np.log1p(post_sample)
    X_pred = post_sample.mean(2)

    if scipy.sparse.issparse(X_true):
        X_true = X_true.toarray()
    if scipy.sparse.issparse(X_pred):
        X_pred = X_pred.toarray()

    if scale:
        X_pred = sc.pp.scale(X_pred, zero_center=False)
        X_true = sc.pp.scale(X_true, zero_center=False)

    cosine_all = sklearn.metrics.pairwise.cosine_distances(X_true, X_pred)
    return np.diag(cosine_all)


def scArches_atlas_mappingQClabels(adata: AnnData, outdir: str, **kwargs):
    """Worflow for OOR state detection with scArches embedding and label probability mapping QC - AR design."""
    return scArches_mappingQClabels(adata, embedding_reference="atlas", outdir=outdir, **kwargs)


def scArches_ctrl_mappingQClabels(adata: AnnData, outdir: str, **kwargs):
    """Worflow for OOR state detection with scArches embedding and label probability mapping QC - AR design."""
    return scArches_mappingQClabels(adata, embedding_reference="ctrl", outdir=outdir, **kwargs)


def scArches_atlas_mappingQCreconstruction(adata: AnnData, outdir: str, **kwargs):
    """Worflow for OOR state detection with scArches embedding and reconstruction mapping QC - AR design."""
    return scArches_mappingQCreconstruction(adata, embedding_reference="atlas", outdir=outdir, **kwargs)


def scArches_ctrl_mappingQCreconstruction(adata: AnnData, outdir: str, **kwargs):
    """Worflow for OOR state detection with scArches embedding and reconstruction mapping QC - AR design."""
    return scArches_mappingQCreconstruction(adata, embedding_reference="ctrl", outdir=outdir, **kwargs)
