import logging
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
    # Subset to datasets of interest
    try:
        assert embedding_reference in adata.obs["dataset_group"].unique()
    except AssertionError:
        raise ValueError(f"Embedding reference '{embedding_reference}' not found in adata.obs['dataset_group']")

    adata = adata[adata.obs["dataset_group"].isin([embedding_reference, "query"])]
    adata = adata[adata.obs.sort_values("dataset_group").index].copy()

    # for testing (remove later?)
    if "X_scVI" not in adata.obsm:
        if outdir is not None:
            try:
                # if os.path.exists(outdir + f"/model_{embedding_reference}/") and os.path.exists(outdir + f"/model_fit_query2{embedding_reference}/"):
                vae_ref = scvi.model.SCVI.load(outdir + f"/model_{embedding_reference}/")
                vae_q = scvi.model.SCVI.load(outdir + f"/model_fit_query2{embedding_reference}/")
                assert vae_ref.adata.obs_names.isin(
                    adata[adata.obs["dataset_group"] == embedding_reference].obs_names
                ).all()
                X_scVI_ref = pd.DataFrame(vae_ref.get_latent_representation(), index=vae_ref.adata.obs_names)
                X_scVI_q = pd.DataFrame(vae_q.get_latent_representation(), index=vae_q.adata.obs_names)
                X_scVI = pd.concat([X_scVI_q, X_scVI_ref], axis=0)
                adata.obsm["X_scVI"] = X_scVI.loc[adata.obs_names].values
                logging.info("Loading saved scVI models")
                del vae_ref
                del vae_q
            except (ValueError, FileNotFoundError):
                logging.info("Saved scVI models not found, running scVI and scArches embedding")
                embedding_scArches(
                    adata, ref_dataset=embedding_reference, outdir=outdir, batch_key="sample_id", **kwargs
                )
            except AssertionError:
                logging.info(
                    "Saved scVI model doesn't match cells in reference dataset, running scVI and scArches embedding"
                )
                embedding_scArches(
                    adata, ref_dataset=embedding_reference, outdir=outdir, batch_key="sample_id", **kwargs
                )
        else:
            embedding_scArches(adata, ref_dataset=embedding_reference, outdir=outdir, batch_key="sample_id", **kwargs)

    # Train KNN classifier
    _train_weighted_knn(
        adata[adata.obs["dataset_group"] == embedding_reference],
        outfile=outdir + f"weighted_KNN_classifier.{embedding_reference}.pkl",
        n_neighbors=k_neighbors,
    )

    # Compute label transfer probability
    mappingQC_labels = _weighted_knn_transfer_uncertainty(
        outdir + f"weighted_KNN_classifier.{embedding_reference}.pkl",
        query_adata=adata[adata.obs["dataset_group"] != embedding_reference],
        train_labels=adata[adata.obs["dataset_group"] == embedding_reference].obs[annotation_col],
    )["pred_uncertainty"]

    adata.obs["mappingQC_labels"] = np.nan
    adata.obs.loc[adata.obs["dataset_group"] != embedding_reference, "mappingQC_labels"] = mappingQC_labels
    adata.obs["mappingQC_labels"] = adata.obs["mappingQC_labels"].fillna(0)

    # Harmonize output (here groups are cells)
    sample_adata = anndata.AnnData(var=adata.obs[["mappingQC_labels"]])
    sample_adata.var["OOR_score"] = sample_adata.var["mappingQC_labels"].copy()
    sample_adata.var["OOR_signif"] = 0
    sample_adata.varm["groups"] = csc_matrix(identity(n=sample_adata.n_vars))
    adata.uns["sample_adata"] = sample_adata.copy()
    return adata


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
    # Subset to datasets of interest
    try:
        assert embedding_reference in adata.obs["dataset_group"].unique()
    except AssertionError:
        raise ValueError(f"Embedding reference '{embedding_reference}' not found in adata.obs['dataset_group']")

    adata = adata[adata.obs["dataset_group"].isin([embedding_reference, "query"])]
    adata = adata[adata.obs.sort_values("dataset_group").index].copy()

    # for testing (remove later?)
    if "X_scVI" not in adata.obsm:
        if outdir is not None:
            try:
                # if os.path.exists(outdir + f"/model_{embedding_reference}/") and os.path.exists(outdir + f"/model_fit_query2{embedding_reference}/"):
                vae_ref = scvi.model.SCVI.load(outdir + f"/model_{embedding_reference}/")
                vae_q = scvi.model.SCVI.load(outdir + f"/model_fit_query2{embedding_reference}/")
                assert vae_ref.adata.obs_names.isin(
                    adata[adata.obs["dataset_group"] == embedding_reference].obs_names
                ).all()
                X_scVI_ref = pd.DataFrame(vae_ref.get_latent_representation(), index=vae_ref.adata.obs_names)
                X_scVI_q = pd.DataFrame(vae_q.get_latent_representation(), index=vae_q.adata.obs_names)
                X_scVI = pd.concat([X_scVI_q, X_scVI_ref], axis=0)
                adata.obsm["X_scVI"] = X_scVI.loc[adata.obs_names].values
                logging.info("Loading saved scVI models")
            except (ValueError, FileNotFoundError):
                logging.info("Saved scVI models not found, running scVI and scArches embedding")
                embedding_scArches(
                    adata, ref_dataset=embedding_reference, outdir=outdir, batch_key="sample_id", **kwargs
                )
            except AssertionError:
                logging.info(
                    "Saved scVI model doesn't match cells in reference dataset, running scVI and scArches embedding"
                )
                embedding_scArches(
                    adata, ref_dataset=embedding_reference, outdir=outdir, batch_key="sample_id", **kwargs
                )
        else:
            embedding_scArches(adata, ref_dataset=embedding_reference, outdir=outdir, batch_key="sample_id", **kwargs)

    vae_q = scvi.model.SCVI.load(outdir + f"/model_fit_query2{embedding_reference}/")

    mappingQC_reconstruction = _reconstruction_dist_cosine(vae_q, adata[adata.obs["dataset_group"] == "query"])
    adata.obs["mappingQC_reconstruction"] = np.nan
    adata.obs.loc[
        adata.obs["dataset_group"] != embedding_reference, "mappingQC_reconstruction"
    ] = mappingQC_reconstruction

    # Harmonize output (here groups are cells)
    sample_adata = anndata.AnnData(var=adata.obs[["mappingQC_reconstruction"]])
    sample_adata.var["OOR_score"] = sample_adata.var["mappingQC_reconstruction"].copy()
    sample_adata.var["OOR_signif"] = 0
    sample_adata.varm["groups"] = csc_matrix(identity(n=sample_adata.n_vars))
    adata.uns["sample_adata"] = sample_adata.copy()
    return adata


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
    Cosine distance between true and reconstructed gene expression profile
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
    assert query_adata.obs_names.isin(vae.adata.obs_names).all(), "The model was trained on a different set of cells"
    # assert all(vae.adata.obs_names == query_adata.obs_names), "The model was trained on cells in a different order"

    # Compute true log-normalized gene expression profile
    if "log1p" not in query_adata.uns.keys():
        sc.pp.normalize_total(query_adata, target_sum=10000)
        sc.pp.log1p(query_adata)
    X_true = query_adata[:, vae.adata.var_names].X.copy()

    # Sample posterior
    scvi.settings.seed = seed
    post_sample = vae.posterior_predictive_sample(
        indices=np.where(vae.adata.obs_names.isin(query_adata.obs_names))[0], n_samples=n_samples
    )
    # Log normalize
    for s in range(post_sample.shape[2]):
        post_sample[:, :, s] = ((post_sample[:, :, s].T / post_sample[:, :, s].sum(1)).T) * 10000
    post_sample = np.log1p(post_sample)
    X_pred = post_sample.mean(2)
    X_pred_df = pd.DataFrame(X_pred, index=vae.adata.obs_names[vae.adata.obs_names.isin(query_adata.obs_names)])
    X_pred = X_pred_df.loc[query_adata.obs_names].values

    if scipy.sparse.issparse(X_true):
        X_true = X_true.toarray()
    if scipy.sparse.issparse(X_pred):
        X_pred = X_pred.toarray()

    if scale:
        X_pred = sc.pp.scale(X_pred, zero_center=False)
        X_true = sc.pp.scale(X_true, zero_center=False)

    # Compare true and predicted gene expression profile
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
