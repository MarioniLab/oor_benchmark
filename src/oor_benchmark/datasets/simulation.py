from typing import List, Union

import numpy as np
import scanpy as sc
from anndata import AnnData
from sklearn.neighbors import KNeighborsClassifier


def _split_train_test(adata: AnnData, annotation_col: str = "leiden", test_frac: float = 0.2):

    test_cells = np.empty(shape=[0, 0]).ravel()
    for p in np.unique(adata.obs[annotation_col]):
        p_cells = adata.obs_names[adata.obs[annotation_col] == p].values
        p_test_cells = np.random.choice(p_cells, size=int(np.round(len(p_cells) * test_frac)))
        test_cells = np.hstack([test_cells, p_test_cells])

    train_cells = adata.obs_names[~adata.obs_names.isin(test_cells)]
    return (train_cells, test_cells)


def simulate_query_reference(
    adata: AnnData,
    batch_col: str = None,
    query_batch: Union[List[str], None] = None,
    ctrl_batch: Union[List[str], None] = None,
    annotation_col: str = "leiden",
    query_annotation: Union[List[str], None] = None,
    perturbation_type: str = "remove",
    test_frac: float = 0.2,
    # DA_frac: float = 0.2,
    split_pc: int = 0,
    seed=42,
    use_rep_shift: str = "X_scVI",
):
    """
    Split single-cell dataset in a atlas, control and query dataset.

    One population will be either absent, depleted or enriched from the
    reference and control datasets.

    Parameters:
    ------------
    adata: AnnData
        Single-cell dataset to be split in atlas, control and query datasets.
    batch_col:
        column in adata.obs containing sample identity (should correspond to samples used in differential analysis)
        (default: None, cells are split at random)
    query_batch:
        list of samples to assign to query group
    ctrl_batch:
        list of samples to assign to control group (default: None, no control group is specified)
    annotation_col:
        column in adata.obs containing cell type population identity
    query_annotation:
        which cell type population should be perturbed (defaults to None, pick one population at random)
    perturbation_type:
        one of 'remove', 'expansion' or 'depletion'. If equal to 'remove' (default) the population specified in query_annotation will be removed
        from the reference and control,
        if equal to 'expansion' a fraction of the cells in population specified in query_annotation
        will be removed from the samples in ctrl_batch (the fraction specified by DA_test)
        if equal to 'depletion' a fraction of the cells in population specified in query_annotation
        will be removed from the samples in query_batch (the fraction specified by DA_test)
        if equal to shift, the query population will be shifted along a principal component

    test_frac:
        fraction of cells in each population to be included in the query group (only used if batch_col is None)
    DA_frac:
        the fraction of cells of query_annotation to keep in control if perturbation_type is 'expansion', or in query if perturbation_type is 'depletion'
    split_pc:
        index of PC to use for splitting (default: 0, using PC1) (only used if perturbation_type=shift)
    seed:
        random seed for sampling
    use_rep_shift:
        representation to use to find neighbors in atlas dataset for shift perturbation (default: 'X_scVI')

    Returns:
    --------
    None, updates adata.obs in place adding `adata.obs['dataset_group']` column
    """
    np.random.seed(seed)
    if query_annotation is not None:
        if not isinstance(query_annotation, list):
            raise TypeError("A list of strings should be passed to query_annotation")
    if query_batch is not None:
        if not isinstance(query_batch, list):
            raise TypeError("A list of strings should be passed to query_batch")
    if ctrl_batch is not None:
        if not isinstance(ctrl_batch, list):
            raise TypeError("A list of strings should be passed to ctrl_batch")

    # Split in query-control-reference
    if batch_col is not None:
        # split by defined batches
        query = np.array([s in query_batch for s in adata.obs[batch_col]])
        adata.obs["is_train"] = (~query).astype(int)
        adata.obs["is_test"] = query.astype("int")
        if ctrl_batch is not None:
            ctrl = np.array([s in ctrl_batch for s in adata.obs[batch_col]])
            adata.obs["is_ctrl"] = ctrl.astype("int")
            adata.obs["is_train"] = adata.obs["is_train"] - adata.obs["is_ctrl"]
    else:
        # random split
        train_cells, test_cells = _split_train_test(adata, annotation_col=annotation_col, test_frac=test_frac)
        adata.obs["is_train"] = adata.obs_names.isin(train_cells).astype("int")
        adata.obs["is_test"] = adata.obs_names.isin(test_cells).astype("int")

    # Pick cell population to perturb
    if query_annotation is None:
        query_annotation = np.random.choice(adata.obs[annotation_col].unique(), size=1)

    #  Apply perturbation
    if perturbation_type == "remove":
        adata.obs.loc[(adata.obs[annotation_col].isin(query_annotation)), "is_train"] = 0
        if ctrl_batch is not None:
            adata.obs.loc[(adata.obs[annotation_col].isin(query_annotation)), "is_ctrl"] = 0

    # elif perturbation_type == "expansion":
    #     for b in ctrl_batch:
    #         query_pop_cells = adata.obs_names[
    #             (adata.obs[batch_col] == b) & (adata.obs[annotation_col].isin(query_annotation))
    #         ]
    #         cells2remove = np.random.choice(query_pop_cells, size=int(np.round(len(query_pop_cells) * (1 - DA_frac))))
    #         adata.obs.loc[cells2remove, "is_ctrl"] = 0

    # elif perturbation_type == "depletion":
    #     for b in query_batch:
    #         query_pop_cells = adata.obs_names[
    #             (adata.obs[batch_col] == b) & (adata.obs[annotation_col].isin(query_annotation))
    #         ]
    #         cells2remove = np.random.choice(query_pop_cells, size=int(np.round(len(query_pop_cells) * (1 - DA_frac))))
    #         adata.obs.loc[cells2remove, "is_query"] = 0
    elif perturbation_type == "shift":
        split_pop_cells = adata.obs_names[
            (adata.obs[annotation_col].isin(query_annotation)) & (adata.obs["is_train"] == 0)
        ]
        # Run PCA on perturbation population (just query dataset to avoid batch effects)
        split_pop_adata = adata[adata.obs_names.isin(split_pop_cells)].copy()
        sc.pp.normalize_per_cell(split_pop_adata)
        sc.pp.log1p(split_pop_adata)
        sc.pp.pca(split_pop_adata)
        pc2split = split_pop_adata.obsm["X_pca"][:, split_pc]
        test_size = int(np.round(len(split_pop_cells) * 0.5))
        idx = np.argpartition(pc2split, test_size)
        cells2remove = split_pop_cells[idx[:test_size]].values

        # Find neighbors in atlas cells
        split_pop_adata.obs["remove"] = split_pop_adata.obs_names.isin(cells2remove).astype(int)
        split_pop_cells_atlas = adata.obs_names[
            (adata.obs[annotation_col].isin(query_annotation)) & (adata.obs["is_train"] == 1)
        ]
        X_train = adata[split_pop_cells].obsm[use_rep_shift]
        Y_train = split_pop_adata.obs["remove"]
        X_atlas = adata[split_pop_cells_atlas].obsm[use_rep_shift]

        neigh = KNeighborsClassifier(n_neighbors=10)
        neigh = neigh.fit(X_train, Y_train)
        atlas_cells2remove = split_pop_cells_atlas[neigh.predict(X_atlas) == 1]

        adata.obs.loc[cells2remove, "is_ctrl"] = 0
        adata.obs.loc[atlas_cells2remove, "is_train"] = 0
        oor_cells = adata.obs_names[(adata.obs["is_test"] == 1) & (adata.obs_names.isin(cells2remove))]

    else:
        raise ValueError("perturbation type should be one of 'remove' or 'shift'")
    adata.uns["perturbation"] = {
        "annotation_col": annotation_col,
        "batch_col": batch_col,
        "query_annotation": query_annotation,
        "query_batch": query_batch,
        "ctrl_batch": ctrl_batch,
        "perturbation_type": perturbation_type,
    }

    adata.obs["dataset_group"] = "exclude"
    adata.obs["dataset_group"] = np.where(adata.obs["is_test"] == 1, "query", adata.obs["dataset_group"])
    adata.obs["dataset_group"] = np.where(adata.obs["is_ctrl"] == 1, "ctrl", adata.obs["dataset_group"])
    adata.obs["dataset_group"] = np.where(adata.obs["is_train"] == 1, "atlas", adata.obs["dataset_group"])
    adata = adata[adata.obs["dataset_group"] != "exclude"].copy()  # remove cells that are not in any group

    if perturbation_type == "remove":
        adata.obs["OOR_state"] = (adata.obs[annotation_col].isin(query_annotation)).astype(int)
    elif perturbation_type == "shift":
        adata.obs["OOR_state"] = (adata.obs_names.isin(oor_cells)).astype(int)

    adata.obs["cell_annotation"] = adata.obs[annotation_col].copy()
    adata.obs["sample_id"] = adata.obs[batch_col].copy()
    return adata
