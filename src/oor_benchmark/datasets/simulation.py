from typing import List

import numpy as np
from anndata import AnnData


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
    query_batch: List[str] = None,
    ctrl_batch: List[str] = None,
    annotation_col: str = "leiden",
    query_annotation: List[str] = None,
    perturbation_type: str = "remove",
    test_frac: float = 0.2,
    DA_frac: float = 0.2,
    seed=42,
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
    test_frac:
        fraction of cells in each population to be included in the query group (only used if batch_col is None)
    DA_frac:
        the fraction of cells of query_annotation to keep in control if perturbation_type is 'expansion', or in query if perturbation_type is 'depletion'
    seed:
        random seed for sampling

    Returns:
    --------
    None, updates adata.obs in place adding `adata.obs['dataset_group']` column
    """
    np.random.seed(seed)

    if not isinstance(query_annotation, list):
        raise TypeError("A list of strings should be passed to query_annotation")
    if not isinstance(query_batch, list):
        raise TypeError("A list of strings should be passed to query_batch")
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

    elif perturbation_type == "expansion":
        for b in ctrl_batch:
            query_pop_cells = adata.obs_names[
                (adata.obs[batch_col] == b) & (adata.obs[annotation_col].isin(query_annotation))
            ]
            cells2remove = np.random.choice(query_pop_cells, size=int(np.round(len(query_pop_cells) * (1 - DA_frac))))
            adata.obs.loc[cells2remove, "is_ctrl"] = 0

    elif perturbation_type == "depletion":
        for b in query_batch:
            query_pop_cells = adata.obs_names[
                (adata.obs[batch_col] == b) & (adata.obs[annotation_col].isin(query_annotation))
            ]
            cells2remove = np.random.choice(query_pop_cells, size=int(np.round(len(query_pop_cells) * (1 - DA_frac))))
            adata.obs.loc[cells2remove, "is_query"] = 0

    else:
        raise ValueError("perturbation type should be one of 'remove' or 'perturb_pc'")
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

    adata.obs["OOR_state"] = (adata.obs[annotation_col].isin(query_annotation)).astype(int)

    adata.obs["cell_annotation"] = adata.obs[annotation_col].copy()
    adata.obs["sample_id"] = adata.obs[batch_col].copy()
    return adata
