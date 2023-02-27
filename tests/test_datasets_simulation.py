import pytest
import scanpy as sc
from scvi.data import heart_cell_atlas_subsampled

from oor_benchmark.api import check_dataset
from oor_benchmark.datasets.simulation import simulate_query_reference

adata = heart_cell_atlas_subsampled()


def test_output():
    """
    Test that perturbed population is correctly removed when perturbation_type == 'remove
    """
    annotation_col = "cell_type"
    perturb_pop = ["Myeloid"]
    batch_col = "donor"
    query_batch = ["D4", "D6"]
    ctrl_batch = ["D2", "D3"]

    adata_sim = simulate_query_reference(
        adata,
        query_annotation=perturb_pop,
        annotation_col=annotation_col,
        batch_col=batch_col,
        query_batch=query_batch,
        ctrl_batch=ctrl_batch,
        perturbation_type="remove",
    )

    assert check_dataset(adata_sim)


def test_query_specific_pop():
    """
    Test that perturbed population is correctly removed when perturbation_type == 'remove
    """
    annotation_col = "cell_type"
    perturb_pop = ["Myeloid"]
    batch_col = "donor"
    query_batch = ["D4", "D6"]
    ctrl_batch = ["D2", "D3"]

    adata_sim = simulate_query_reference(
        adata,
        query_annotation=perturb_pop,
        annotation_col=annotation_col,
        batch_col=batch_col,
        query_batch=query_batch,
        ctrl_batch=ctrl_batch,
        perturbation_type="remove",
    )

    # Checks
    assert not any(adata_sim.obs[adata_sim.obs[annotation_col].isin(perturb_pop)]["dataset_group"] != "query")


def test_batch_group():
    """
    Test that no batch is in multiple groups
    """
    annotation_col = "cell_type"
    perturb_pop = ["Myeloid"]
    batch_col = "donor"
    query_batch = ["D4", "D6"]
    ctrl_batch = ["D2", "D3"]

    adata_sim = simulate_query_reference(
        adata,
        query_annotation=perturb_pop,
        annotation_col=annotation_col,
        batch_col=batch_col,
        query_batch=query_batch,
        ctrl_batch=ctrl_batch,
        perturbation_type="remove",
    )

    # Checks
    assert all(adata_sim.obs[[batch_col, "dataset_group"]].groupby(batch_col).nunique() == 1)
    q_adata = adata_sim.obs[(adata_sim.obs[annotation_col].isin(perturb_pop)) & (["dataset_group"] == "query")]
    assert sum(q_adata["OOR_state"] == 0) == 0


def test_shift():
    """
    Test that perturbed population is correctly removed when perturbation_type == 'remove
    """
    annotation_col = "cell_type"
    perturb_pop = ["Myeloid"]
    batch_col = "donor"
    query_batch = ["D4", "D6"]
    ctrl_batch = ["D2", "D3"]

    # Log normalize adata heart
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.pp.pca(adata, n_comps=10)

    adata_sim = simulate_query_reference(
        adata,
        query_annotation=perturb_pop,
        annotation_col=annotation_col,
        batch_col=batch_col,
        query_batch=query_batch,
        ctrl_batch=ctrl_batch,
        perturbation_type="shift",
        use_rep_shift="X_pca",
    )

    # Checks
    assert any(adata_sim.obs[adata_sim.obs[annotation_col].isin(perturb_pop)]["dataset_group"] != "query")
    q_adata = adata_sim.obs[
        (adata_sim.obs[annotation_col].isin(perturb_pop)) & (adata_sim.obs["dataset_group"] == "query")
    ]
    assert sum(q_adata["OOR_state"] == 0) > 0


def test_multi_perturb():
    """
    Test that perturbed population is correctly removed when perturbation_type == 'remove
    """
    annotation_col = "cell_type"
    perturb_pop = ["Myeloid", "Fibroblast"]
    batch_col = "donor"
    query_batch = ["D4", "D6"]
    ctrl_batch = ["D2", "D3"]

    # Log normalize adata heart
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.pp.pca(adata, n_comps=10)

    adata_sim = adata.copy()
    adata_sim = simulate_query_reference(
        adata_sim,
        query_annotation=perturb_pop,
        annotation_col=annotation_col,
        batch_col=batch_col,
        query_batch=query_batch,
        ctrl_batch=ctrl_batch,
        perturbation_type="shift",
        use_rep_shift="X_pca",
    )

    # Checks
    assert any(adata_sim.obs[adata_sim.obs[annotation_col].isin(perturb_pop)]["dataset_group"] != "query")
    q_adata = adata_sim.obs[
        (adata_sim.obs[annotation_col].isin(perturb_pop)) & (adata_sim.obs["dataset_group"] == "query")
    ]
    assert sum(q_adata["OOR_state"] == 0) > 0

    adata_sim = adata.copy()
    with pytest.raises(AssertionError):
        adata_sim = simulate_query_reference(
            adata_sim,
            query_annotation=["Myeloid"],
            annotation_col=annotation_col,
            batch_col=batch_col,
            query_batch=query_batch,
            ctrl_batch=ctrl_batch,
            perturbation_type=["shift", "shift"],
            use_rep_shift="X_pca",
        )

    with pytest.raises(AssertionError):
        adata_sim = simulate_query_reference(
            adata_sim,
            query_annotation=["Myeloid", "Fibroblast"],
            annotation_col=annotation_col,
            batch_col=batch_col,
            query_batch=query_batch,
            ctrl_batch=ctrl_batch,
            perturbation_type=["shift"],
            use_rep_shift="X_pca",
        )

    adata_sim = adata.copy()
    adata_sim = simulate_query_reference(
        adata_sim,
        query_annotation=["Myeloid", "Fibroblast"],
        annotation_col=annotation_col,
        batch_col=batch_col,
        query_batch=query_batch,
        ctrl_batch=ctrl_batch,
        perturbation_type=["shift", "remove"],
        use_rep_shift="X_pca",
    )

    # Checks
    assert all(adata_sim.obs[adata_sim.obs[annotation_col] == "Fibroblast"]["dataset_group"] == "query")
    assert any(adata_sim.obs[adata_sim.obs[annotation_col] == "Myeloid"]["dataset_group"] != "query")
    q_adata = adata_sim.obs[(adata_sim.obs[annotation_col] == "Myeloid") & (adata_sim.obs["dataset_group"] == "query")]
    assert sum(q_adata["OOR_state"] == 0) > 0
