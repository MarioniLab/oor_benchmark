from scvi.data import heart_cell_atlas_subsampled

from oor_benchmark.datasets.simulation import simulate_query_reference

adata = heart_cell_atlas_subsampled()


def test_query_specific_pop():
    """
    Test that perturbed population is correctly removed when perturbation_type == 'remove
    """
    annotation_col = "cell_type"
    perturb_pop = ["Myeloid"]
    batch_obs = "donor"
    query_batch = ["D4", "D6"]
    ctrl_batch = ["D2", "D3"]

    adata_sim = simulate_query_reference(
        adata,
        query_annotation=perturb_pop,
        annotation_col=annotation_col,
        batch_obs=batch_obs,
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
    batch_obs = "donor"
    query_batch = ["D4", "D6"]
    ctrl_batch = ["D2", "D3"]

    adata_sim = simulate_query_reference(
        adata,
        query_annotation=perturb_pop,
        annotation_col=annotation_col,
        batch_obs=batch_obs,
        query_batch=query_batch,
        ctrl_batch=ctrl_batch,
        perturbation_type="remove",
    )

    # Checks
    assert all(adata_sim.obs[[batch_obs, "dataset_group"]].groupby(batch_obs).nunique() == 1)
