import cna
from anndata import AnnData
from multianndata import MultiAnnData


def run_cna(adata_design: AnnData, query_group: str, reference_group: str, sample_col: str = "sample_id"):
    """
    Run MELD to compute probability estimate per condition.

    Following tutorial in https://nbviewer.org/github/yakirr/cna/blob/master/demo/demo.ipynb

    Parameters:
    ------------
    adata_design : AnnData
        AnnData object of disease and reference cells to compare
    query_group : str
        Name of query group in adata_design.obs['dataset_group']
    reference_group : str
        Name of reference group in adata_design.obs['dataset_group']
    sample_col : str
        Name of column in adata_design.obs to use as sample ID
    """
    adata_design = MultiAnnData(adata_design, sampleid=sample_col)
    adata_design.obs["dataset_group"] = adata_design.obs["dataset_group"].astype("category")
    adata_design.obs["dataset_group_code"] = (
        adata_design.obs["dataset_group"].cat.reorder_categories([reference_group, query_group]).cat.codes
    )
    adata_design.obs_to_sample(["dataset_group_code"])

    res = cna.tl.association(adata_design, adata_design.samplem.dataset_group_code)

    adata_design.obs["CNA_ncorrs"] = res.ncorrs

    return None
