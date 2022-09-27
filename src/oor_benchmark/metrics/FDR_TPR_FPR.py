import pandas as pd
from anndata import AnnData
from pandas import DataFrame

from .utils import _get_sample_adata, make_OOR_per_group


def FDR_TPR_FPR(adata: AnnData) -> DataFrame:
    """Calculate sensitivity and specificity metrics.

    These metrics use the significance/confidence of the OOR state prediction to compute True positive rate (TPR),
    False positive rate (FPR), and False discovery rate (FDR).

    Parameters:
    -----------
    adata: AnnData
        AnnData object after running method

    Returns:
    --------
    DataFrame storing TPR, FPR and FDR
    """
    if "OOR_state_group" not in _get_sample_adata(adata).var:
        make_OOR_per_group(adata)

    out_df = _get_sample_adata(adata).var[["OOR_signif", "OOR_state_group"]]
    out_df = out_df.astype(bool)

    out_df["TP"] = out_df["OOR_state_group"] & (out_df["OOR_signif"])
    out_df["FN"] = out_df["OOR_state_group"] & (~out_df["OOR_signif"])
    out_df["FP"] = (~out_df["OOR_state_group"]) & (out_df["OOR_signif"])
    out_df["TN"] = (~out_df["OOR_state_group"]) & (~out_df["OOR_signif"])

    tpr_df = pd.DataFrame(out_df[["TP", "FP", "FN", "TN"]].sum()).T
    tpr_df["TPR"] = tpr_df["TP"] / (tpr_df["TP"] + tpr_df["FN"])
    tpr_df["FPR"] = tpr_df["FP"] / (tpr_df["FP"] + tpr_df["TN"])
    tpr_df["FDR"] = tpr_df["FP"] / (tpr_df["FP"] + tpr_df["TP"])
    tpr_df.loc[tpr_df["FDR"].isna(), "FDR"] = 0
    # return out_df
    return tpr_df
