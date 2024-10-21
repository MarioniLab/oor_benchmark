from typing import Union

import numpy as np
import pandas as pd
from anndata import AnnData
from pandas import DataFrame
from sklearn.metrics import auc, precision_recall_curve
from sklearn.utils import resample

from .utils import _get_sample_adata, make_OOR_per_group


def auprc(oor_results: Union[AnnData, DataFrame], return_curve: bool = False, 
          bootstrap_iterations: Union[int, None] = None,
          bootstrap_seed: int = 2023
          ) -> DataFrame:
    """Calculate area under precision-recall curve for OOR state detection.

    This metric doesn't use the significance/confidence of the OOR state prediction

    Parameters:
    -----------
    oor_results: AnnData or DataFrame
        AnnData object after running method (storing results in oor_results.uns['sample_adata'].var) or DataFrame of results
    return_curve: bool
        Return precision-recall curve (default: False)
    bootstrap_iterations: int or None
        Number of bootstrap iterations to run (default: None, no bootstrap). If not none, 95% confidence intervals for AUPRC based 
        on bootstrap sampling with replacement will be returned.
    bootstrap_seed: int
        Seed for bootstrap sampling (default: 2023)

    Returns:
    --------
    DataFrame storing AUPRC and no-skill threshold, if return_curve is False
    DataFrame of precision, recall, AUPRC and no-skill threshold, if return_curve is True
    If bootstrap_iterations is not None, AUPRC confidence intervals will be returned.
    """
    if isinstance(oor_results, AnnData):
        if "OOR_state_group" not in _get_sample_adata(oor_results).var:
            make_OOR_per_group(oor_results)

        out_df = _get_sample_adata(oor_results).var[["OOR_score", "OOR_state_group"]]
    else:
        out_df = oor_results.copy()
    precision, recall, _ = precision_recall_curve(out_df.OOR_state_group, out_df.OOR_score)
    no_skill = sum(out_df.OOR_state_group) / out_df.shape[0]
    AUC = auc(recall, precision)
    if return_curve:
        AUPRC_df = pd.DataFrame(np.vstack([recall, precision]), index=["Recall", "Precision"]).T
        AUPRC_df["AUPRC"] = AUC
        AUPRC_df["no_skill_thresh"] = no_skill
    else:
        AUPRC_df = pd.DataFrame(index=["AUPRC", "no_skill_thresh"]).T
        AUPRC_df.loc[0, "AUPRC"] = AUC
        AUPRC_df.loc[0, "no_skill_thresh"] = no_skill

    # Run bootstrap
    if bootstrap_iterations is not None:
        bs_AUPRCS = []
        print("Running bootstrap with {} iterations".format(bootstrap_iterations))
        for i in range(bootstrap_iterations):
            bs_OOR_score, bs_OOR_state_group = resample(out_df.OOR_score.values, out_df.OOR_state_group.values, 
                            replace = True,
                            random_state = bootstrap_seed + i, 
                            stratify = out_df.OOR_state_group.values)
            bs_precision, bs_recall, _ = precision_recall_curve(bs_OOR_state_group, bs_OOR_score)
            bs_AUC = auc(bs_recall, bs_precision)
            bs_AUPRCS.append(bs_AUC)
        # Add confidence interval
        AUPRC_df['CI_lower'] = np.percentile(pd.Series(bs_AUPRCS).sort_values(), q=2.5)
        AUPRC_df['CI_upper'] = np.percentile(pd.Series(bs_AUPRCS).sort_values(), q=97.5)
        
    return AUPRC_df
