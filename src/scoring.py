"""Proper scoring rules and Diebold-Mariano comparison utilities.

Scoring rules implemented
-------------------------
log_score : Whole-density log predictive score (negative KLIC contribution).
csl_score : Censored Likelihood Score for left-tail evaluation.

Comparison utilities
--------------------
compute_score_differential : Pairwise differential + DM test.
build_comparison_table     : Table 1-style mean-diff and t-stat matrices.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .models import student_t_logpdf, student_t_logsf
from .utils import dm_test


# ── Whole-density score ───────────────────────────────────────────────────────

def log_score(y_realized: float, mu: float, h_forecast: float, nu: float) -> float:
    """Log predictive score for whole-density evaluation.

    Equal to the log predictive density evaluated at the realised return.
    Averaging over the evaluation window gives the mean log score (higher = better).

    Parameters
    ----------
    y_realized  : Realised return (decimal).
    mu          : Conditional mean forecast.
    h_forecast  : One-step-ahead conditional variance forecast (decimal^2).
    nu          : Student-t degrees of freedom.

    Returns
    -------
    float
        log f(y_realized | mu, h_forecast, nu).
    """
    return student_t_logpdf(y_realized, mu, h_forecast, nu)


# ── Censored Likelihood Score ─────────────────────────────────────────────────

def csl_score(
    y_realized: float,
    mu: float,
    h_forecast: float,
    nu: float,
    threshold: float,
) -> float:
    """Censored Likelihood Score (CSL) for left-tail evaluation.

    Scoring rule:
        CSL = log f(y)        if y <= r  (observation falls in the tail)
            = log(1 - F(r))   if y >  r  (observation is right-censored)

    where r = threshold (e.g. -0.025 for a -2.5 % VaR level in decimal returns).
    The right-censored term uses the numerically stable log survival function.

    Parameters
    ----------
    y_realized  : Realised return (decimal).
    mu          : Conditional mean forecast.
    h_forecast  : One-step-ahead conditional variance forecast (decimal^2).
    nu          : Student-t degrees of freedom.
    threshold   : Censoring point r (typically a negative VaR quantile).

    Returns
    -------
    float
        CSL score contribution for this observation (higher = better).
    """
    if y_realized <= threshold:
        return student_t_logpdf(y_realized, mu, h_forecast, nu)
    else:
        return student_t_logsf(threshold, mu, h_forecast, nu)


# ── Pairwise score differential and DM test ───────────────────────────────────

def compute_score_differential(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> Tuple[np.ndarray, float, float, float]:
    """Compute pairwise score differential and apply the Diebold-Mariano test.

    Parameters
    ----------
    scores_a : Score series for model A.
    scores_b : Score series for model B.

    Returns
    -------
    (d, mean_d, t_stat, p_value)
        d       : Element-wise differential scores_a - scores_b.
        mean_d  : Mean of d; positive means model A outperforms model B.
        t_stat  : DM test statistic (Newey-West HAC standard error).
        p_value : Two-sided p-value under the standard normal.
    """
    d = scores_a - scores_b
    mean_d = float(np.mean(d))
    t_stat, p_value = dm_test(d)
    return d, mean_d, t_stat, p_value


# ── Pairwise comparison table ─────────────────────────────────────────────────

def build_comparison_table(
    all_scores: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build Table 1-style pairwise comparison matrices.

    For each ordered pair (model i, model j) the entry records
    mean(score_i - score_j) and the corresponding DM t-statistic.
    Positive values indicate that model i outperforms model j.
    Diagonal entries are set to NaN (model compared with itself).

    Parameters
    ----------
    all_scores : Mapping from model name to its array of per-period scores.

    Returns
    -------
    (mean_diff_table, tstat_table)
        Both are DataFrames indexed and columned by model name.
        mean_diff_table[i][j] = mean(score_i - score_j).
        tstat_table[i][j]     = DM t-statistic for the same differential.
    """
    models = list(all_scores.keys())
    n = len(models)

    mean_diff = pd.DataFrame(
        np.zeros((n, n)), index=models, columns=models, dtype=float
    )
    tstat_mat = pd.DataFrame(
        np.zeros((n, n)), index=models, columns=models, dtype=float
    )

    for m_i in models:
        for m_j in models:
            if m_i == m_j:
                mean_diff.loc[m_i, m_j] = np.nan
                tstat_mat.loc[m_i, m_j] = np.nan
            else:
                d = all_scores[m_i] - all_scores[m_j]
                mean_d = float(np.mean(d))
                t_stat, _ = dm_test(d)
                mean_diff.loc[m_i, m_j] = mean_d
                tstat_mat.loc[m_i, m_j] = t_stat

    return mean_diff, tstat_mat
