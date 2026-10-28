"""Experiment 2: Left-tail density forecast comparison using Censored Likelihood Score (CSL).

Reproduces Table 2 of the paper.  The CSL censors observations above the
threshold τ = −2.5 % and evaluates forecast quality only in the left tail.
Expected results from the paper: Bayesian methods significantly outperform
frequentist MLE (DM t-statistics ≈ 5), and asymmetric models (GJR, EGARCH)
outperform the symmetric GARCH(1,1).
"""

import sys
sys.path.insert(0, '/workspace/project')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

from data.fetch_data import fetch_index_data
from src.forecasting import rolling_forecast_window
from src.scoring import build_comparison_table

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = Path('/workspace/project/results')
RESULTS_DIR.mkdir(exist_ok=True)

N_FORECASTS = 600
WINDOW_YEARS = 10
WINDOW_DAYS = WINDOW_YEARS * 252       # 2 520 trading days
START_DATE_DATA = '1997-01-01'
END_DATE_DATA = '2010-04-20'
FORECAST_START = '2008-01-01'
FORECAST_END = '2010-04-19'
THRESHOLD = -0.025                      # left-tail censoring threshold (−2.5 %)

# Model display labels — same order as Table 2 in the paper
# method values match forecasting.py: 'MLE' and 'MCMC'
MODEL_METHODS = [
    ('garch',   'MLE',  'GARCH-MLE'),
    ('gjr',     'MLE',  'GJR-MLE'),
    ('egarch',  'MLE',  'EGARCH-MLE'),
    ('garch',   'MCMC', 'GARCH-Bayes'),
    ('gjr',     'MCMC', 'GJR-Bayes'),
    ('egarch',  'MCMC', 'EGARCH-Bayes'),
]


# ---------------------------------------------------------------------------
# Core routine
# ---------------------------------------------------------------------------

def run_experiment_2(
    index_name: str,
    returns: pd.Series,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run left-tail forecast comparison for *index_name* using CSL scoring.

    Parameters
    ----------
    index_name:
        Label used in output filenames (e.g. ``'sp500'``, ``'nikkei'``).
    returns:
        Full daily log-return series (decimal form) indexed by date.
    verbose:
        Print progress to stdout when ``True``.

    Returns
    -------
    tuple
        ``(results_df, mean_diff_table, tstat_table)``
    """
    print(f'\n=== Experiment 2: {index_name} Left-Tail (CSL) Forecast Comparison ===')
    print(f'    Threshold τ = {THRESHOLD:.1%}')

    # Rolling out-of-sample forecasts (csl_score column populated inside)
    results_df = rolling_forecast_window(
        y=returns.values,
        dates=returns.index,
        n_forecasts=N_FORECASTS,
        window_size=WINDOW_DAYS,
        model_types=['garch', 'gjr', 'egarch'],
        threshold=THRESHOLD,
        verbose=verbose,
    )

    # Persist raw per-forecast scores
    out_raw = RESULTS_DIR / f'exp2_{index_name}_raw_scores.csv'
    results_df.to_csv(out_raw, index=False)
    if verbose:
        print(f'Raw scores saved → {out_raw}')

    # Build per-model CSL score arrays for pairwise DM comparisons
    print(f'\nBuilding pairwise CSL comparison tables for {index_name} ...')
    scores_dict: dict[str, np.ndarray] = {}
    for model, method, label in MODEL_METHODS:
        mask = (results_df['model'] == model) & (results_df['method'] == method)
        scores = results_df.loc[mask, 'csl_score'].values
        if len(scores) > 0:
            scores_dict[label] = scores

    mean_diff_table, tstat_table = build_comparison_table(scores_dict)

    # Persist comparison tables
    mean_diff_table.to_csv(RESULTS_DIR / f'exp2_{index_name}_mean_diff.csv')
    tstat_table.to_csv(RESULTS_DIR / f'exp2_{index_name}_tstat.csv')

    print(f'\nTable 2 ({index_name}) - Mean CSL Differentials (positive = row > col):')
    print(mean_diff_table.to_string())
    print(f'\nTable 2 ({index_name}) - DM t-statistics:')
    print(tstat_table.to_string())

    return results_df, mean_diff_table, tstat_table


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('Fetching data ...')
    sp500_returns = fetch_index_data('sp500', START_DATE_DATA, END_DATE_DATA)
    nikkei_returns = fetch_index_data('nikkei', START_DATE_DATA, END_DATE_DATA)

    print(
        f'S&P 500 : {len(sp500_returns)} daily returns'
        f'  [{sp500_returns.index[0]}  →  {sp500_returns.index[-1]}]'
    )
    print(
        f'Nikkei  : {len(nikkei_returns)} daily returns'
        f'  [{nikkei_returns.index[0]}  →  {nikkei_returns.index[-1]}]'
    )

    sp500_results, sp500_mean, sp500_tstat = run_experiment_2(
        'sp500', sp500_returns
    )
    nikkei_results, nikkei_mean, nikkei_tstat = run_experiment_2(
        'nikkei', nikkei_returns
    )

    print('\nExperiment 2 complete.  Results saved to results/')
