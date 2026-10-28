"""Experiment 1: Whole-density forecast comparison using KLIC log-scores.

Reproduces Table 1 of the paper comparing frequentist MLE vs Bayesian MCMC
estimation of GARCH(1,1), GJR(1,1), and EGARCH(1,1) on S&P 500 and Nikkei 225
daily returns.  600 one-step-ahead density forecasts are evaluated via the
Kullback–Leibler Information Criterion (KLIC) log-score.  Pairwise Diebold–
Mariano tests with Newey–West standard errors are reported.
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
THRESHOLD = -0.025                      # not used in Exp 1 scoring but forwarded

# Model display labels in the same order used throughout the paper
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

def run_experiment_1(
    index_name: str,
    returns: pd.Series,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run whole-density forecast comparison for *index_name*.

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
    print(f'\n=== Experiment 1: {index_name} Whole-Density Forecast Comparison ===')

    # Rolling out-of-sample forecasts
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
    out_raw = RESULTS_DIR / f'exp1_{index_name}_raw_scores.csv'
    results_df.to_csv(out_raw, index=False)
    if verbose:
        print(f'Raw scores saved → {out_raw}')

    # Build per-model score arrays for pairwise DM comparisons
    print(f'\nBuilding pairwise comparison tables for {index_name} ...')
    scores_dict: dict[str, np.ndarray] = {}
    for model, method, label in MODEL_METHODS:
        mask = (results_df['model'] == model) & (results_df['method'] == method)
        scores = results_df.loc[mask, 'log_score'].values
        if len(scores) > 0:
            scores_dict[label] = scores

    mean_diff_table, tstat_table = build_comparison_table(scores_dict)

    # Persist comparison tables
    mean_diff_table.to_csv(RESULTS_DIR / f'exp1_{index_name}_mean_diff.csv')
    tstat_table.to_csv(RESULTS_DIR / f'exp1_{index_name}_tstat.csv')

    print(f'\nTable 1 ({index_name}) - Mean Log-Score Differentials (positive = row > col):')
    print(mean_diff_table.to_string())
    print(f'\nTable 1 ({index_name}) - DM t-statistics:')
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

    sp500_results, sp500_mean, sp500_tstat = run_experiment_1(
        'sp500', sp500_returns
    )
    nikkei_results, nikkei_mean, nikkei_tstat = run_experiment_1(
        'nikkei', nikkei_returns
    )

    print('\nExperiment 1 complete.  Results saved to results/')
