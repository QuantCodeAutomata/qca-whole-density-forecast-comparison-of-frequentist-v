"""Master runner: Whole-density and left-tail forecast comparison.

Runs both experiments from a single rolling-window forecast pipeline:
  * Exp 1: Whole-density (log-score / KLIC)
  * Exp 2: Left-tail (censored likelihood score, threshold = -2.5 %)

Usage
-----
    python run_all.py [--fast] [--workers N]

Flags
-----
--fast      Use n_burn=500, n_draws=2000 (quick validation run, ~10 min).
            Default: n_burn=5000, n_draws=25000 (paper specification, ~60 min).
--workers N Number of parallel worker processes (default: CPU count capped at 4).

Implementation notes
--------------------
* The same forecast DataFrame contains both log_score and csl_score columns,
  so Exp 1 and Exp 2 share one rolling-window pass per index.
* Results are cached to data/<index>_forecast_cache.pkl so the run can be
  resumed after interruption.
* The Newey-West bandwidth follows the Andrews (1991) rule applied to the
  score differential series: L = floor(4 * (T/100)^(2/9)), documented in
  src/utils.py::dm_test.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from data.fetch_data import fetch_index_data
from src.parallel_forecast import parallel_rolling_forecast
from src.scoring import build_comparison_table
from src.visualisation import (
    plot_cumulative_scores,
    plot_dm_heatmap,
    plot_pairwise_table,
    plot_score_timeseries,
)

# ── Constants ─────────────────────────────────────────────────────────────────

RESULTS_DIR = Path('results')
DATA_DIR    = Path('data')
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

N_FORECASTS     = 600
WINDOW_DAYS     = 10 * 252          # ≈2520 trading days (10-year rolling window)
THRESHOLD       = -0.025            # −2.5 % in decimal form
FORECAST_START  = '2008-01-01'
FORECAST_END    = '2010-04-19'
DATA_START      = '1997-01-01'
DATA_END        = '2010-04-20'

MODEL_METHODS = [
    ('garch',  'MLE',  'GARCH-MLE'),
    ('gjr',    'MLE',  'GJR-MLE'),
    ('egarch', 'MLE',  'EGARCH-MLE'),
    ('garch',  'MCMC', 'GARCH-Bayes'),
    ('gjr',    'MCMC', 'GJR-Bayes'),
    ('egarch', 'MCMC', 'EGARCH-Bayes'),
]


# ── Data ──────────────────────────────────────────────────────────────────────

def load_returns(index_name: str) -> pd.Series:
    """Fetch and validate daily log returns for *index_name*."""
    print(f'  Fetching {index_name} data [{DATA_START} → {DATA_END}] ...')
    ret = fetch_index_data(index_name, DATA_START, DATA_END)
    ret.index = pd.DatetimeIndex(ret.index)
    ret = ret.sort_index().dropna()
    print(f'  {index_name}: {len(ret)} obs  [{ret.index[0].date()} → {ret.index[-1].date()}]')
    return ret


# ── Comparison tables ─────────────────────────────────────────────────────────

def _extract_scores(df: pd.DataFrame, score_col: str) -> dict[str, np.ndarray]:
    """Extract per-model score arrays aligned to the same date grid."""
    # Find the common date set (dates where all 6 model×method combos have data)
    valid_dates: set | None = None
    for model, method, label in MODEL_METHODS:
        mask = (df['model'] == model) & (df['method'] == method)
        rows = df.loc[mask, ['date', score_col]].dropna(subset=[score_col])
        s = set(rows['date'].tolist())
        valid_dates = s if valid_dates is None else valid_dates & s

    scores: dict[str, np.ndarray] = {}
    for model, method, label in MODEL_METHODS:
        mask = (df['model'] == model) & (df['method'] == method)
        rows = df.loc[mask, ['date', score_col]].copy()
        rows = rows[rows['date'].isin(valid_dates)].sort_values('date')
        scores[label] = rows[score_col].values.astype(float)

    return scores


def build_and_print_tables(
    df: pd.DataFrame,
    score_col: str,
    index_name: str,
    exp_tag: str,
    score_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build pairwise comparison tables and print them."""
    scores = _extract_scores(df, score_col)
    n_periods = next(iter(scores.values())).shape[0] if scores else 0
    print(f'\n{score_label} pairwise comparison — {index_name} (T={n_periods})')

    mean_diff, tstat = build_comparison_table(scores)

    # Annotate significance in the t-stat table
    sig_tstat = tstat.copy().round(3)
    print(f'  Mean differential (positive = row > col):')
    print(mean_diff.round(4).to_string())
    print(f'\n  DM t-statistics (|t|>1.96 → 5%, |t|>2.576 → 1%):')
    print(sig_tstat.to_string())

    mean_diff.to_csv(RESULTS_DIR / f'{exp_tag}_{index_name}_mean_diff.csv')
    tstat.to_csv(RESULTS_DIR / f'{exp_tag}_{index_name}_tstat.csv')
    return mean_diff, tstat


# ── Plots ─────────────────────────────────────────────────────────────────────

def _make_plots(
    df: pd.DataFrame,
    index_name: str,
    mean_diff_ls: pd.DataFrame,
    tstat_ls: pd.DataFrame,
    mean_diff_csl: pd.DataFrame,
    tstat_csl: pd.DataFrame,
) -> None:
    """Generate and save all plots for one index."""
    tag = index_name.lower().replace(' ', '_').replace('&', '')

    # Cumulative log-score
    try:
        fig = plot_cumulative_scores(df, 'log_score', MODEL_METHODS, index_name)
        fig.savefig(RESULTS_DIR / f'{tag}_cumulative_logscore.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'  Warning: cumulative log-score plot failed: {e}')

    # Cumulative CSL
    try:
        fig = plot_cumulative_scores(df, 'csl_score', MODEL_METHODS, index_name)
        fig.savefig(RESULTS_DIR / f'{tag}_cumulative_csl.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'  Warning: cumulative CSL plot failed: {e}')

    # DM heatmap — log score
    try:
        fig = plot_dm_heatmap(tstat_ls, f'{index_name} Log-Score DM t-stats')
        fig.savefig(RESULTS_DIR / f'{tag}_dm_logscore.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'  Warning: DM log-score heatmap failed: {e}')

    # DM heatmap — CSL
    try:
        fig = plot_dm_heatmap(tstat_csl, f'{index_name} CSL DM t-stats')
        fig.savefig(RESULTS_DIR / f'{tag}_dm_csl.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'  Warning: DM CSL heatmap failed: {e}')

    print(f'  Plots saved to {RESULTS_DIR}/')


# ── RESULTS.md ────────────────────────────────────────────────────────────────

def _significance_marker(tstat: float) -> str:
    a = abs(tstat)
    if a > 2.576:
        return '**'
    if a > 1.960:
        return '*'
    return ''


def write_results_md(all_results: dict, n_burn: int = 5000, n_draws: int = 25000) -> None:
    """Write a comprehensive RESULTS.md summarising all findings."""
    mode_note = (
        f'n_burn={n_burn}, n_draws={n_draws}'
        + (' *(FAST mode — use --full for paper-spec 5000/25000)*' if n_draws < 25000 else '')
    )
    lines = [
        '# RESULTS — Bayesian vs Frequentist GARCH Density Forecast Comparison',
        '',
        '## Experiment Setup',
        '- **Indices**: S&P 500 and Nikkei 225',
        f'- **Out-of-sample period**: {FORECAST_START} to {FORECAST_END}',
        f'- **Number of forecasts**: {N_FORECASTS} (rolling {WINDOW_DAYS//252}-year window)',
        '- **Models**: GARCH(1,1), GJR(1,1), EGARCH(1,1) — each with MLE and Bayes',
        '- **Student-t innovations**: scale s²=h(ν−2)/ν, so Var(ε|I)=h',
        '- **Bayesian prior for ν**: p(ν)=0.05·exp[−0.05·(ν−2)], ν>2',
        f'- **MCMC**: Independence-chain MH, {mode_note}',
        '- **Scoring**: Log-score (Exp 1) and CSL with threshold r=−2.5% (Exp 2)',
        '- **DM test**: Newey-West HAC SE, bandwidth L=floor(4·(T/100)^(2/9))',
        '',
        '---',
        '',
    ]

    for index_name, res in all_results.items():
        lines += [
            f'## {index_name}',
            '',
            '### Exp 1 — Whole-Density (Log-Score) Pairwise DM t-statistics',
            '*(positive = row model outperforms column model)*',
            '',
        ]
        tstat_ls = res.get('tstat_ls')
        if tstat_ls is not None:
            lines.append('```')
            lines.append(tstat_ls.round(3).to_string())
            lines.append('```')
            lines.append('')
            # Key same-model comparisons
            lines.append('**Same-model (Bayes vs MLE) log-score t-stats:**')
            for m_name, mle_label, bay_label in [
                ('GARCH', 'GARCH-MLE', 'GARCH-Bayes'),
                ('GJR',   'GJR-MLE',   'GJR-Bayes'),
                ('EGARCH','EGARCH-MLE','EGARCH-Bayes'),
            ]:
                try:
                    t = float(tstat_ls.loc[bay_label, mle_label])
                    sig = _significance_marker(t)
                    lines.append(f'  - {m_name} Bayes vs MLE: t={t:.3f}{sig}')
                except Exception:
                    pass
            lines.append('')

        lines += [
            '### Exp 2 — Left-Tail (CSL, r=−2.5%) Pairwise DM t-statistics',
            '*(positive = row model outperforms column model)*',
            '',
        ]
        tstat_csl = res.get('tstat_csl')
        if tstat_csl is not None:
            lines.append('```')
            lines.append(tstat_csl.round(3).to_string())
            lines.append('```')
            lines.append('')
            lines.append('**Same-model (Bayes vs MLE) CSL t-stats (paper target ≈ 5.0):**')
            for m_name, mle_label, bay_label in [
                ('GARCH', 'GARCH-MLE', 'GARCH-Bayes'),
                ('GJR',   'GJR-MLE',   'GJR-Bayes'),
                ('EGARCH','EGARCH-MLE','EGARCH-Bayes'),
            ]:
                try:
                    t = float(tstat_csl.loc[bay_label, mle_label])
                    sig = _significance_marker(t)
                    lines.append(f'  - {m_name} Bayes vs MLE: t={t:.3f}{sig}')
                except Exception:
                    pass
            lines.append('')
        lines.append('---')
        lines.append('')

    lines += [
        '## Key Findings',
        '',
        '### Experiment 1 (Whole-Density)',
        'Expected: Bayesian vs MLE t-stats near zero, insignificant (p > 0.05).',
        'This would reproduce the paper\'s finding that whole-density accuracy',
        'is not materially improved by Bayesian estimation.',
        '',
        '### Experiment 2 (Left-Tail, CSL)',
        'Expected: Bayesian vs MLE t-stats ≈ 5 (strongly significant, p < 0.01).',
        'The paper reports t-stats of approximately 5.274, 5.480, 4.227 for',
        'S&P 500 (GARCH, GJR, EGARCH) and 5.277, 5.054, 5.422 for Nikkei 225.',
        '',
        '### Significance key',
        '- `*`  = significant at 5% (|t| > 1.96)',
        '- `**` = significant at 1% (|t| > 2.576)',
        '',
        '## File Index',
        '| File | Description |',
        '|------|-------------|',
        '| exp1_*_mean_diff.csv | Mean log-score differentials (Table 1 panel A) |',
        '| exp1_*_tstat.csv     | DM t-statistics for log-score (Table 1 panel A) |',
        '| exp2_*_mean_diff.csv | Mean CSL differentials (Table 1 panel B) |',
        '| exp2_*_tstat.csv     | DM t-statistics for CSL (Table 1 panel B) |',
        '| *_cumulative_*.png   | Cumulative score plots |',
        '| *_dm_*.png           | DM t-stat heatmaps |',
    ]

    md_path = RESULTS_DIR / 'RESULTS.md'
    md_path.write_text('\n'.join(lines) + '\n')
    print(f'\nResults summary written → {md_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main(fast: bool = False, n_workers: int = 4) -> None:
    n_burn  = 500   if fast else 5000
    n_draws = 2000  if fast else 25000
    mode_tag = 'FAST' if fast else 'FULL'

    print(f'\n{"="*65}')
    print(f'  GARCH Density Forecast Comparison  [{mode_tag} MODE]')
    print(f'  n_burn={n_burn}, n_draws={n_draws}, workers={n_workers}')
    print(f'{"="*65}\n')

    indices = [('sp500', 'S&P 500'), ('nikkei', 'Nikkei 225')]
    all_results: dict = {}
    t_global = time.time()

    for index_key, index_label in indices:
        print(f'\n{"─"*55}')
        print(f'  Index: {index_label}')
        print(f'{"─"*55}')
        t0 = time.time()

        # ── 1. Fetch data ─────────────────────────────────────────────────
        returns = load_returns(index_key)

        # ── 2. Rolling forecast (parallel, cached) ────────────────────────
        cache_path = DATA_DIR / f'{index_key}_forecast_cache.pkl'
        print(f'\n  Running parallel rolling forecast ...')
        df = parallel_rolling_forecast(
            y=returns.values,
            dates=returns.index,
            n_forecasts=N_FORECASTS,
            window_size=WINDOW_DAYS,
            model_types=['garch', 'gjr', 'egarch'],
            threshold=THRESHOLD,
            n_burn=n_burn,
            n_draws=n_draws,
            random_state=42,
            n_workers=n_workers,
            verbose=True,
            cache_path=cache_path,
            forecast_start=FORECAST_START,
            forecast_end=FORECAST_END,
        )

        # Save full forecast DataFrame
        df.to_csv(RESULTS_DIR / f'{index_key}_all_forecasts.csv', index=False)

        # ── 3. Exp 1 — Whole-density (log-score) tables ───────────────────
        print(f'\n  [Exp 1] Building whole-density comparison tables ...')
        mean_diff_ls, tstat_ls = build_and_print_tables(
            df, 'log_score', index_label, 'exp1', 'Log-Score (KLIC)'
        )

        # ── 4. Exp 2 — Left-tail (CSL) tables ────────────────────────────
        print(f'\n  [Exp 2] Building CSL comparison tables ...')
        mean_diff_csl, tstat_csl = build_and_print_tables(
            df, 'csl_score', index_label, 'exp2', 'CSL (r=−2.5%)'
        )

        # ── 5. Plots ──────────────────────────────────────────────────────
        print(f'\n  Generating plots ...')
        _make_plots(df, index_label, mean_diff_ls, tstat_ls, mean_diff_csl, tstat_csl)

        all_results[index_label] = {
            'df': df,
            'mean_diff_ls':  mean_diff_ls,
            'tstat_ls':      tstat_ls,
            'mean_diff_csl': mean_diff_csl,
            'tstat_csl':     tstat_csl,
        }

        print(f'\n  {index_label} completed in {time.time()-t0:.1f}s')

    # ── 6. Write RESULTS.md ───────────────────────────────────────────────────
    write_results_md(all_results, n_burn=n_burn, n_draws=n_draws)

    print(f'\n{"="*65}')
    print(f'  All experiments finished in {time.time()-t_global:.1f}s')
    print(f'  Results in: {RESULTS_DIR.resolve()}')
    print(f'{"="*65}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run GARCH density forecast comparison.')
    parser.add_argument('--fast', action='store_true',
                        help='Quick run: n_burn=500, n_draws=2000.')
    parser.add_argument('--workers', type=int,
                        default=min(4, os.cpu_count() or 1),
                        help='Parallel worker count (default: min(4, CPUs)).')
    args = parser.parse_args()
    main(fast=args.fast, n_workers=args.workers)
