"""Visualisation utilities for GARCH density forecast comparison experiments.

All public functions return matplotlib Figure objects so the caller controls
saving (path, dpi, format).  Legacy file-saving helpers are preserved for
backward compatibility.

All plots use matplotlib; seaborn is used for colour palettes / style only.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')           # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style='whitegrid', palette='muted')

_RESULTS_DIR = Path('/workspace/project/results')

_LINESTYLES = {'MLE': '--', 'MCMC': '-'}
_COLOURS = {
    'garch':  '#1f77b4',
    'gjr':    '#ff7f0e',
    'egarch': '#2ca02c',
}
_SIG_5PCT = 1.96
_SIG_1PCT = 2.576


# ── Heatmap of DM t-statistics (returns Figure) ────────────────────────────────

def plot_dm_heatmap(
    tstat_df: pd.DataFrame,
    title: str,
    sig_5pct: float = _SIG_5PCT,
    sig_1pct: float = _SIG_1PCT,
) -> plt.Figure:
    """Heatmap of pairwise DM t-statistics; returns a Figure.

    Parameters
    ----------
    tstat_df : Square DataFrame of t-statistics (diagonal = NaN).
    title    : Plot title.
    sig_5pct : |t| threshold for 5 % significance.
    sig_1pct : |t| threshold for 1 % significance.
    """
    n = len(tstat_df)
    fig, ax = plt.subplots(figsize=(max(8, n * 1.5), max(6, n * 1.3)))

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    arr = tstat_df.values.astype(float)
    vmax = float(np.nanmax(np.abs(arr))) if np.any(np.isfinite(arr)) else 1.0
    vmax = max(vmax, 1.0)

    sns.heatmap(
        tstat_df,
        annot=False,
        cmap=cmap,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.5,
        linecolor='grey',
        ax=ax,
        cbar_kws={'label': 'DM t-statistic', 'shrink': 0.8},
    )

    for i, row in enumerate(tstat_df.index):
        for j, col in enumerate(tstat_df.columns):
            val = tstat_df.loc[row, col]
            if pd.isna(val):
                ax.text(j + 0.5, i + 0.5, '—', ha='center', va='center',
                        fontsize=11, color='#555555')
            else:
                stars = '**' if abs(val) > sig_1pct else ('*' if abs(val) > sig_5pct else '')
                text_col = 'white' if abs(val) > vmax * 0.7 else 'black'
                ax.text(j + 0.5, i + 0.5, f'{val:.2f}{stars}',
                        ha='center', va='center', fontsize=9, color=text_col)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel('Column model (denominator)', fontsize=10)
    ax.set_ylabel('Row model (numerator)', fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    return fig


# Alias used by exp scripts that import the old name
plot_tstat_heatmap_fig = plot_dm_heatmap


def plot_tstat_heatmap(
    tstat_df: pd.DataFrame,
    title: str,
    filename: str,
    results_dir: Path = _RESULTS_DIR,
    sig_5pct: float = _SIG_5PCT,
    sig_1pct: float = _SIG_1PCT,
) -> None:
    """Legacy file-saving wrapper around plot_dm_heatmap."""
    fig = plot_dm_heatmap(tstat_df, title, sig_5pct=sig_5pct, sig_1pct=sig_1pct)
    out = results_dir / filename
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ── Pairwise mean-diff table rendered as a heatmap (returns Figure) ────────────

def plot_pairwise_table(
    mean_diff_df: pd.DataFrame,
    title: str,
) -> plt.Figure:
    """Render the mean-difference matrix as a heatmap; returns a Figure."""
    n = len(mean_diff_df)
    fig, ax = plt.subplots(figsize=(max(8, n * 1.5), max(6, n * 1.3)))

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    arr = mean_diff_df.values.astype(float)
    vmax = float(np.nanmax(np.abs(arr))) if np.any(np.isfinite(arr)) else 1e-6
    vmax = max(vmax, 1e-6)

    sns.heatmap(
        mean_diff_df,
        annot=True,
        fmt='.5f',
        cmap=cmap,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.5,
        linecolor='grey',
        ax=ax,
        cbar_kws={'label': 'Mean score differential', 'shrink': 0.8},
        annot_kws={'size': 8},
    )

    ax.set_title(title, fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel('Column model (denominator)', fontsize=10)
    ax.set_ylabel('Row model (numerator)', fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    return fig


def plot_mean_diff_table(
    mean_diff_df: pd.DataFrame,
    title: str,
    filename: str,
    results_dir: Path = _RESULTS_DIR,
) -> None:
    """Legacy file-saving wrapper around plot_pairwise_table."""
    fig = plot_pairwise_table(mean_diff_df, title)
    out = results_dir / filename
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ── Cumulative score line plot (returns Figure) ────────────────────────────────

def plot_cumulative_scores(
    results_df: pd.DataFrame,
    score_col: str,
    model_methods: List[Tuple],
    index_name: str,
) -> plt.Figure:
    """Cumulative per-model score plot; returns a Figure.

    Parameters
    ----------
    results_df    : Rolling forecast DataFrame with columns
                    [date, model, method, log_score, csl_score].
    score_col     : ``'log_score'`` or ``'csl_score'``.
    model_methods : List of (model, method, label) tuples in display order.
    index_name    : Index label for the title (e.g. 'S&P 500').
    """
    fig, ax = plt.subplots(figsize=(13, 6))

    for model, method, label in model_methods:
        mask = (results_df['model'] == model) & (results_df['method'] == method)
        sub  = (
            results_df.loc[mask, ['date', score_col]]
            .dropna(subset=[score_col])
            .sort_values('date')
        )
        if sub.empty:
            continue
        dates  = pd.DatetimeIndex(sub['date'])
        cumsum = sub[score_col].values.cumsum()
        ls     = _LINESTYLES.get(method, '-')
        col    = _COLOURS.get(model, 'grey')
        ax.plot(dates, cumsum, linestyle=ls, color=col, linewidth=1.5, label=label)

    score_name = 'Log-Score (KLIC)' if score_col == 'log_score' else 'CSL (r=−2.5%)'
    ax.set_title(f'{index_name} — Cumulative {score_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel(f'Cumulative {score_col}', fontsize=10)
    ax.legend(fontsize=9, ncol=2, loc='best')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=30)
    plt.tight_layout()
    return fig


# ── Daily score time series (returns Figure) ───────────────────────────────────

def plot_score_timeseries(
    results_df: pd.DataFrame,
    score_col: str,
    model_methods: List[Tuple],
    index_name: str,
) -> plt.Figure:
    """Daily per-model score plot; returns a Figure.

    Parameters
    ----------
    results_df    : Rolling forecast DataFrame.
    score_col     : ``'log_score'`` or ``'csl_score'``.
    model_methods : List of (model, method, label) tuples.
    index_name    : Index label for the title.
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    for model, method, label in model_methods:
        mask = (results_df['model'] == model) & (results_df['method'] == method)
        sub  = (
            results_df.loc[mask, ['date', score_col]]
            .dropna(subset=[score_col])
            .sort_values('date')
        )
        if sub.empty:
            continue
        dates  = pd.DatetimeIndex(sub['date'])
        values = sub[score_col].values
        ls     = _LINESTYLES.get(method, '-')
        col    = _COLOURS.get(model, 'grey')
        ax.plot(dates, values, linestyle=ls, color=col, linewidth=0.8,
                alpha=0.7, label=label)

    score_name = 'Log-Score (KLIC)' if score_col == 'log_score' else 'CSL (r=−2.5%)'
    ax.set_title(f'{index_name} — Daily {score_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel(score_col, fontsize=10)
    ax.legend(fontsize=9, ncol=2, loc='best')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=30)
    plt.tight_layout()
    return fig


# ── Mean score differential bar chart (returns Figure) ────────────────────────

def plot_mean_diff_bar(
    mean_diff_df: pd.DataFrame,
    title: str,
    filename: Optional[str] = None,
    results_dir: Path = _RESULTS_DIR,
) -> plt.Figure:
    """Bar chart of mean score differentials vs. the first-column model.

    Returns a Figure; optionally also saves it when *filename* is provided.
    """
    models  = mean_diff_df.index.tolist()
    ref_col = mean_diff_df.columns[0]
    vals    = [float(mean_diff_df.loc[m, ref_col]) for m in models]
    colours = ['#2166ac' if v >= 0 else '#d73027' for v in vals]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(models, vals, color=colours, edgecolor='white', linewidth=0.8)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Mean score differential vs. {ref_col}', fontsize=10)
    ax.set_xticklabels(models, rotation=25, ha='right', fontsize=9)

    for bar, val in zip(bars, vals):
        ypos = bar.get_height() + (0.002 if val >= 0 else -0.006)
        ax.text(bar.get_x() + bar.get_width() / 2.0, ypos,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if filename is not None:
        out = results_dir / filename
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {out}')

    return fig
