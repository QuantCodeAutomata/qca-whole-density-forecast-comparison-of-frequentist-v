"""Parallel rolling-window forecasting via joblib (loky backend).

Each forecast origin is statistically independent, so the 600 origins are
distributed across CPU cores. We use joblib with the loky backend, which is
safe with numba JIT-compiled functions (avoids fork+numba deadlocks).

Numerical implementation notes
-------------------------------
* Random seed per origin: base_seed + origin_index * n_models + model_idx
  ensures reproducibility while avoiding correlated chains across windows.
* Partial result caching: if a .pkl cache file exists for this index, only
  missing origins are re-computed.
"""

from __future__ import annotations

import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats
from scipy.special import logsumexp

# Ensure the project root is importable inside worker processes
sys.path.insert(0, '/workspace/project')

from src import models as _models
from src.estimation import frequentist_estimate, bayesian_estimate, MODEL_PARAMS


# ── per-origin worker ─────────────────────────────────────────────────────────

def _score_bayesian(
    y_realized: float,
    threshold: float,
    draws: np.ndarray,
    mu_idx: int,
    nu_idx: int,
    h_forecasts: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """Compute Bayesian predictive log-score and CSL from posterior draws.

    Returns (mu_mean, nu_mean, h_fc_mean, log_score, csl_score).
    """
    mu_vec = draws[:, mu_idx]
    nu_vec = draws[:, nu_idx]
    h_vec  = h_forecasts

    valid = np.isfinite(h_vec) & (h_vec > 0.0) & (nu_vec > 2.0)
    if not np.any(valid):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    mu_v = mu_vec[valid]
    nu_v = nu_vec[valid]
    h_v  = h_vec[valid]
    M    = int(np.sum(valid))

    mu_mean  = float(np.mean(mu_v))
    nu_mean  = float(np.mean(nu_v))
    h_fc_mean = float(np.mean(h_v))

    s_vec    = np.sqrt(h_v * (nu_v - 2.0) / nu_v)
    log_dens = scipy.stats.t.logpdf(float(y_realized), df=nu_v, loc=mu_v, scale=s_vec)
    ls_bay   = float(logsumexp(log_dens) - np.log(M))

    if y_realized <= threshold:
        csl_vals = log_dens
    else:
        csl_vals = scipy.stats.t.logsf(float(threshold), df=nu_v, loc=mu_v, scale=s_vec)
    csl_bay = float(logsumexp(csl_vals) - np.log(M))

    return mu_mean, nu_mean, h_fc_mean, ls_bay, csl_bay


def _forecast_one_origin(
    t: int,
    y: np.ndarray,
    date_t,
    y_realized: float,
    window_size: int,
    model_types: List[str],
    threshold: float,
    n_burn: int,
    n_draws: int,
    base_seed: int,
    origin_index: int,
) -> List[Dict]:
    """Forecast a single rolling-window origin.

    Parameters
    ----------
    t            : Index of the forecast origin in the full return array.
    y            : Full return series.
    date_t       : Date label for this origin.
    y_realized   : Realized return at t (y[t]).
    window_size  : Number of in-sample observations to use.
    model_types  : List of model strings, e.g. ['garch', 'gjr', 'egarch'].
    threshold    : CSL censoring threshold (decimal, e.g. -0.025).
    n_burn       : MCMC burn-in draws.
    n_draws      : MCMC retained draws.
    base_seed    : Base random seed.
    origin_index : Index of this origin (for reproducible seeding).

    Returns
    -------
    List of row dicts (one per model × method).
    """
    import warnings; warnings.filterwarnings('ignore')
    training = y[t - window_size : t]
    rows: List[Dict] = []

    for m_idx, model_type in enumerate(model_types):
        model_seed = base_seed + origin_index * len(model_types) + m_idx

        # ── Frequentist MLE ─────────────────────────────────────────────────
        try:
            mle      = frequentist_estimate(training, model_type)
            mu_mle   = float(mle['params']['mu'])
            nu_mle   = float(mle['params']['nu'])
            h_fc_mle = float(mle['h_forecast'])
            ls_mle   = _models.student_t_logpdf(y_realized, mu_mle, h_fc_mle, nu_mle)
            if y_realized <= threshold:
                csl_mle = ls_mle
            else:
                csl_mle = _models.student_t_logsf(threshold, mu_mle, h_fc_mle, nu_mle)
        except Exception:
            mu_mle = nu_mle = h_fc_mle = ls_mle = csl_mle = np.nan

        rows.append({
            'date': date_t, 'y_realized': float(y_realized),
            'model': model_type, 'method': 'MLE',
            'mu': mu_mle, 'h_forecast': h_fc_mle, 'nu': nu_mle,
            'log_score': ls_mle, 'csl_score': csl_mle,
        })

        # ── Bayesian MCMC ───────────────────────────────────────────────────
        try:
            bay = bayesian_estimate(
                training, model_type,
                n_burn=n_burn, n_draws=n_draws, random_state=model_seed,
            )
            pnames   = bay['param_names']
            mu_idx_p = pnames.index('mu')
            nu_idx_p = pnames.index('nu')
            mu_bay, nu_bay, h_fc_bay, ls_bay, csl_bay = _score_bayesian(
                y_realized, threshold,
                bay['draws'], mu_idx_p, nu_idx_p, bay['h_forecasts'],
            )
        except Exception:
            mu_bay = nu_bay = h_fc_bay = ls_bay = csl_bay = np.nan

        rows.append({
            'date': date_t, 'y_realized': float(y_realized),
            'model': model_type, 'method': 'MCMC',
            'mu': mu_bay, 'h_forecast': h_fc_bay, 'nu': nu_bay,
            'log_score': ls_bay, 'csl_score': csl_bay,
        })

    return rows


# ── parallel rolling-window driver ────────────────────────────────────────────

def parallel_rolling_forecast(
    y: np.ndarray,
    dates,
    n_forecasts: int,
    window_size: int = 10 * 252,
    model_types: Optional[List[str]] = None,
    threshold: float = -0.025,
    n_burn: int = 5000,
    n_draws: int = 25000,
    random_state: int = 42,
    n_workers: int = 4,
    verbose: bool = True,
    cache_path: Optional[Path] = None,
    forecast_start: str = '2008-01-01',
    forecast_end: str = '2010-04-19',
) -> pd.DataFrame:
    """Parallel rolling-window one-step-ahead forecasting via joblib.

    Parameters
    ----------
    y              : Full return series (decimal).
    dates          : Index aligned with *y* (any date-like index).
    n_forecasts    : Maximum number of consecutive forecast origins.
    window_size    : In-sample window length (trading days).
    model_types    : GARCH models; defaults to ['garch', 'gjr', 'egarch'].
    threshold      : CSL censoring threshold (default −0.025).
    n_burn         : MCMC burn-in draws.
    n_draws        : MCMC retained draws.
    random_state   : Base RNG seed.
    n_workers      : Number of parallel processes.
    verbose        : Print progress.
    cache_path     : Optional .pkl path; existing rows are skipped.
    forecast_start : Start of evaluation window (YYYY-MM-DD).
    forecast_end   : End of evaluation window (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame  columns: date, y_realized, model, method, mu, h_forecast,
                           nu, log_score, csl_score
    """
    from joblib import Parallel, delayed

    if model_types is None:
        model_types = ['garch', 'gjr', 'egarch']

    y     = np.asarray(y, dtype=np.float64)
    dates = pd.DatetimeIndex(dates)

    start_dt = pd.Timestamp(forecast_start)
    end_dt   = pd.Timestamp(forecast_end)
    mask     = (dates >= start_dt) & (dates <= end_dt)
    cand_idx = np.where(mask)[0]
    cand_idx = cand_idx[cand_idx >= window_size]

    if len(cand_idx) == 0:
        raise ValueError(
            f"No valid forecast origins in [{forecast_start}, {forecast_end}] "
            f"with window_size={window_size}."
        )

    forecast_indices = cand_idx[:n_forecasts] if len(cand_idx) > n_forecasts else cand_idx
    n_actual = len(forecast_indices)

    # ── Load cache if available ───────────────────────────────────────────────
    cached_rows: List[Dict] = []
    cached_dates: set = set()
    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path, 'rb') as f:
            cached_rows = pickle.load(f)
        cached_dates = {r['date'] for r in cached_rows}
        if verbose:
            print(f"  Cache loaded: {len(cached_rows)} rows from {len(cached_dates)} origins.")

    # ── Filter to missing origins ─────────────────────────────────────────────
    todo_idx = [
        (i, t) for i, t in enumerate(forecast_indices)
        if dates[t] not in cached_dates
    ]
    n_todo = len(todo_idx)

    if verbose:
        print(f"  Forecast origins: {n_actual} total, {n_todo} to compute "
              f"({n_actual - n_todo} cached), workers={n_workers}")

    if n_todo > 0:
        new_results = Parallel(
            n_jobs=n_workers,
            backend='loky',
            verbose=5 if verbose else 0,
        )(
            delayed(_forecast_one_origin)(
                int(t), y, dates[t], float(y[t]),
                window_size, model_types, threshold,
                n_burn, n_draws, random_state, i,
            )
            for i, t in todo_idx
        )

        new_rows: List[Dict] = []
        for chunk in new_results:
            new_rows.extend(chunk)

        all_rows = cached_rows + new_rows

        # Update cache
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(all_rows, f)
            if verbose:
                print(f"  Cache updated → {cache_path}")
    else:
        all_rows = cached_rows

    df = pd.DataFrame(all_rows, columns=[
        'date', 'y_realized', 'model', 'method',
        'mu', 'h_forecast', 'nu', 'log_score', 'csl_score',
    ])
    df = df.sort_values(['date', 'model', 'method']).reset_index(drop=True)

    if verbose:
        n_ok = int(df['log_score'].notna().sum())
        print(f"  Done. {n_ok}/{len(df)} log scores computed.")

    return df
