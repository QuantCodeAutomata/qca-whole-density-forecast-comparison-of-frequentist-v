"""One-step-ahead predictive density evaluation and rolling-window forecasting.

Predictive log scores
---------------------
* **Log score** – log predictive density at the realised return.
* **Censored Shifted Likelihood (CSL)** – focuses on lower-tail fit:
    - y ≤ τ (bad day): uses log predictive density f(y).
    - y > τ (normal day): uses log predictive survival P(Y > τ).

Bayesian mixture
----------------
The Bayesian predictive is a mixture over M posterior draws:
    p(y | data) = (1/M) Σ_m f(y ; θ_m, h_{T+1}^m)
Both scores are computed via log-mean-exp (numerically stable) over draw-level
log-densities/log-survivals.

Rolling-window design
---------------------
Forecast origin t (index in the full return array):
    training data  = y[t - window_size : t]   (window_size ≈ 10 × 252 = 2520)
    realised return = y[t]
    forecast       = h_{T+1} from the training window → predictive density at y[t].
"""

from __future__ import annotations

import sys
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats
from scipy.special import logsumexp

sys.path.insert(0, os.path.dirname(__file__))
import models
from estimation import frequentist_estimate, bayesian_estimate, MODEL_PARAMS


# ---------------------------------------------------------------------------
# Single-estimate predictive density
# ---------------------------------------------------------------------------

def compute_predictive_log_density(
    y_realized: float,
    params: Dict[str, float],
    model_type: str,
    h_forecast: float,
    nu_val: float,
) -> float:
    """Log predictive density at *y_realized* given a single parameter estimate.

    The predictive distribution is:
        Y_{T+1} | θ, h_{T+1}  ~  Student-t(μ, h_{T+1}, ν)

    where the scale satisfies Var(Y_{T+1}) = h_{T+1}.

    Parameters
    ----------
    y_realized : float
        Realised return at the forecast horizon.
    params : dict
        Fitted parameter dict (must contain ``'mu'``).
    model_type : str
        One of ``'garch'``, ``'gjr'``, ``'egarch'``.
    h_forecast : float
        One-step-ahead conditional variance h_{T+1}.
    nu_val : float
        Degrees of freedom ν (may differ from ``params['nu']`` for
        robustness checks, but is typically the same).

    Returns
    -------
    float
        Log predictive density value.
    """
    mu = params['mu']
    return models.student_t_logpdf(float(y_realized), float(mu),
                                   float(h_forecast), float(nu_val))


# ---------------------------------------------------------------------------
# Bayesian predictive scores (mixture over MCMC draws)
# ---------------------------------------------------------------------------

def compute_bayesian_predictive_scores(
    y_realized: float,
    draws: np.ndarray,
    h_forecasts: np.ndarray,
    param_names: List[str],
    model_type: str,
    threshold: Optional[float] = None,
) -> Tuple[float, float]:
    """Average log predictive density and CSL over posterior draws.

    The Bayesian predictive density is the mixture
        p(y | data) = (1/M) Σ_m f(y ; θ_m, h_{T+1}^m)

    Both scores are computed via log-sum-exp for numerical stability.

    Parameters
    ----------
    y_realized : float
        Realised return to evaluate.
    draws : np.ndarray, shape (M, n_params)
        Posterior parameter draws.
    h_forecasts : np.ndarray, shape (M,)
        One-step-ahead variance forecast h_{T+1} for each draw.
    param_names : list of str
        Parameter names matching the column order of *draws*.
    model_type : str
        One of ``'garch'``, ``'gjr'``, ``'egarch'``.
    threshold : float or None
        CSL threshold τ (e.g. -0.025 for −2.5 %).  If None, csl_score is NaN.

    Returns
    -------
    (log_score, csl_score) : (float, float)
        ``log_score`` – log of the mixture predictive density at y_realized.
        ``csl_score``  – log of the mixture CSL score at y_realized.
    """
    mu_idx = param_names.index('mu')
    nu_idx = param_names.index('nu')

    mu_vec = draws[:, mu_idx]    # (M,)
    nu_vec = draws[:, nu_idx]    # (M,)
    h_vec  = h_forecasts         # (M,)

    # Filter out draws with non-finite h_forecast (failed filter evaluations)
    valid = np.isfinite(h_vec) & (h_vec > 0.0) & (nu_vec > 2.0)
    if not np.any(valid):
        return np.nan, np.nan

    mu_v  = mu_vec[valid]
    nu_v  = nu_vec[valid]
    h_v   = h_vec[valid]

    # Vectorised log-density: log f_m(y_realized) for each draw
    s_vec = np.sqrt(h_v * (nu_v - 2.0) / nu_v)
    log_dens = scipy.stats.t.logpdf(
        float(y_realized), df=nu_v, loc=mu_v, scale=s_vec
    )

    # Log score: log(mean_m exp(log_f_m)) = logsumexp(log_f_m) - log(M)
    M = len(log_dens)
    log_score = float(logsumexp(log_dens) - np.log(M))

    # CSL score
    if threshold is None:
        csl_score = np.nan
    else:
        if float(y_realized) <= threshold:
            # Observation in the tail: use density
            csl_vals = log_dens
        else:
            # Observation above threshold: use log survival P_m(Y > τ)
            csl_vals = scipy.stats.t.logsf(
                float(threshold), df=nu_v, loc=mu_v, scale=s_vec
            )
        csl_score = float(logsumexp(csl_vals) - np.log(M))

    return log_score, csl_score


# ---------------------------------------------------------------------------
# Rolling-window forecasting
# ---------------------------------------------------------------------------

def rolling_forecast_window(
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    n_forecasts: int,
    window_size: int = 10 * 252,
    model_types: Optional[List[str]] = None,
    threshold: float = -0.025,
    n_burn: int = 5000,
    n_draws: int = 25000,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Rolling-window one-step-ahead forecasting for all models and methods.

    At each forecast origin t (index in the full return array):
        * training data  = y[t - window_size : t]  (T = window_size observations)
        * realised return = y[t]
        * h_{T+1} and predictive density are computed from the training window.
        * The predictive density is evaluated at y[t].

    Forecast origins are drawn from the date range [2008-01-01, 2010-04-19].
    Up to *n_forecasts* origins are used (all available if fewer exist).

    Parameters
    ----------
    y : np.ndarray
        Full return series in decimal form.
    dates : pd.DatetimeIndex
        Dates corresponding to each element of *y* (same length).
    n_forecasts : int
        Maximum number of forecast origins.
    window_size : int
        Number of in-sample trading days (default 10 × 252 = 2520).
    model_types : list of str or None
        Models to estimate; defaults to ``['garch', 'gjr', 'egarch']``.
    threshold : float
        CSL threshold in decimal returns (default −0.025).
    n_burn : int
        MCMC burn-in draws.
    n_draws : int
        MCMC retained draws.
    random_state : int
        Base RNG seed (incremented per forecast origin × model to ensure
        reproducibility across partial runs).
    verbose : bool
        Print progress every 10 forecast origins if True.

    Returns
    -------
    pd.DataFrame with columns:
        ``date``, ``y_realized``, ``model``, ``method``,
        ``mu``, ``h_forecast``, ``nu``, ``log_score``, ``csl_score``.
    """
    if model_types is None:
        model_types = ['garch', 'gjr', 'egarch']

    y = np.asarray(y, dtype=np.float64)
    dates = pd.DatetimeIndex(dates)

    # Identify forecast-origin indices: dates in [2008-01-01, 2010-04-19]
    start_date = pd.Timestamp('2008-01-01')
    end_date   = pd.Timestamp('2010-04-19')
    mask = (dates >= start_date) & (dates <= end_date)
    candidate_indices = np.where(mask)[0]

    # Drop origins without enough history
    candidate_indices = candidate_indices[candidate_indices >= window_size]

    if len(candidate_indices) == 0:
        raise ValueError(
            "No valid forecast origins found in [2008-01-01, 2010-04-19] "
            f"with window_size={window_size}.  Check that the date range and "
            "return series are aligned."
        )

    # Limit to n_forecasts (take first n_forecasts in chronological order)
    if len(candidate_indices) > n_forecasts:
        forecast_indices = candidate_indices[:n_forecasts]
    else:
        forecast_indices = candidate_indices

    n_actual = len(forecast_indices)
    rows: List[Dict] = []

    for i, t in enumerate(forecast_indices):
        training   = y[t - window_size : t]
        y_realized = float(y[t])
        date_t     = dates[t]

        if verbose and (i % 10 == 0):
            print(f"[rolling] origin {i+1}/{n_actual}  date={date_t.date()}  "
                  f"y_realized={y_realized:.4f}")

        for model_type in model_types:
            # ---------------------------------------------------------- #
            # Frequentist MLE                                             #
            # ---------------------------------------------------------- #
            try:
                mle = frequentist_estimate(training, model_type)
                mu_mle  = mle['params']['mu']
                nu_mle  = mle['params']['nu']
                h_fc_mle = mle['h_forecast']
                ls_mle = compute_predictive_log_density(
                    y_realized, mle['params'], model_type, h_fc_mle, nu_mle
                )
                if y_realized <= threshold:
                    csl_mle = ls_mle
                else:
                    csl_mle = models.student_t_logsf(
                        threshold, mu_mle, h_fc_mle, nu_mle
                    )
            except Exception as exc:
                warnings.warn(
                    f"MLE failed for model={model_type} at t={t}: {exc}"
                )
                mu_mle = nu_mle = h_fc_mle = ls_mle = csl_mle = np.nan

            rows.append({
                'date':       date_t,
                'y_realized': y_realized,
                'model':      model_type,
                'method':     'MLE',
                'mu':         mu_mle,
                'h_forecast': h_fc_mle,
                'nu':         nu_mle,
                'log_score':  ls_mle,
                'csl_score':  csl_mle,
            })

            # ---------------------------------------------------------- #
            # Bayesian MCMC                                               #
            # ---------------------------------------------------------- #
            # Vary the seed per origin and model so reruns are reproducible
            model_seed = random_state + i * len(model_types) + model_types.index(model_type)
            try:
                bay = bayesian_estimate(
                    training,
                    model_type,
                    n_burn=n_burn,
                    n_draws=n_draws,
                    random_state=model_seed,
                )
                # Representative point estimates from posterior mean
                pnames   = bay['param_names']
                mu_idx   = pnames.index('mu')
                nu_idx   = pnames.index('nu')
                mu_bay   = float(np.mean(bay['draws'][:, mu_idx]))
                nu_bay   = float(np.mean(bay['draws'][:, nu_idx]))
                h_fc_bay = float(np.nanmean(bay['h_forecasts']))

                ls_bay, csl_bay = compute_bayesian_predictive_scores(
                    y_realized,
                    bay['draws'],
                    bay['h_forecasts'],
                    pnames,
                    model_type,
                    threshold=threshold,
                )
            except Exception as exc:
                warnings.warn(
                    f"MCMC failed for model={model_type} at t={t}: {exc}"
                )
                mu_bay = nu_bay = h_fc_bay = ls_bay = csl_bay = np.nan

            rows.append({
                'date':       date_t,
                'y_realized': y_realized,
                'model':      model_type,
                'method':     'MCMC',
                'mu':         mu_bay,
                'h_forecast': h_fc_bay,
                'nu':         nu_bay,
                'log_score':  ls_bay,
                'csl_score':  csl_bay,
            })

    df = pd.DataFrame(rows, columns=[
        'date', 'y_realized', 'model', 'method',
        'mu', 'h_forecast', 'nu', 'log_score', 'csl_score',
    ])
    return df
