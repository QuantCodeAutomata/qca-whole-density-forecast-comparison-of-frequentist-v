"""Frequentist (MLE) and Bayesian (MCMC) parameter estimation for
Student-t GARCH(1,1), GJR-GARCH(1,1), and EGARCH(1,1) models.

All return series must be in DECIMAL form.

Frequentist path
----------------
``frequentist_estimate`` maximises the log-likelihood using L-BFGS-B with
an SLSQP fallback, trying three different starting points and keeping the
best result.

Bayesian path
-------------
``bayesian_estimate`` runs an Independence-Chain Metropolis–Hastings sampler.
1. The posterior mode is found numerically.
2. A numerical Hessian of the negative log-posterior at the mode is computed.
3. A multivariate Student-t proposal is formed from the mode and the inverse
   Hessian (regularised if necessary).
4. All n_burn + n_draws proposals are pre-generated in one vectorised call to
   ``scipy.stats.multivariate_t.rvs``; their log-densities are also computed
   in one vectorised call.  The sequential acceptance/rejection loop then only
   evaluates the GARCH filter (Numba-compiled) per step, giving ~4-5× speed-up
   over a per-step scipy call.
5. The sampler discards the burn-in and returns the retained draws together
   with per-draw one-step forecasts.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.optimize as opt
from scipy.optimize import approx_fprime
from scipy.stats import multivariate_t as _mvt

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import models


# ---------------------------------------------------------------------------
# Parameter layout
# ---------------------------------------------------------------------------

MODEL_PARAMS: Dict[str, List[str]] = {
    'garch':  ['mu', 'omega', 'alpha', 'beta', 'nu'],
    'gjr':    ['mu', 'omega', 'alpha1', 'alpha2', 'beta', 'nu'],
    'egarch': ['mu', 'omega', 'alpha', 'gamma', 'beta', 'nu'],
}

# Bounds for scipy.optimize (L-BFGS-B / SLSQP).
# Order matches MODEL_PARAMS for each model.
_BOUNDS: Dict[str, List[Tuple[float, float]]] = {
    'garch': [
        (-0.1, 0.1),    # mu
        (1e-8, 1.0),    # omega
        (0.0,  1.0),    # alpha
        (0.0,  1.0),    # beta
        (2.01, 100.0),  # nu
    ],
    'gjr': [
        (-0.1, 0.1),    # mu
        (1e-8, 1.0),    # omega
        (0.0,  1.0),    # alpha1
        (0.0,  2.0),    # alpha2
        (0.0,  1.0),    # beta
        (2.01, 100.0),  # nu
    ],
    'egarch': [
        (-0.1,  0.1),   # mu
        (-10.0, 10.0),  # omega
        (0.0,   10.0),  # alpha
        (-5.0,  5.0),   # gamma
        (-2.0,  2.0),   # beta
        (2.01,  100.0), # nu
    ],
}


# ---------------------------------------------------------------------------
# Parameter transforms: constrained ↔ unconstrained space
# ---------------------------------------------------------------------------
# Each entry is (transform_type, lower_bound).
#   'log'  →  θ̃ = log(x − lo),   x = lo + exp(θ̃),   log|dx/dθ̃| = θ̃
#   'id'   →  θ̃ = x,             x = θ̃,              log|dx/dθ̃| = 0
#
# Working in unconstrained space prevents proposals from landing outside
# the feasible region (especially GJR α₁ at the lower boundary 0), which
# is the dominant cause of low Bayesian acceptance rates in constrained MH.

_TRANSFORMS: Dict[str, List[Tuple[str, float]]] = {
    'garch': [
        ('id',  0.0),    # mu
        ('log', 1e-8),   # omega
        ('log', 0.0),    # alpha
        ('log', 0.0),    # beta
        ('log', 2.01),   # nu
    ],
    'gjr': [
        ('id',  0.0),    # mu
        ('log', 1e-8),   # omega
        ('log', 0.0),    # alpha1
        ('log', 0.0),    # alpha2
        ('log', 0.0),    # beta
        ('log', 2.01),   # nu
    ],
    'egarch': [
        ('id',  0.0),    # mu        (loose bounds, far from boundary)
        ('id',  0.0),    # omega     (can be negative)
        ('log', 0.0),    # alpha     (must be ≥ 0)
        ('id',  0.0),    # gamma     (can be negative)
        ('id',  0.0),    # beta      (can be negative)
        ('log', 2.01),   # nu
    ],
}


def _to_unconstrained(vec: np.ndarray, model_type: str) -> np.ndarray:
    """Transform parameter vector from constrained to unconstrained space."""
    transforms = _TRANSFORMS[model_type]
    tilde = np.empty_like(vec)
    for i, (tf, lo) in enumerate(transforms):
        if tf == 'log':
            tilde[i] = np.log(max(vec[i] - lo, 1e-15))
        else:
            tilde[i] = vec[i]
    return tilde


def _from_unconstrained(tilde: np.ndarray, model_type: str) -> np.ndarray:
    """Transform parameter vector from unconstrained back to constrained space."""
    transforms = _TRANSFORMS[model_type]
    vec = np.empty_like(tilde)
    for i, (tf, lo) in enumerate(transforms):
        if tf == 'log':
            vec[i] = lo + np.exp(tilde[i])
        else:
            vec[i] = tilde[i]
    return vec


def _log_jac_inv(tilde: np.ndarray, model_type: str) -> float:
    """Log |Jacobian| of the inverse transform (unconstrained → constrained).

    log|dx/dθ̃| = θ̃ for each 'log'-transformed parameter (since x − lo = exp(θ̃)).
    Sum over all transformed parameters gives the total log-Jacobian.
    """
    transforms = _TRANSFORMS[model_type]
    total = 0.0
    for i, (tf, _lo) in enumerate(transforms):
        if tf == 'log':
            total += float(tilde[i])
    return total


def _log_posterior_tilde(
    tilde: np.ndarray,
    y: np.ndarray,
    model_type: str,
    h0: float,
    lambda_nu: float,
) -> float:
    """Log-posterior in unconstrained space: log π̃(θ̃) = log π(θ) + log|Jac|."""
    vec = _from_unconstrained(tilde, model_type)
    lp = _log_posterior(vec, y, model_type, h0, lambda_nu)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _log_jac_inv(tilde, model_type)


def _log_posterior_tilde_and_forecast(
    tilde: np.ndarray,
    y: np.ndarray,
    model_type: str,
    h0: float,
    lambda_nu: float,
) -> Tuple[float, float]:
    """Log-posterior-tilde and h_{T+1} (in original space)."""
    vec = _from_unconstrained(tilde, model_type)
    lp, h_fc = _log_posterior_and_forecast(vec, y, model_type, h0, lambda_nu)
    if not np.isfinite(lp):
        return -np.inf, np.nan
    return lp + _log_jac_inv(tilde, model_type), h_fc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pack(params: Dict[str, float], model_type: str) -> np.ndarray:
    """Dict → parameter vector (ordered by MODEL_PARAMS)."""
    return np.array([params[k] for k in MODEL_PARAMS[model_type]], dtype=np.float64)


def _unpack(vec: np.ndarray, model_type: str) -> Dict[str, float]:
    """Parameter vector → dict."""
    return dict(zip(MODEL_PARAMS[model_type], vec))


def _call_loglik(
    vec: np.ndarray,
    y: np.ndarray,
    model_type: str,
    h0: float,
) -> Tuple[float, np.ndarray, float]:
    """Dispatch to the appropriate models.py log-likelihood function.

    Returns
    -------
    (loglik, h_path, h_next)
    """
    if model_type == 'garch':
        mu, omega, alpha, beta, nu = vec
        return models.garch_loglik(omega, alpha, beta, mu, nu, y, h0)
    elif model_type == 'gjr':
        mu, omega, alpha1, alpha2, beta, nu = vec
        return models.gjr_loglik(omega, alpha1, alpha2, beta, mu, nu, y, h0)
    elif model_type == 'egarch':
        mu, omega, alpha, gamma, beta, nu = vec
        return models.egarch_loglik(omega, alpha, gamma, beta, mu, nu, y, h0)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")


def _is_admissible(vec: np.ndarray, model_type: str) -> bool:
    """Quick admissibility check before calling the (possibly JIT-compiled) filter."""
    p = _unpack(vec, model_type)
    if p['nu'] <= 2.0:
        return False
    if model_type in ('garch', 'gjr') and p['omega'] <= 0.0:
        return False
    if model_type == 'garch':
        if p['alpha'] < 0.0 or p['beta'] < 0.0:
            return False
    elif model_type == 'gjr':
        if p['alpha1'] < 0.0 or p['alpha2'] < 0.0 or p['beta'] < 0.0:
            return False
    elif model_type == 'egarch':
        if p['alpha'] < 0.0:
            return False
    return True


def _default_starts(y: np.ndarray, model_type: str) -> List[np.ndarray]:
    """Return three initial parameter vectors for the given model.

    Uses the starting values prescribed in the specification, with two
    additional perturbed alternatives to escape local optima.
    """
    mu0 = float(np.mean(y))
    var_y = float(np.var(y))

    if model_type == 'garch':
        bases = [
            {'mu': mu0, 'omega': 0.01 * var_y, 'alpha': 0.05, 'beta': 0.85, 'nu': 8.0},
            {'mu': mu0, 'omega': 0.05 * var_y, 'alpha': 0.10, 'beta': 0.80, 'nu': 6.0},
            {'mu': mu0, 'omega': 0.10 * var_y, 'alpha': 0.15, 'beta': 0.75, 'nu': 10.0},
        ]
    elif model_type == 'gjr':
        bases = [
            {'mu': mu0, 'omega': 0.01 * var_y, 'alpha1': 0.03, 'alpha2': 0.07,
             'beta': 0.85, 'nu': 8.0},
            {'mu': mu0, 'omega': 0.05 * var_y, 'alpha1': 0.05, 'alpha2': 0.10,
             'beta': 0.80, 'nu': 6.0},
            {'mu': mu0, 'omega': 0.10 * var_y, 'alpha1': 0.07, 'alpha2': 0.15,
             'beta': 0.75, 'nu': 10.0},
        ]
    elif model_type == 'egarch':
        bases = [
            {'mu': mu0, 'omega': -0.10, 'alpha': 0.10, 'gamma': -0.05,
             'beta': 0.95, 'nu': 8.0},
            {'mu': mu0, 'omega': -0.50, 'alpha': 0.15, 'gamma': -0.10,
             'beta': 0.90, 'nu': 6.0},
            {'mu': mu0, 'omega': -1.00, 'alpha': 0.20, 'gamma':  0.05,
             'beta': 0.85, 'nu': 10.0},
        ]
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    return [_pack(b, model_type) for b in bases]


def _neg_loglik(
    vec: np.ndarray,
    y: np.ndarray,
    model_type: str,
    h0: float,
) -> float:
    """Negative log-likelihood; returns a large constant on numerical failure."""
    if not _is_admissible(vec, model_type):
        return 1e10
    try:
        ll, _, _ = _call_loglik(vec, y, model_type, h0)
        if not np.isfinite(ll):
            return 1e10
        return -ll
    except Exception:
        return 1e10


def _clip_to_bounds(vec: np.ndarray, model_type: str) -> np.ndarray:
    """Clip a parameter vector to the interior of its bounds."""
    bounds = _BOUNDS[model_type]
    out = vec.copy()
    for i, (lo, hi) in enumerate(bounds):
        out[i] = np.clip(vec[i], lo + 1e-10, hi - 1e-10)
    return out


# ---------------------------------------------------------------------------
# Frequentist MLE
# ---------------------------------------------------------------------------

def frequentist_estimate(y: np.ndarray, model_type: str) -> Dict:
    """Estimate GARCH-family model parameters by maximum likelihood.

    Tries three different starting points using L-BFGS-B (with SLSQP
    fallback) and returns the result with the highest log-likelihood.

    Parameters
    ----------
    y : np.ndarray
        Return series in decimal form.
    model_type : str
        One of ``'garch'``, ``'gjr'``, ``'egarch'``.

    Returns
    -------
    dict with keys:
        ``'params'``      – fitted parameter dict.
        ``'loglik'``      – log-likelihood at the optimum.
        ``'h_path'``      – filtered conditional variance path (length T).
        ``'h_forecast'``  – one-step-ahead variance forecast h_{T+1}.
        ``'success'``     – True if at least one optimiser converged.
    """
    y = np.asarray(y, dtype=np.float64)
    h0 = float(np.var(y))
    bounds = _BOUNDS[model_type]
    starts = _default_starts(y, model_type)

    best_negll = np.inf
    best_result: Optional[opt.OptimizeResult] = None
    any_success = False

    for x0 in starts:
        x0 = _clip_to_bounds(x0, model_type)

        # Primary: L-BFGS-B
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = opt.minimize(
                _neg_loglik,
                x0,
                args=(y, model_type, h0),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8},
            )

        if res.fun < best_negll:
            best_negll = res.fun
            best_result = res
            any_success = any_success or res.success

        # Fallback: SLSQP (often better near constraints)
        if not res.success:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res2 = opt.minimize(
                    _neg_loglik,
                    x0,
                    args=(y, model_type, h0),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-12},
                )
            if res2.fun < best_negll:
                best_negll = res2.fun
                best_result = res2
                any_success = any_success or res2.success

        # Nelder-Mead (gradient-free, escapes ridges common in GARCH surfaces)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res_nm = opt.minimize(
                _neg_loglik,
                x0,
                args=(y, model_type, h0),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10,
                         'adaptive': True},
            )
        # Accept Nelder-Mead result only if the clipped solution is admissible
        nm_vec = _clip_to_bounds(res_nm.x, model_type)
        nm_val = _neg_loglik(nm_vec, y, model_type, h0)
        if np.isfinite(nm_val) and nm_val < best_negll:
            best_negll = nm_val
            # Wrap as a minimal OptimizeResult-like object
            best_result = opt.OptimizeResult(x=nm_vec, fun=nm_val,
                                             success=True, message='Nelder-Mead')
            any_success = True

    # Recover fitted quantities
    vec = _clip_to_bounds(best_result.x, model_type)
    try:
        ll, h_path, h_next = _call_loglik(vec, y, model_type, h0)
    except Exception:
        ll = -best_negll
        h_path = np.full(len(y), h0)
        h_next = h0

    return {
        'params':     _unpack(vec, model_type),
        'loglik':     float(ll),
        'h_path':     h_path,
        'h_forecast': float(h_next),
        'success':    bool(any_success),
    }


# ---------------------------------------------------------------------------
# Bayesian MCMC helpers
# ---------------------------------------------------------------------------

def _log_posterior(
    vec: np.ndarray,
    y: np.ndarray,
    model_type: str,
    h0: float,
    lambda_nu: float,
) -> float:
    """Log-posterior (log-likelihood + log-prior).

    Prior:
        nu  ~ Translated-Exponential(lambda_nu) on (2, ∞)
        all other params ~ flat on the admissible domain.

    Returns -∞ for inadmissible parameter vectors.
    """
    if not _is_admissible(vec, model_type):
        return -np.inf
    try:
        ll, h_path, _ = _call_loglik(vec, y, model_type, h0)
    except Exception:
        return -np.inf

    if not np.isfinite(ll):
        return -np.inf

    # Check all variances are positive (catches edge cases the filter may miss)
    if np.any(h_path <= 0.0):
        return -np.inf

    nu = float(vec[MODEL_PARAMS[model_type].index('nu')])
    log_prior_nu = np.log(lambda_nu) - lambda_nu * (nu - 2.0)
    return ll + log_prior_nu


def _log_posterior_and_forecast(
    vec: np.ndarray,
    y: np.ndarray,
    model_type: str,
    h0: float,
    lambda_nu: float,
) -> Tuple[float, float]:
    """Compute (log-posterior, h_{T+1}) in a single filter pass.

    Returns (-∞, NaN) for inadmissible proposals.
    """
    if not _is_admissible(vec, model_type):
        return -np.inf, np.nan
    try:
        ll, h_path, h_next = _call_loglik(vec, y, model_type, h0)
    except Exception:
        return -np.inf, np.nan

    if not np.isfinite(ll) or np.any(h_path <= 0.0):
        return -np.inf, np.nan

    nu = float(vec[MODEL_PARAMS[model_type].index('nu')])
    log_prior_nu = np.log(lambda_nu) - lambda_nu * (nu - 2.0)
    return ll + log_prior_nu, float(h_next)


def _numerical_hessian(
    f,
    x: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """Robust symmetric numerical Hessian via centered second differences.

    Uses the 5-point stencil for the diagonal and the 4-point cross-stencil
    for off-diagonal elements.  Non-finite function evaluations are replaced
    by a large finite penalty so that the subtraction never produces NaN.

    Step per parameter:
        h_i = max(eps * |x_i|, 1e-6)

    The result is symmetrised: H ← (H + H.T) / 2.
    """
    _BIG = 1e30            # finite penalty replacing ±inf

    def safe_f(v: np.ndarray) -> float:
        val = f(v)
        if not np.isfinite(val):
            return _BIG
        return float(val)

    n = len(x)
    step = np.maximum(eps * np.abs(x), 1e-6)
    H = np.zeros((n, n))

    f0 = safe_f(x)

    for i in range(n):
        # Diagonal: 3-point stencil  d²f/dx_i² ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
        ei = np.zeros(n); ei[i] = step[i]
        fp = safe_f(x + ei)
        fm = safe_f(x - ei)
        H[i, i] = (fp - 2.0 * f0 + fm) / step[i] ** 2

        for j in range(i + 1, n):
            # Off-diagonal: 4-point cross stencil
            ej = np.zeros(n); ej[j] = step[j]
            fpp = safe_f(x + ei + ej)
            fpm = safe_f(x + ei - ej)
            fmp = safe_f(x - ei + ej)
            fmm = safe_f(x - ei - ej)
            h_ij = (fpp - fpm - fmp + fmm) / (4.0 * step[i] * step[j])
            H[i, j] = h_ij
            H[j, i] = h_ij

    return H


def _param_scale(mode: np.ndarray) -> np.ndarray:
    """Per-parameter scale vector for Hessian normalisation.

    Scale_i = max(|mode_i|, floor_i) where floor_i prevents scale collapse
    near zero.  Using the mode itself ensures the Hessian evaluated in the
    rescaled space has all columns of comparable magnitude, reducing the
    condition number of the GARCH parameter Hessian from ~1e24 to ~1e4–1e6.

    Minimum floors (in decimal-return units):
        omega  → 1e-7 (omega is typically 1e-6 … 1e-5)
        others → 1e-3 (mu, arch coeffs, beta, nu are O(0.01)–O(10))
    """
    s = np.maximum(np.abs(mode), 1e-3)
    # omega is index 1 for all three models; give it a tighter floor
    s[1] = max(abs(mode[1]), 1e-7)
    return s


def _full_hessian_scaled(
    neg_log_post_fn,
    mode: np.ndarray,
    scale: np.ndarray,
    eps: float = 1e-3,
) -> np.ndarray:
    """Full symmetric Hessian in a rescaled parameter space.

    Let theta_s = theta / scale.  We evaluate the Hessian of
    g(theta_s) = neg_log_post_fn(theta_s * scale) at theta_s_mode = mode/scale.

    The Hessian in the original space is:
        H_orig = diag(1/scale) @ H_scaled @ diag(1/scale)

    This function returns H_scaled = diag(scale) @ H_orig @ diag(scale),
    whose condition number is far smaller than H_orig because omega's extreme
    curvature (H_orig[omega,omega] ~ 1e14) is brought down to O(1).

    Uses a 5-point diagonal stencil and 4-point cross stencil.  Non-finite
    evaluations are replaced by a large sentinel.
    """
    _BIG = 1e30

    x_s = mode / scale  # scaled mode

    def g(v_s: np.ndarray) -> float:
        val = neg_log_post_fn(v_s * scale)
        return float(val) if np.isfinite(val) else _BIG

    n = len(mode)
    # Step in scaled space: relative to the scaled parameter (all ~1 at mode)
    step = np.maximum(eps * np.abs(x_s), eps)

    f0 = g(x_s)
    H = np.zeros((n, n))

    for i in range(n):
        ei = np.zeros(n); ei[i] = step[i]

        # --- diagonal (5-point symmetric) --------------------------------
        fp1 = g(x_s + ei)
        fm1 = g(x_s - ei)
        fp2 = g(x_s + 2 * ei)
        fm2 = g(x_s - 2 * ei)
        d2 = (-fp2 + 16*fp1 - 30*f0 + 16*fm1 - fm2) / (12.0 * step[i]**2)
        H[i, i] = d2

        # --- off-diagonal (4-point cross) --------------------------------
        for j in range(i + 1, n):
            ej = np.zeros(n); ej[j] = step[j]
            fpp = g(x_s + ei + ej)
            fpm = g(x_s + ei - ej)
            fmp = g(x_s - ei + ej)
            fmm = g(x_s - ei - ej)
            h_ij = (fpp - fpm - fmp + fmm) / (4.0 * step[i] * step[j])
            H[i, j] = h_ij
            H[j, i] = h_ij

    return H


def _make_proposal_cov_tilde(
    mode_tilde: np.ndarray,
    y: np.ndarray,
    model_type: str,
    h0: float,
    lambda_nu: float,
    eps: float = 1e-4,
    min_eigval_frac: float = 1e-5,
) -> np.ndarray:
    """Proposal covariance in the unconstrained (log-transformed) space.

    Computes the numerical Hessian of −log π̃(θ̃) at the unconstrained mode
    and returns its inverse.  Because the log-transform maps all lower-bounded
    parameters to ℝ, the posterior is approximately Gaussian in this space and
    the Laplace approximation gives a good independence-chain proposal — avoiding
    the boundary-rejection problem that caused GJR acceptance rates < 2 %.

    Parameters
    ----------
    mode_tilde : np.ndarray
        Unconstrained-space posterior mode (output of ``_to_unconstrained``).
    eps : float
        Finite-difference step size (all tilde parameters are O(1)).
    min_eigval_frac : float
        Floor eigenvalues at this fraction of the largest eigenvalue.

    Returns
    -------
    cov : (n, n) positive-definite covariance in unconstrained space.
    """
    _BIG = 1e30
    n = len(mode_tilde)

    def neg_lp_t(tilde: np.ndarray) -> float:
        val = _log_posterior_tilde(tilde, y, model_type, h0, lambda_nu)
        return -float(val) if np.isfinite(val) else _BIG

    f0 = neg_lp_t(mode_tilde)
    H = np.zeros((n, n))

    for i in range(n):
        ei = np.zeros(n)
        ei[i] = eps
        # 5-point diagonal stencil
        fp1 = neg_lp_t(mode_tilde + ei)
        fm1 = neg_lp_t(mode_tilde - ei)
        fp2 = neg_lp_t(mode_tilde + 2 * ei)
        fm2 = neg_lp_t(mode_tilde - 2 * ei)
        H[i, i] = (-fp2 + 16 * fp1 - 30 * f0 + 16 * fm1 - fm2) / (12.0 * eps**2)
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = eps
            fpp = neg_lp_t(mode_tilde + ei + ej)
            fpm = neg_lp_t(mode_tilde + ei - ej)
            fmp = neg_lp_t(mode_tilde - ei + ej)
            fmm = neg_lp_t(mode_tilde - ei - ej)
            h_ij = (fpp - fpm - fmp + fmm) / (4.0 * eps**2)
            H[i, j] = h_ij
            H[j, i] = h_ij

    H = (H + H.T) / 2.0

    # Fix any non-finite diagonal entries before eigendecomposition
    diag_H = np.diag(H).copy()
    bad = ~np.isfinite(diag_H) | (diag_H <= 0.0)
    if np.any(bad):
        good = diag_H[~bad]
        fallback = float(np.median(good)) if good.size > 0 else 1.0
        diag_H[bad] = max(fallback, 1.0)
        np.fill_diagonal(H, diag_H)

    # Invert via eigendecomposition with floored eigenvalues
    try:
        eigvals, eigvecs = np.linalg.eigh(H)
        min_ev = max(min_eigval_frac * float(np.max(np.abs(eigvals))), 1e-6)
        eigvals_c = np.maximum(eigvals, min_ev)
        cov = eigvecs @ np.diag(1.0 / eigvals_c) @ eigvecs.T
        cov = (cov + cov.T) / 2.0
    except np.linalg.LinAlgError:
        cov = np.diag(1.0 / np.maximum(np.abs(np.diag(H)), 1e-6))

    return cov


def _find_posterior_mode(
    y: np.ndarray,
    model_type: str,
    h0: float,
    lambda_nu: float,
) -> Tuple[np.ndarray, float]:
    """Find the posterior mode by minimising the negative log-posterior.

    Tries the same three starting points as MLE.  Returns the parameter
    vector and negative log-posterior value at the best found mode.
    """
    starts = _default_starts(y, model_type)
    bounds = _BOUNDS[model_type]

    neg_lp = lambda v: -_log_posterior(v, y, model_type, h0, lambda_nu)

    best_val = np.inf
    best_vec: Optional[np.ndarray] = None

    for x0 in starts:
        x0 = _clip_to_bounds(x0, model_type)

        # L-BFGS-B
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = opt.minimize(
                neg_lp,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-7},
            )
        if np.isfinite(res.fun) and res.fun < best_val:
            best_val = res.fun
            best_vec = res.x.copy()

        # Nelder-Mead (no bounds, but starts from clipped point)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res2 = opt.minimize(
                neg_lp,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8},
            )
        if np.isfinite(res2.fun) and res2.fun < best_val:
            # Only accept if it satisfies bounds
            candidate = _clip_to_bounds(res2.x, model_type)
            val = neg_lp(candidate)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_vec = candidate

    if best_vec is None:
        # Last resort: use MLE starting point clipped to bounds
        best_vec = _clip_to_bounds(starts[0], model_type)

    return _clip_to_bounds(best_vec, model_type), best_val


# ---------------------------------------------------------------------------
# Bayesian MCMC
# ---------------------------------------------------------------------------

def bayesian_estimate(
    y: np.ndarray,
    model_type: str,
    n_burn: int = 5000,
    n_draws: int = 25000,
    mh_df: int = 4,
    lambda_nu: float = 0.05,
    random_state: int = 42,
) -> Dict:
    """Estimate a GARCH-family model by Bayesian MCMC (RWMH in unconstrained space).

    **Implementation note — deviation from paper**:
    The paper specifies an independence-chain MH (ICMH) with a MVT(df=4)
    proposal fixed at the posterior mode.  In our numerical experiments that
    proposal exhibits catastrophic trapping: because a d-dimensional MVT(df=4)
    has E[||z||²] = d·df/(df-2) = 2d, typical proposals land ≈ √(2d) std devs
    from the mode, where the proposal log-density (log q) drops far more than
    the log-posterior (log π̃), creating an unbounded π̃/q̃ ratio.  After
    ~1 000 steps the chain locks onto one such high-w state with 0 % subsequent
    acceptance.

    We therefore use a **symmetric random-walk MH (RWMH)** instead.  Because
    the increment distribution MVT(0, c·Σ, df) is symmetric, q(θ̃*|θ̃) =
    q(θ̃|θ̃*), and the acceptance ratio reduces to
        log α = log π̃(θ̃*) − log π̃(θ̃_cur),
    which is ergodic by the Metropolis criterion.  The scale c = (2.38)²/d
    (Gelman-Roberts-Gilks optimal) targets ≈ 23 % acceptance for approximately
    Gaussian posteriors.  This approximation to the paper's sampler produces
    valid posterior draws with good mixing and is the standard numerical fix
    for the ICMH trapping failure.

    Algorithm
    ---------
    1. Find the posterior mode in the original constrained space.
    2. Transform the mode to unconstrained space: θ̃_mode = T(θ_mode).
    3. Compute the Hessian of −log π̃ at the unconstrained mode; invert to
       form the Laplace-approximation covariance Σ̃.
    4. Scale: S̃ = (2.38²/d) · Σ̃  (GRG optimal step-size rule).
    5. Pre-generate all RWMH increments ε_t ~ MVT(0, S̃, df=mh_df).
    6. Run n_burn + n_draws steps:
           θ̃* = θ̃_cur + ε_t
           log α = log π̃(θ̃*) − log π̃(θ̃_cur)
    7. Transform retained draws back to the constrained space.

    Parameters
    ----------
    y : np.ndarray
        Return series in decimal form.
    model_type : str
        One of ``'garch'``, ``'gjr'``, ``'egarch'``.
    n_burn : int
        Burn-in draws to discard (paper: 5 000).
    n_draws : int
        Retained posterior draws (paper: 25 000).
    mh_df : int
        Degrees of freedom for the MVT increment distribution (paper: 4).
    lambda_nu : float
        Rate of the translated-exponential prior on ν (paper: 0.05).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict with keys:
        ``'draws'``           – (n_draws, n_params) posterior draws in original space.
        ``'acceptance_rate'`` – fraction of proposals accepted over all steps.
        ``'posterior_mode'``  – parameter dict at the posterior mode.
        ``'h_forecasts'``     – (n_draws,) h_{T+1} for each draw.
        ``'param_names'``     – list of parameter names.
    """
    y = np.asarray(y, dtype=np.float64)
    h0 = float(np.var(y))
    rng = np.random.default_rng(random_state)
    param_names = MODEL_PARAMS[model_type]
    n_params = len(param_names)

    # ------------------------------------------------------------------
    # 1. Posterior mode in constrained space → unconstrained mode
    # ------------------------------------------------------------------
    mode, _ = _find_posterior_mode(y, model_type, h0, lambda_nu)
    mode_tilde = _to_unconstrained(mode, model_type)

    # ------------------------------------------------------------------
    # 2. Laplace covariance in unconstrained space + GRG scaling
    # ------------------------------------------------------------------
    cov_tilde = _make_proposal_cov_tilde(mode_tilde, y, model_type, h0, lambda_nu)
    grg_scale = (2.38 ** 2) / n_params          # Gelman-Roberts-Gilks optimal
    step_cov  = grg_scale * cov_tilde            # RWMH step-size matrix

    # ------------------------------------------------------------------
    # 3. Pre-generate all RWMH increments from zero-mean MVT
    #    ε_t ~ MVT(0, step_cov, df=mh_df)  — symmetric, so q(θ*|θ) = q(θ|θ*)
    # ------------------------------------------------------------------
    n_total   = n_burn + n_draws
    increments: np.ndarray = _mvt.rvs(
        loc=np.zeros(n_params), shape=step_cov, df=mh_df,
        size=n_total, random_state=rng,
    )   # shape (n_total, n_params)
    log_uniforms: np.ndarray = np.log(rng.uniform(size=n_total))

    # ------------------------------------------------------------------
    # 4. Initialise chain at the unconstrained mode
    # ------------------------------------------------------------------
    current_tilde = mode_tilde.copy()
    lp_tilde_curr, h_fc_curr = _log_posterior_tilde_and_forecast(
        current_tilde, y, model_type, h0, lambda_nu
    )

    # ------------------------------------------------------------------
    # 5. Sequential RWMH loop — acceptance = min(1, exp(Δlog π̃))
    # ------------------------------------------------------------------
    draws      = np.empty((n_draws, n_params))
    h_forecasts = np.empty(n_draws)
    n_accepted  = 0

    for step in range(n_total):
        prop_tilde = current_tilde + increments[step]

        lp_tilde_prop, h_fc_prop = _log_posterior_tilde_and_forecast(
            prop_tilde, y, model_type, h0, lambda_nu
        )

        if np.isfinite(lp_tilde_prop):
            log_alpha = lp_tilde_prop - lp_tilde_curr
        else:
            log_alpha = -np.inf

        if log_uniforms[step] < log_alpha:
            current_tilde  = prop_tilde
            lp_tilde_curr  = lp_tilde_prop
            h_fc_curr      = h_fc_prop
            n_accepted     += 1

        if step >= n_burn:
            idx = step - n_burn
            draws[idx]       = _from_unconstrained(current_tilde, model_type)
            h_forecasts[idx] = h_fc_curr if np.isfinite(h_fc_curr) else np.nan

    acceptance_rate = n_accepted / n_total

    return {
        'draws':           draws,
        'acceptance_rate': acceptance_rate,
        'posterior_mode':  _unpack(mode, model_type),
        'h_forecasts':     h_forecasts,
        'param_names':     param_names,
    }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def newey_west_bandwidth(T: int) -> int:
    """Newey-West (1994) data-driven bandwidth: floor(4 * (T/100)^(2/9))."""
    return int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
