"""GARCH-family conditional variance models with Student-t innovations.

All returns and residuals are in DECIMAL units (not percent).
Conditional variances h_t are therefore in decimal^2.

Models implemented
------------------
GARCH(1,1) : h_t = omega + alpha*eps_{t-1}^2 + beta*h_{t-1}
GJR(1,1)   : h_t = omega + alpha1*eps_{t-1}^2*I{eps>=0}
                          + alpha2*eps_{t-1}^2*I{eps<0}  + beta*h_{t-1}
EGARCH(1,1): log h_t = omega + alpha*|z_{t-1}| + gamma*z_{t-1} + beta*log h_{t-1}
             where z_t = eps_t / sqrt(h_t)  (no centering term E|z_t| added)

Innovation distribution: Student-t(0, scale, nu) where scale^2 = h_t*(nu-2)/nu
so that Var(eps_t) = h_t.
"""

import math
from typing import Optional, Tuple

import numpy as np
import scipy.stats

try:
    import numba
    HAS_NUMBA = True
    njit = numba.njit
except ImportError:
    HAS_NUMBA = False

    def njit(func):  # no-op decorator when numba is unavailable
        return func


# ── Student-t density helpers (scalar, use scipy for numerical stability) ────

def student_t_logpdf(y: float, mu: float, h: float, nu: float) -> float:
    """Log PDF of Student-t with mean *mu*, conditional variance *h*, df *nu*.

    The scale parameter s satisfies s^2 = h*(nu-2)/nu so that Var(eps) = h.
    Delegates to scipy.stats.t for numerical accuracy.
    """
    s = np.sqrt(h * (nu - 2.0) / nu)
    return float(scipy.stats.t.logpdf(y, df=nu, loc=mu, scale=s))


def student_t_logcdf(y: float, mu: float, h: float, nu: float) -> float:
    """Log CDF of Student-t with mean *mu*, conditional variance *h*, df *nu*."""
    s = np.sqrt(h * (nu - 2.0) / nu)
    return float(scipy.stats.t.logcdf(y, df=nu, loc=mu, scale=s))


def student_t_logsf(y: float, mu: float, h: float, nu: float) -> float:
    """Log survival function log(1 - CDF) of Student-t.

    Uses scipy.stats.t.logsf for numerical stability in the far right tail.
    """
    s = np.sqrt(h * (nu - 2.0) / nu)
    return float(scipy.stats.t.logsf(y, df=nu, loc=mu, scale=s))


# ── Vectorised Student-t log PDF ─────────────────────────────────────────────

def student_t_logpdf_vec(
    y: np.ndarray, mu: float, h: np.ndarray, nu: float
) -> np.ndarray:
    """Vectorised log PDF of Student-t over arrays *y* and *h*.

    Each element h[t] gives the conditional variance for observation y[t].
    """
    s = np.sqrt(h * (nu - 2.0) / nu)
    return scipy.stats.t.logpdf(y, df=nu, loc=mu, scale=s)


# ── GARCH(1,1) ───────────────────────────────────────────────────────────────

@njit
def garch_variance_path(
    omega: float,
    alpha: float,
    beta: float,
    eps: np.ndarray,
    h0: float,
) -> Tuple[np.ndarray, float]:
    """Compute the GARCH(1,1) conditional variance path.

    Recursion
    ---------
        h[0]  = h0   (initialised to unconditional / sample variance)
        h[t]  = omega + alpha*eps[t-1]^2 + beta*h[t-1],  t = 1 … T-1

    Parameters
    ----------
    omega, alpha, beta : GARCH parameters.
    eps : Residual array eps_t = y_t - mu, length T.
    h0  : Initial conditional variance (h for period 0).

    Returns
    -------
    (h, h_next)
        h      : Array of length T; h[t] is the conditional variance for eps[t].
        h_next : One-step-ahead forecast h_{T+1}.
    """
    T = len(eps)
    h = np.empty(T)
    h[0] = h0
    for t in range(1, T):
        h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
    h_next = omega + alpha * eps[T - 1] ** 2 + beta * h[T - 1]
    return h, h_next


@njit
def _garch_loglik_inner(
    omega: float,
    alpha: float,
    beta: float,
    mu: float,
    nu: float,
    y: np.ndarray,
    h0: float,
) -> Tuple[float, np.ndarray, float]:
    """GARCH(1,1) Student-t log-likelihood (numba inner loop).

    Computes the variance path and the log-likelihood in a single pass.
    The Student-t log PDF formula (with scale^2 = h*(nu-2)/nu) is:

        log f = lgamma((nu+1)/2) - lgamma(nu/2)
                - 0.5*log(pi*nu) - 0.5*log(s^2)
                - (nu+1)/2 * log(1 + eps^2 / (s^2 * nu))

    where s^2 = h * (nu-2) / nu.

    Returns
    -------
    (ll, h, h_next)
    """
    T = len(y)
    eps = y - mu

    # ── variance path ────────────────────────────────────────────────────────
    h = np.empty(T)
    h[0] = h0
    for t in range(1, T):
        h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
    h_next = omega + alpha * eps[T - 1] ** 2 + beta * h[T - 1]

    # ── log-likelihood accumulation ──────────────────────────────────────────
    lgamma_nu1 = math.lgamma((nu + 1.0) / 2.0)
    lgamma_nu = math.lgamma(nu / 2.0)
    log_const = lgamma_nu1 - lgamma_nu - 0.5 * math.log(math.pi * nu)

    ll = 0.0
    for t in range(T):
        s2 = h[t] * (nu - 2.0) / nu  # scale^2
        ll += (
            log_const
            - 0.5 * math.log(s2)
            - 0.5 * (nu + 1.0) * math.log(1.0 + eps[t] ** 2 / (s2 * nu))
        )
    return ll, h, h_next


def garch_loglik(
    omega: float,
    alpha: float,
    beta: float,
    mu: float,
    nu: float,
    y: np.ndarray,
    h0: Optional[float] = None,
) -> Tuple[float, np.ndarray, float]:
    """GARCH(1,1) Student-t log-likelihood (public interface).

    Parameters
    ----------
    omega, alpha, beta : GARCH parameters.
    mu                 : Conditional mean.
    nu                 : Student-t degrees of freedom (nu > 2).
    y                  : Observed return series (decimal).
    h0                 : Initial conditional variance; defaults to sample
                         variance of y.

    Returns
    -------
    (log_likelihood, variance_path, h_next)
    """
    if h0 is None:
        h0 = float(np.var(y))
    return _garch_loglik_inner(
        omega, alpha, beta, mu, nu, np.asarray(y, dtype=np.float64), h0
    )


# ── GJR(1,1) ─────────────────────────────────────────────────────────────────

@njit
def gjr_variance_path(
    omega: float,
    alpha1: float,
    alpha2: float,
    beta: float,
    eps: np.ndarray,
    h0: float,
) -> Tuple[np.ndarray, float]:
    """Compute the GJR(1,1) conditional variance path.

    Recursion
    ---------
        h[0] = h0
        h[t] = omega + alpha1*eps[t-1]^2 * I{eps[t-1] >= 0}
                     + alpha2*eps[t-1]^2 * I{eps[t-1] <  0}
                     + beta*h[t-1]

    Parameters
    ----------
    omega        : Constant term.
    alpha1       : ARCH coefficient for non-negative shocks.
    alpha2       : ARCH coefficient for negative shocks (leverage effect).
    beta         : GARCH coefficient.
    eps          : Residual array, length T.
    h0           : Initial conditional variance.

    Returns
    -------
    (h, h_next)
    """
    T = len(eps)
    h = np.empty(T)
    h[0] = h0
    for t in range(1, T):
        e_lag = eps[t - 1]
        if e_lag >= 0.0:
            h[t] = omega + alpha1 * e_lag ** 2 + beta * h[t - 1]
        else:
            h[t] = omega + alpha2 * e_lag ** 2 + beta * h[t - 1]
    e_last = eps[T - 1]
    if e_last >= 0.0:
        h_next = omega + alpha1 * e_last ** 2 + beta * h[T - 1]
    else:
        h_next = omega + alpha2 * e_last ** 2 + beta * h[T - 1]
    return h, h_next


@njit
def _gjr_loglik_inner(
    omega: float,
    alpha1: float,
    alpha2: float,
    beta: float,
    mu: float,
    nu: float,
    y: np.ndarray,
    h0: float,
) -> Tuple[float, np.ndarray, float]:
    """GJR(1,1) Student-t log-likelihood (numba inner loop).

    Returns
    -------
    (ll, h, h_next)
    """
    T = len(y)
    eps = y - mu

    # ── variance path ────────────────────────────────────────────────────────
    h = np.empty(T)
    h[0] = h0
    for t in range(1, T):
        e_lag = eps[t - 1]
        if e_lag >= 0.0:
            h[t] = omega + alpha1 * e_lag ** 2 + beta * h[t - 1]
        else:
            h[t] = omega + alpha2 * e_lag ** 2 + beta * h[t - 1]
    e_last = eps[T - 1]
    if e_last >= 0.0:
        h_next = omega + alpha1 * e_last ** 2 + beta * h[T - 1]
    else:
        h_next = omega + alpha2 * e_last ** 2 + beta * h[T - 1]

    # ── log-likelihood accumulation ──────────────────────────────────────────
    lgamma_nu1 = math.lgamma((nu + 1.0) / 2.0)
    lgamma_nu = math.lgamma(nu / 2.0)
    log_const = lgamma_nu1 - lgamma_nu - 0.5 * math.log(math.pi * nu)

    ll = 0.0
    for t in range(T):
        s2 = h[t] * (nu - 2.0) / nu
        ll += (
            log_const
            - 0.5 * math.log(s2)
            - 0.5 * (nu + 1.0) * math.log(1.0 + eps[t] ** 2 / (s2 * nu))
        )
    return ll, h, h_next


def gjr_loglik(
    omega: float,
    alpha1: float,
    alpha2: float,
    beta: float,
    mu: float,
    nu: float,
    y: np.ndarray,
    h0: Optional[float] = None,
) -> Tuple[float, np.ndarray, float]:
    """GJR(1,1) Student-t log-likelihood (public interface).

    Parameters
    ----------
    omega, alpha1, alpha2, beta : GJR parameters.
    mu                          : Conditional mean.
    nu                          : Student-t degrees of freedom (nu > 2).
    y                           : Observed return series (decimal).
    h0                          : Initial conditional variance; defaults to
                                  sample variance of y.

    Returns
    -------
    (log_likelihood, variance_path, h_next)
    """
    if h0 is None:
        h0 = float(np.var(y))
    return _gjr_loglik_inner(
        omega, alpha1, alpha2, beta, mu, nu,
        np.asarray(y, dtype=np.float64), h0,
    )


# ── EGARCH(1,1) ───────────────────────────────────────────────────────────────

@njit
def egarch_variance_path(
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    eps: np.ndarray,
    h0: float,
) -> Tuple[np.ndarray, float]:
    """Compute the EGARCH(1,1) conditional variance path.

    Recursion (no centering term E|z_t|)
    -------------------------------------
        log h[0] = log h0
        log h[t] = omega + alpha*|z[t-1]| + gamma*z[t-1] + beta*log h[t-1]
        where z[t] = eps[t] / sqrt(h[t])

    Parameters
    ----------
    omega        : Constant term.
    alpha        : Coefficient on |z| (magnitude effect).
    gamma        : Coefficient on z (sign / asymmetry effect).
    beta         : GARCH coefficient in log-variance equation.
    eps          : Residual array, length T.
    h0           : Initial conditional variance.

    Returns
    -------
    (h, h_next)
    """
    T = len(eps)
    h = np.empty(T)
    log_h = np.empty(T)
    h[0] = h0
    log_h[0] = math.log(h0)

    for t in range(1, T):
        z = eps[t - 1] / math.sqrt(h[t - 1])
        log_h[t] = omega + alpha * math.fabs(z) + gamma * z + beta * log_h[t - 1]
        h[t] = math.exp(log_h[t])

    z_last = eps[T - 1] / math.sqrt(h[T - 1])
    log_h_next = omega + alpha * math.fabs(z_last) + gamma * z_last + beta * log_h[T - 1]
    h_next = math.exp(log_h_next)
    return h, h_next


@njit
def _egarch_loglik_inner(
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    mu: float,
    nu: float,
    y: np.ndarray,
    h0: float,
) -> Tuple[float, np.ndarray, float]:
    """EGARCH(1,1) Student-t log-likelihood (numba inner loop).

    Returns
    -------
    (ll, h, h_next)
    """
    T = len(y)
    eps = y - mu

    # ── variance path ────────────────────────────────────────────────────────
    h = np.empty(T)
    log_h = np.empty(T)
    h[0] = h0
    log_h[0] = math.log(h0)

    for t in range(1, T):
        z = eps[t - 1] / math.sqrt(h[t - 1])
        log_h[t] = omega + alpha * math.fabs(z) + gamma * z + beta * log_h[t - 1]
        h[t] = math.exp(log_h[t])

    z_last = eps[T - 1] / math.sqrt(h[T - 1])
    log_h_next = omega + alpha * math.fabs(z_last) + gamma * z_last + beta * log_h[T - 1]
    h_next = math.exp(log_h_next)

    # ── log-likelihood accumulation ──────────────────────────────────────────
    lgamma_nu1 = math.lgamma((nu + 1.0) / 2.0)
    lgamma_nu = math.lgamma(nu / 2.0)
    log_const = lgamma_nu1 - lgamma_nu - 0.5 * math.log(math.pi * nu)

    ll = 0.0
    for t in range(T):
        s2 = h[t] * (nu - 2.0) / nu
        ll += (
            log_const
            - 0.5 * math.log(s2)
            - 0.5 * (nu + 1.0) * math.log(1.0 + eps[t] ** 2 / (s2 * nu))
        )
    return ll, h, h_next


def egarch_loglik(
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    mu: float,
    nu: float,
    y: np.ndarray,
    h0: Optional[float] = None,
) -> Tuple[float, np.ndarray, float]:
    """EGARCH(1,1) Student-t log-likelihood (public interface).

    Parameters
    ----------
    omega, alpha, gamma, beta : EGARCH parameters.
    mu                        : Conditional mean.
    nu                        : Student-t degrees of freedom (nu > 2).
    y                         : Observed return series (decimal).
    h0                        : Initial conditional variance; defaults to
                                sample variance of y.

    Returns
    -------
    (log_likelihood, variance_path, h_next)
    """
    if h0 is None:
        h0 = float(np.var(y))
    return _egarch_loglik_inner(
        omega, alpha, gamma, beta, mu, nu,
        np.asarray(y, dtype=np.float64), h0,
    )
