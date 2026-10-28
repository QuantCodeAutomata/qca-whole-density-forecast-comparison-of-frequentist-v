import math
from typing import Optional, Tuple

import numpy as np
from scipy import stats


def newey_west_se(d: np.ndarray, n_lags: int) -> float:
    """Compute Newey-West standard error of the mean of d.

    Uses the Bartlett kernel with bandwidth n_lags:
        NW_var = gamma_0 + 2 * sum_{h=1}^{L} w_h * gamma_h
    where w_h = 1 - h/(L+1) and gamma_h is the sample autocovariance at lag h
    (normalised by T, not T-h).

    Returns sqrt(NW_var / T) where T = len(d).
    """
    T = len(d)
    d_dm = d - d.mean()  # demeaned series

    gamma_0 = np.dot(d_dm, d_dm) / T
    nw_var = gamma_0

    for h in range(1, n_lags + 1):
        w_h = 1.0 - h / (n_lags + 1)
        gamma_h = np.dot(d_dm[h:], d_dm[:-h]) / T
        nw_var += 2.0 * w_h * gamma_h

    return math.sqrt(nw_var / T)


def dm_test(d: np.ndarray, n_lags: Optional[int] = None) -> Tuple[float, float]:
    """Diebold-Mariano test on score differentials d.

    Positive mean(d) means the model producing scores_a outperforms scores_b.

    Parameters
    ----------
    d:
        Array of pairwise score differentials (scores_a - scores_b).
    n_lags:
        Newey-West lag truncation.  Defaults to floor(4*(T/100)^(2/9)).

    Returns
    -------
    (t_stat, p_value)
        Two-sided test.  t_stat = mean(d) / NW_SE;
        p_value = 2*(1 - Phi(|t_stat|)).
    """
    T = len(d)
    if n_lags is None:
        n_lags = int(math.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))

    se = newey_west_se(d, n_lags)
    t_stat = float(d.mean()) / se
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(t_stat)))
    return t_stat, p_value
