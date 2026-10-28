"""Tests for src/estimation.py.

Covers both frequentist MLE (via ``frequentist_estimate``) and Bayesian MCMC
(via ``bayesian_estimate``) for all three model families.  Tests use a short
simulated GARCH(1,1) series to keep runtime manageable while exercising real
code paths.
"""

import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest
from scipy.stats import t as scipy_t

from src.estimation import frequentist_estimate, bayesian_estimate


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def simulated_returns():
    """Generate a synthetic GARCH(1,1) Student-t return series for testing."""
    np.random.seed(42)
    T = 300
    omega, alpha, beta, mu, nu = 1e-5, 0.05, 0.90, 1e-4, 8.0
    h = np.empty(T)
    y = np.empty(T)
    h[0] = omega / (1 - alpha - beta)
    for i in range(T):
        s = np.sqrt(h[i] * (nu - 2) / nu)
        y[i] = mu + scipy_t.rvs(df=nu, scale=s)
        if i < T - 1:
            eps = y[i] - mu
            h[i + 1] = omega + alpha * eps ** 2 + beta * h[i]
    return y


# ---------------------------------------------------------------------------
# Frequentist MLE
# ---------------------------------------------------------------------------

class TestFrequentistEstimate:
    def test_garch_returns_dict(self, simulated_returns):
        result = frequentist_estimate(simulated_returns, 'garch')
        assert isinstance(result, dict)
        assert 'params' in result
        assert 'h_forecast' in result
        assert 'loglik' in result

    def test_garch_h_forecast_positive(self, simulated_returns):
        result = frequentist_estimate(simulated_returns, 'garch')
        assert result['h_forecast'] > 0

    def test_garch_params_sensible(self, simulated_returns):
        result = frequentist_estimate(simulated_returns, 'garch')
        p = result['params']
        assert p['omega'] > 0
        assert p['alpha'] >= 0
        assert p['beta'] >= 0
        assert p['alpha'] + p['beta'] < 1.0, 'Stationarity constraint'
        assert p['nu'] > 2

    def test_garch_loglik_finite(self, simulated_returns):
        result = frequentist_estimate(simulated_returns, 'garch')
        # Log-lik can be positive for narrow densities (returns scaled to ≈ 0.01).
        assert np.isfinite(result['loglik'])

    def test_gjr_params_valid(self, simulated_returns):
        result = frequentist_estimate(simulated_returns, 'gjr')
        p = result['params']
        assert p['omega'] > 0
        assert p['alpha1'] >= 0
        assert p['alpha2'] >= 0
        assert p['beta'] >= 0
        assert p['nu'] > 2

    def test_gjr_h_forecast_positive(self, simulated_returns):
        result = frequentist_estimate(simulated_returns, 'gjr')
        assert result['h_forecast'] > 0

    def test_egarch_h_forecast_positive(self, simulated_returns):
        result = frequentist_estimate(simulated_returns, 'egarch')
        assert result['h_forecast'] > 0

    def test_egarch_params_valid(self, simulated_returns):
        result = frequentist_estimate(simulated_returns, 'egarch')
        p = result['params']
        assert p['nu'] > 2
        # The persistence parameter is named 'beta' (not 'phi') in MODEL_PARAMS.
        assert abs(p['beta']) < 1.0, 'EGARCH log-variance persistence |β| < 1 for stationarity'

    def test_invalid_model_raises(self, simulated_returns):
        with pytest.raises((ValueError, KeyError)):
            frequentist_estimate(simulated_returns, 'unknown_model')


# ---------------------------------------------------------------------------
# Bayesian MCMC
# ---------------------------------------------------------------------------

class TestBayesianEstimate:
    # Use small burn-in / draw counts so tests run fast
    N_BURN = 200
    N_DRAWS = 500

    def test_garch_returns_dict_keys(self, simulated_returns):
        result = bayesian_estimate(simulated_returns, 'garch',
                                   n_burn=self.N_BURN, n_draws=self.N_DRAWS)
        assert 'draws' in result
        assert 'acceptance_rate' in result
        assert 'h_forecasts' in result
        assert 'param_names' in result

    def test_garch_acceptance_rate_reasonable(self, simulated_returns):
        result = bayesian_estimate(simulated_returns, 'garch',
                                   n_burn=self.N_BURN, n_draws=self.N_DRAWS)
        ar = result['acceptance_rate']
        # Independence-chain MH acceptance rate depends entirely on how well
        # the MVT proposal (centred at MLE mode) covers the posterior.
        # For small samples (T=300) the rate can be low. We only verify the
        # chain is operating: at least some proposals accepted, not all.
        assert 0.0 < ar < 1.0, f'Acceptance rate {ar:.3f} not in (0, 1)'

    def test_garch_draws_shape(self, simulated_returns):
        result = bayesian_estimate(simulated_returns, 'garch',
                                   n_burn=self.N_BURN, n_draws=self.N_DRAWS)
        assert result['draws'].shape[0] == self.N_DRAWS

    def test_garch_h_forecasts_shape_and_positive(self, simulated_returns):
        result = bayesian_estimate(simulated_returns, 'garch',
                                   n_burn=self.N_BURN, n_draws=self.N_DRAWS)
        assert result['h_forecasts'].shape[0] == self.N_DRAWS
        assert np.all(result['h_forecasts'] > 0)

    def test_garch_omega_positive_across_draws(self, simulated_returns):
        result = bayesian_estimate(simulated_returns, 'garch',
                                   n_burn=self.N_BURN, n_draws=self.N_DRAWS)
        draws = result['draws']
        param_names = result['param_names']
        omega_idx = param_names.index('omega')
        assert np.all(draws[:, omega_idx] > 0), 'omega must be positive in all draws'

    def test_garch_nu_greater_than_two(self, simulated_returns):
        result = bayesian_estimate(simulated_returns, 'garch',
                                   n_burn=self.N_BURN, n_draws=self.N_DRAWS)
        draws = result['draws']
        param_names = result['param_names']
        nu_idx = param_names.index('nu')
        assert np.all(draws[:, nu_idx] > 2), 'nu must be > 2 for finite variance'

    def test_gjr_draws_positive_omega_and_nu(self, simulated_returns):
        result = bayesian_estimate(simulated_returns, 'gjr',
                                   n_burn=100, n_draws=200)
        draws = result['draws']
        param_names = result['param_names']
        omega_idx = param_names.index('omega')
        nu_idx = param_names.index('nu')
        assert np.all(draws[:, omega_idx] > 0)
        assert np.all(draws[:, nu_idx] > 2)

    def test_egarch_h_forecasts_positive(self, simulated_returns):
        result = bayesian_estimate(simulated_returns, 'egarch',
                                   n_burn=100, n_draws=200)
        assert np.all(result['h_forecasts'] > 0)

    def test_bayesian_param_names_nonempty(self, simulated_returns):
        for model in ('garch', 'gjr', 'egarch'):
            result = bayesian_estimate(simulated_returns, model,
                                       n_burn=50, n_draws=100)
            assert len(result['param_names']) > 0
            assert result['draws'].shape[1] == len(result['param_names'])
