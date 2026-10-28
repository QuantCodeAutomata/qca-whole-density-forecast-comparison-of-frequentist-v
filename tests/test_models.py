"""Tests for src/models.py.

Covers Student-t distribution helpers and GARCH-family variance recursions /
log-likelihoods for correctness, positivity, and basic statistical properties.
"""

import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest
from scipy.stats import t as scipy_t

from src.models import (
    student_t_logcdf,
    student_t_logpdf,
    student_t_logsf,
    garch_variance_path,
    gjr_variance_path,
    egarch_variance_path,
    garch_loglik,
    gjr_loglik,
    egarch_loglik,
)


# ---------------------------------------------------------------------------
# Student-t helpers
# ---------------------------------------------------------------------------

class TestStudentTLogpdf:
    def test_scalar_finite(self):
        """log PDF at the centre is finite."""
        logpdf = student_t_logpdf(0.0, 0.0, 0.0001, 8.0)
        assert np.isfinite(logpdf)
        # For very narrow distributions (scale << 1) the PDF peak exceeds 1,
        # so logpdf can be positive — only finiteness is checked here.
        # Monotone-tails test below verifies the shape is correct.

    def test_variance_matches_h(self):
        """Student-t parameterised by h has variance equal to h."""
        np.random.seed(42)
        nu = 8.0
        h = 0.0001
        # scale s such that Var = s^2 * nu/(nu-2) = h  =>  s = sqrt(h*(nu-2)/nu)
        s = np.sqrt(h * (nu - 2) / nu)
        samples = scipy_t.rvs(df=nu, loc=0, scale=s, size=20_000, random_state=42)
        assert abs(np.var(samples) - h) < 0.15 * h, (
            f'Student-t variance {np.var(samples):.6f} should be close to h={h}'
        )

    def test_symmetric_around_mu(self):
        """log PDF should be symmetric around the mean mu."""
        mu, h, nu = 0.001, 0.0001, 6.0
        delta = 0.005
        lp_left = student_t_logpdf(mu - delta, mu, h, nu)
        lp_right = student_t_logpdf(mu + delta, mu, h, nu)
        assert abs(lp_left - lp_right) < 1e-10

    def test_monotone_tails(self):
        """log PDF decreases as |y| moves away from mu."""
        mu, h, nu = 0.0, 0.0001, 8.0
        lp_center = student_t_logpdf(mu, mu, h, nu)
        lp_moderate = student_t_logpdf(mu + 0.01, mu, h, nu)
        lp_far = student_t_logpdf(mu + 0.05, mu, h, nu)
        assert lp_center > lp_moderate > lp_far

    def test_matches_scipy_t(self):
        """Results must match scipy.stats.t evaluated at the same parameters."""
        mu, h, nu = 0.0, 0.0001, 8.0
        s = np.sqrt(h * (nu - 2) / nu)
        for y in [-0.03, -0.01, 0.0, 0.01, 0.03]:
            expected = float(scipy_t.logpdf(y, df=nu, loc=mu, scale=s))
            got = student_t_logpdf(y, mu, h, nu)
            assert abs(got - expected) < 1e-12, (
                f'Mismatch at y={y}: got {got}, expected {expected}'
            )


class TestStudentTLogsfLogcdf:
    def test_logsf_right_tail_decreases(self):
        """log SF = log P(X > y) decreases as y moves further RIGHT of the mean."""
        # logsf(0.01) > logsf(0.05): P(X > 0.01) > P(X > 0.05)
        logsf_moderate = student_t_logsf(0.01, 0.0, 0.0001, 8.0)
        logsf_extreme = student_t_logsf(0.05, 0.0, 0.0001, 8.0)
        assert logsf_extreme < logsf_moderate

    def test_logcdf_left_tail_decreases(self):
        """log CDF = log P(X ≤ y) decreases as y moves further LEFT of the mean."""
        # logcdf(-0.01) > logcdf(-0.05): P(X ≤ -0.01) > P(X ≤ -0.05)
        logcdf_moderate = student_t_logcdf(-0.01, 0.0, 0.0001, 8.0)
        logcdf_extreme = student_t_logcdf(-0.05, 0.0, 0.0001, 8.0)
        assert logcdf_extreme < logcdf_moderate

    def test_cdf_sf_complement(self):
        """CDF + SF must equal 1 for any input."""
        y, mu, h, nu = -0.01, 0.0, 0.0001, 6.0
        cdf = np.exp(student_t_logcdf(y, mu, h, nu))
        sf = np.exp(student_t_logsf(y, mu, h, nu))
        assert abs(cdf + sf - 1.0) < 1e-10

    def test_logcdf_at_median_near_log_half(self):
        """CDF at the median (mu) should be ~0.5."""
        logcdf = student_t_logcdf(0.0, 0.0, 0.0001, 8.0)
        assert abs(np.exp(logcdf) - 0.5) < 1e-6

    def test_logsf_finite_for_moderate_returns(self):
        for y in [-0.05, -0.025, 0.0, 0.01]:
            assert np.isfinite(student_t_logsf(y, 0.0, 0.0001, 8.0))
            assert np.isfinite(student_t_logcdf(y, 0.0, 0.0001, 8.0))


# ---------------------------------------------------------------------------
# GARCH variance path
# ---------------------------------------------------------------------------

class TestGarchVariancePath:
    def test_all_positive(self):
        np.random.seed(0)
        eps = np.random.randn(100) * 0.01
        h, h_next = garch_variance_path(1e-6, 0.05, 0.9, eps, h0=0.0001)
        assert np.all(h > 0), 'All in-sample variances must be positive'
        assert h_next > 0, 'One-step-ahead forecast must be positive'

    def test_length_matches_input(self):
        eps = np.random.randn(200) * 0.01
        h, h_next = garch_variance_path(1e-6, 0.05, 0.9, eps, h0=0.0001)
        assert len(h) == 200

    def test_persistence_drives_variance(self):
        """High beta leads to more persistent variance responses."""
        np.random.seed(1)
        eps = np.random.randn(100) * 0.01
        eps[0] = 0.05  # large shock at t=0
        _, h_next_high = garch_variance_path(1e-6, 0.05, 0.93, eps, 0.0001)
        _, h_next_low = garch_variance_path(1e-6, 0.05, 0.70, eps, 0.0001)
        assert h_next_high > h_next_low, 'Higher beta should sustain larger variance'

    def test_stationary_params(self):
        """With α+β < 1 the path should not explode."""
        np.random.seed(2)
        eps = np.random.randn(500) * 0.01
        h, _ = garch_variance_path(1e-6, 0.05, 0.90, eps, 0.0001)
        assert np.all(np.isfinite(h))


# ---------------------------------------------------------------------------
# GJR variance path
# ---------------------------------------------------------------------------

class TestGjrVariancePath:
    def test_asymmetry_negative_over_positive(self):
        """Negative shocks must increase variance more than positive shocks when alpha2 > alpha1."""
        eps_pos = np.array([0.01] * 10 + [0.0] * 90)
        eps_neg = np.array([-0.01] * 10 + [0.0] * 90)
        h0 = 0.0001
        h_pos, _ = gjr_variance_path(1e-6, 0.03, 0.10, 0.88, eps_pos, h0)
        h_neg, _ = gjr_variance_path(1e-6, 0.03, 0.10, 0.88, eps_neg, h0)
        assert h_neg[10] > h_pos[10], (
            'GJR: negative shocks should produce higher variance when alpha2 > alpha1'
        )

    def test_all_positive(self):
        np.random.seed(3)
        eps = np.random.randn(100) * 0.01
        h, h_next = gjr_variance_path(1e-6, 0.03, 0.10, 0.88, eps, h0=0.0001)
        assert np.all(h > 0)
        assert h_next > 0

    def test_length_matches_input(self):
        eps = np.random.randn(150) * 0.01
        h, _ = gjr_variance_path(1e-6, 0.03, 0.07, 0.88, eps, h0=0.0001)
        assert len(h) == 150

    def test_symmetric_case_equals_garch(self):
        """When alpha1 == alpha2, GJR applies the same coefficient to all shocks
        and its variance path must exactly equal GARCH(alpha=alpha1)."""
        np.random.seed(4)
        eps = np.random.randn(100) * 0.01
        h0 = 0.0001
        omega, alpha, beta = 1e-6, 0.05, 0.90
        h_garch, hn_garch = garch_variance_path(omega, alpha, beta, eps, h0)
        # GJR: pos-eps branch uses alpha1, neg-eps branch uses alpha2;
        # setting alpha1 = alpha2 = alpha gives GARCH.
        h_gjr, hn_gjr = gjr_variance_path(omega, alpha, alpha, beta, eps, h0)
        np.testing.assert_allclose(h_gjr, h_garch, rtol=1e-10)
        assert abs(hn_gjr - hn_garch) < 1e-14


# ---------------------------------------------------------------------------
# EGARCH variance path
# ---------------------------------------------------------------------------

class TestEgarchVariancePath:
    def test_all_positive(self):
        np.random.seed(5)
        eps = np.random.randn(100) * 0.01
        h, h_next = egarch_variance_path(-0.1, 0.1, -0.05, 0.95, eps, h0=0.0001)
        assert np.all(h > 0)
        assert h_next > 0

    def test_log_variance_finite(self):
        np.random.seed(6)
        eps = np.random.randn(100) * 0.01
        h, _ = egarch_variance_path(-0.1, 0.1, -0.05, 0.95, eps, h0=0.0001)
        assert np.all(np.isfinite(np.log(h)))

    def test_length_matches_input(self):
        eps = np.random.randn(80) * 0.01
        h, _ = egarch_variance_path(-0.1, 0.1, -0.05, 0.95, eps, h0=0.0001)
        assert len(h) == 80

    def test_asymmetric_response(self):
        """Negative shocks amplify log-variance more than positive when gamma < 0."""
        eps_pos = np.array([0.02] * 5 + [0.0] * 95)
        eps_neg = np.array([-0.02] * 5 + [0.0] * 95)
        h0 = 0.0001
        h_pos, _ = egarch_variance_path(-0.1, 0.1, -0.05, 0.95, eps_pos, h0)
        h_neg, _ = egarch_variance_path(-0.1, 0.1, -0.05, 0.95, eps_neg, h0)
        assert h_neg[5] > h_pos[5], (
            'EGARCH: negative shocks should produce higher variance when gamma < 0'
        )


# ---------------------------------------------------------------------------
# Log-likelihood functions
# ---------------------------------------------------------------------------

class TestLogLikelihoods:
    # garch_loglik  signature: (omega, alpha, beta,          mu, nu, y, h0)
    # gjr_loglik    signature: (omega, alpha1, alpha2, beta, mu, nu, y, h0)
    # egarch_loglik signature: (omega, alpha, gamma, beta,   mu, nu, y, h0)

    @pytest.fixture(autouse=True)
    def _make_data(self):
        np.random.seed(42)
        self.y = np.random.randn(500) * 0.01
        self.h0 = float(np.var(self.y))

    def test_garch_loglik_finite(self):
        # omega=5e-6 gives unconditional var = 5e-6/(1-0.05-0.9)=0.0001 ≈ var(y)
        ll, h, h_next = garch_loglik(5e-6, 0.05, 0.90, 0.0, 8.0, self.y, self.h0)
        assert np.isfinite(ll), 'GARCH log-lik must be finite'
        # Note: log-lik can be positive for narrow densities (scale << 1).

    def test_garch_loglik_returns_correct_shapes(self):
        ll, h, h_next = garch_loglik(5e-6, 0.05, 0.90, 0.0, 8.0, self.y, self.h0)
        assert len(h) == len(self.y)
        assert isinstance(h_next, float)

    def test_gjr_loglik_finite(self):
        ll, h, h_next = gjr_loglik(1e-6, 0.03, 0.07, 0.88, 0.0, 8.0, self.y, self.h0)
        assert np.isfinite(ll)

    def test_gjr_loglik_returns_correct_shapes(self):
        ll, h, h_next = gjr_loglik(1e-6, 0.03, 0.07, 0.88, 0.0, 8.0, self.y, self.h0)
        assert len(h) == len(self.y)
        assert isinstance(h_next, float)

    def test_egarch_loglik_finite(self):
        # omega=-0.1, alpha=0.1, gamma=-0.05, beta=0.95 → well-behaved log-variance path
        ll, h, h_next = egarch_loglik(-0.1, 0.1, -0.05, 0.95, 0.0, 8.0, self.y, self.h0)
        assert np.isfinite(ll)

    def test_egarch_loglik_returns_correct_shapes(self):
        ll, h, h_next = egarch_loglik(-0.1, 0.1, -0.05, 0.95, 0.0, 8.0, self.y, self.h0)
        assert len(h) == len(self.y)
        assert isinstance(h_next, float)

    def test_garch_loglik_sensitive_to_params(self):
        """Near-true params should give higher log-lik than implausible ones."""
        # Good params: unconditional var ≈ var(y); mu=0; nu=8
        ll_good, _, _ = garch_loglik(5e-6, 0.05, 0.90, 0.0, 8.0, self.y, self.h0)
        # Bad params: unconditional var 50× too large; heavy α; low nu
        ll_bad, _, _ = garch_loglik(5e-4, 0.40, 0.40, 0.0, 3.0, self.y, self.h0)
        assert ll_good > ll_bad, 'Reasonable params should yield higher log-lik'
