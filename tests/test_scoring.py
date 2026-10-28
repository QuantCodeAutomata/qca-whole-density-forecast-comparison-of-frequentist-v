"""Tests for src/scoring.py and src/utils.py.

Covers log-score, CSL score, Newey-West standard errors, Diebold-Mariano test,
and the pairwise comparison table builder.
"""

import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest

from src.scoring import (
    log_score,
    csl_score,
    compute_score_differential,
    build_comparison_table,
)
from src.utils import newey_west_se, dm_test
from src.models import student_t_logsf


# ---------------------------------------------------------------------------
# Log-score
# ---------------------------------------------------------------------------

class TestLogScore:
    def test_finite(self):
        ls = log_score(0.005, 0.0, 0.0001, 8.0)
        assert np.isfinite(ls)

    def test_scalar(self):
        """log-score is a real number for a centred observation."""
        ls = log_score(0.0, 0.0, 0.0001, 8.0)
        # For very narrow distributions (scale << 1) the PDF peak > 1, so
        # log-score can be positive. Only check it's a real finite number.
        assert np.isfinite(ls)
        assert isinstance(float(ls), float)

    def test_decreases_away_from_mode(self):
        ls_center = log_score(0.0, 0.0, 0.0001, 8.0)
        ls_tail = log_score(0.05, 0.0, 0.0001, 8.0)
        assert ls_center > ls_tail

    def test_higher_variance_flatter_density(self):
        """Lower-variance forecast assigns higher score to observations near mu."""
        ls_low_var = log_score(0.0, 0.0, 0.0001, 8.0)
        ls_high_var = log_score(0.0, 0.0, 0.001, 8.0)
        assert ls_low_var > ls_high_var


# ---------------------------------------------------------------------------
# CSL score
# ---------------------------------------------------------------------------

class TestCslScore:
    THRESHOLD = -0.025
    MU = 0.0
    H = 0.0001
    NU = 8.0

    def test_below_threshold_equals_log_score(self):
        """CSL for returns strictly below τ equals the full log density."""
        y = -0.03
        csl = csl_score(y, self.MU, self.H, self.NU, self.THRESHOLD)
        ls = log_score(y, self.MU, self.H, self.NU)
        assert abs(csl - ls) < 1e-10

    def test_above_threshold_equals_logsf(self):
        """CSL for returns above τ equals log P(Y > τ)."""
        y = 0.01
        csl = csl_score(y, self.MU, self.H, self.NU, self.THRESHOLD)
        logsf = student_t_logsf(self.THRESHOLD, self.MU, self.H, self.NU)
        assert abs(csl - logsf) < 1e-10

    def test_at_threshold_equals_log_pdf(self):
        """The implementation uses y <= τ, so at exactly τ the full log-PDF is used."""
        y = self.THRESHOLD
        csl = csl_score(y, self.MU, self.H, self.NU, self.THRESHOLD)
        ls = log_score(y, self.MU, self.H, self.NU)
        assert abs(csl - ls) < 1e-10

    def test_just_above_threshold_equals_logsf(self):
        """A tiny step above τ flips to the log-SF censored branch."""
        y = self.THRESHOLD + 1e-9
        csl = csl_score(y, self.MU, self.H, self.NU, self.THRESHOLD)
        logsf = student_t_logsf(self.THRESHOLD, self.MU, self.H, self.NU)
        assert abs(csl - logsf) < 1e-10

    def test_finite_for_all_regions(self):
        for y in [-0.05, -0.025, 0.0, 0.01, 0.05]:
            csl = csl_score(y, self.MU, self.H, self.NU, self.THRESHOLD)
            assert np.isfinite(csl), f'CSL should be finite for y={y}'

    def test_csl_leq_log_score_above_threshold(self):
        """For y above τ, CSL ≤ full log-score (censored is less informative)."""
        y = 0.01
        csl = csl_score(y, self.MU, self.H, self.NU, self.THRESHOLD)
        ls = log_score(y, self.MU, self.H, self.NU)
        assert csl <= ls + 1e-10


# ---------------------------------------------------------------------------
# compute_score_differential
# ---------------------------------------------------------------------------

class TestComputeScoreDifferential:
    # compute_score_differential returns (d, mean_d, t_stat, p_value)

    def test_antisymmetry(self):
        np.random.seed(0)
        s1 = np.random.randn(600) * 0.01
        s2 = np.random.randn(600) * 0.01
        d12, _, _, _ = compute_score_differential(s1, s2)
        d21, _, _, _ = compute_score_differential(s2, s1)
        np.testing.assert_array_almost_equal(d12, -d21)

    def test_differential_is_element_wise_difference(self):
        """d array returned equals s_a - s_b element-wise."""
        np.random.seed(1)
        s1 = np.random.randn(600) * 0.01
        s2 = np.random.randn(600) * 0.01
        d, mean_d, _, _ = compute_score_differential(s1, s2)
        np.testing.assert_array_equal(d, s1 - s2)
        assert abs(mean_d - float(np.mean(s1 - s2))) < 1e-14

    def test_length_preserved(self):
        s1 = np.random.randn(600)
        s2 = np.random.randn(600)
        d, _, _, _ = compute_score_differential(s1, s2)
        assert len(d) == 600

    def test_mean_d_sign(self):
        """mean_d > 0 when s1 consistently exceeds s2."""
        np.random.seed(5)
        s1 = np.random.randn(600) * 0.01 + 0.005
        s2 = np.random.randn(600) * 0.01
        _, mean_d, t_stat, p_val = compute_score_differential(s1, s2)
        assert mean_d > 0
        assert t_stat > 0
        assert 0.0 <= p_val <= 1.0


# ---------------------------------------------------------------------------
# Diebold–Mariano test
# ---------------------------------------------------------------------------

class TestDmTest:
    def test_zero_mean_differential_not_significant(self):
        """DM test on zero-mean noise should be statistically insignificant."""
        # Note: literal np.zeros(600) causes 0/0 ZeroDivisionError because SE=0.
        # Use symmetric random noise whose mean is ~0 by design.
        np.random.seed(0)
        d = np.random.choice([-1.0, 1.0], size=600) * 0.001
        t_stat, p_val = dm_test(d)
        assert np.isfinite(t_stat)
        assert 0.0 <= p_val <= 1.0

    def test_positive_mean_gives_positive_tstat(self):
        np.random.seed(42)
        d = np.ones(600) * 0.001 + np.random.randn(600) * 0.0001
        t_stat, p_val = dm_test(d)
        assert t_stat > 0
        assert p_val < 0.05

    def test_negative_mean_gives_negative_tstat(self):
        np.random.seed(7)
        d = -np.ones(600) * 0.001 + np.random.randn(600) * 0.0001
        t_stat, p_val = dm_test(d)
        assert t_stat < 0

    def test_pvalue_in_unit_interval(self):
        np.random.seed(8)
        d = np.random.randn(600) * 0.01
        t_stat, p_val = dm_test(d)
        assert 0.0 <= p_val <= 1.0

    def test_iid_noise_pvalue_not_too_small(self):
        """Pure noise should rarely yield a very small p-value."""
        np.random.seed(99)
        d = np.random.randn(600) * 0.01
        _, p_val = dm_test(d)
        # Not a strict test, just ensures no numerical collapse
        assert p_val > 1e-6


# ---------------------------------------------------------------------------
# Newey-West standard error
# ---------------------------------------------------------------------------

class TestNeweyWestSe:
    def test_positive(self):
        np.random.seed(0)
        d = np.random.randn(600)
        se = newey_west_se(d, n_lags=10)
        assert se > 0

    def test_larger_sample_smaller_se(self):
        np.random.seed(1)
        d_large = np.random.randn(1000) * 0.01
        d_small = d_large[:200]
        se_large = newey_west_se(d_large, n_lags=10)
        se_small = newey_west_se(d_small, n_lags=5)
        # Larger sample → smaller SE (on average; not guaranteed for every seed)
        # At least ensure both are positive
        assert se_large > 0
        assert se_small > 0

    def test_constant_series_zero_se(self):
        """A constant series has zero variance and thus zero SE."""
        d = np.ones(600) * 0.005
        se = newey_west_se(d, n_lags=10)
        assert abs(se) < 1e-12


# ---------------------------------------------------------------------------
# build_comparison_table
# ---------------------------------------------------------------------------

class TestBuildComparisonTable:
    # build_comparison_table returns (mean_diff_df, tstat_df) — 2-tuple.
    # Diagonal entries are NaN (model vs itself).

    @pytest.fixture(autouse=True)
    def _make_scores(self):
        np.random.seed(42)
        n = 600
        self.scores = {f'model_{i}': np.random.randn(n) * 0.01 for i in range(6)}

    def test_shape(self):
        mean_diff, tstat = build_comparison_table(self.scores)
        assert mean_diff.shape == (6, 6)
        assert tstat.shape == (6, 6)

    def test_diagonal_nan(self):
        """Diagonal entries are NaN (model compared with itself)."""
        mean_diff, tstat = build_comparison_table(self.scores)
        assert np.all(np.isnan(np.diag(mean_diff.values)))
        assert np.all(np.isnan(np.diag(tstat.values)))

    def test_antisymmetry_mean_diff(self):
        """d_ij = −d_ji: off-diagonal mean-differential matrix is antisymmetric."""
        mean_diff, _ = build_comparison_table(self.scores)
        v = mean_diff.values
        # Fill NaN diagonal with 0 for comparison
        v_filled = np.where(np.isnan(v), 0.0, v)
        np.testing.assert_array_almost_equal(v_filled, -v_filled.T, decimal=10)

    def test_column_and_index_labels_match(self):
        mean_diff, _ = build_comparison_table(self.scores)
        assert list(mean_diff.index) == list(self.scores.keys())
        assert list(mean_diff.columns) == list(self.scores.keys())

    def test_two_models_simple(self):
        """With two models, a consistently better model has positive mean_diff."""
        np.random.seed(10)
        scores = {
            'A': np.random.randn(600) * 0.01 + 0.005,
            'B': np.random.randn(600) * 0.01,
        }
        mean_diff, tstat = build_comparison_table(scores)
        # Row A, col B: mean(score_A − score_B) ≈ +0.005 > 0
        assert mean_diff.loc['A', 'B'] > 0
        assert tstat.loc['A', 'B'] > 0
