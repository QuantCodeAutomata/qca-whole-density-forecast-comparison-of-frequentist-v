# AGENTS.md — Repository Knowledge Base

## Project Overview

Reproduces a paper comparing **frequentist MLE** vs **Bayesian MCMC** estimation
of GARCH-family volatility models for whole-density and left-tail density
forecasting on S&P 500 and Nikkei 225 daily returns.

- **Out-of-sample period**: 2008-01-01 – 2010-04-19 (600 one-step-ahead forecasts)
- **Estimation window**: rolling 10-year ≈ 2 520 trading days
- **Full data range**: ~1997-01-01 – 2010-04-20

---

## Architecture

| Module | Purpose |
|--------|---------|
| `src/models.py` | Numba-JIT variance recursions (`garch_variance_path`, `gjr_variance_path`, `egarch_variance_path`) and log-likelihood functions; Student-t `logpdf`, `logcdf`, `logsf` helpers |
| `src/estimation.py` | `frequentist_estimate` (L-BFGS-B MLE, SLSQP fallback, 3 random starts) and `bayesian_estimate` (independence-chain MH-MCMC) |
| `src/forecasting.py` | `rolling_forecast_window` — iterates over 600 forecast origins, fits all 6 model-method combinations, computes `log_score` and `csl_score` per forecast, returns a long-format DataFrame |
| `src/scoring.py` | `log_score`, `csl_score`, `compute_score_differential`, `build_comparison_table` (pairwise DM test matrix) |
| `src/utils.py` | `newey_west_se` (HAC SE), `dm_test` (two-sided, returns t-stat and p-value) |
| `data/fetch_data.py` | Massive API wrappers: `fetch_returns` (raw ticker) and `fetch_index_data` ('sp500'→SPY, 'nikkei'→EWJ) |
| `exp/exp_1_whole_density.py` | Experiment 1: KLIC log-score comparison, saves `exp1_<index>_*.csv` |
| `exp/exp_2_left_tail.py` | Experiment 2: CSL left-tail comparison (τ = −2.5 %), saves `exp2_<index>_*.csv` |
| `tests/` | pytest test suite covering models, scoring, and estimation |
| `results/` | Output CSV tables (created at runtime, tracked in git via `!results/*.csv` exception) |

---

## Key Design Decisions

### Returns and units
- All returns are in **decimal form** (e.g. −0.025 for −2.5 %)
- Log returns: `y_t = log(P_t / P_{t-1})`

### Variance initialisation
- `h0` is always set to the **sample variance of the current training window**

### Student-t parameterisation
- The scale parameter `s` satisfies `s² = h · (ν − 2)/ν` so that `Var(ε) = h`
- This means `h` directly equals the conditional variance

### GJR parameterisation
- Uses **alpha1** (symmetric ARCH term) and **alpha2** (additional leverage term
  for negative shocks) — **not** the `alpha / gamma` convention used in the
  `arch` library
- Recursion: `h_{t+1} = ω + α₁ ε² + α₂ ε² 𝟙(ε < 0) + β h`

### EGARCH parameterisation
- **No centering** of the absolute shock term (matches the paper's specification)
- `log h_{t+1} = ω + α |z_t| + γ z_t + φ log h_t`

### MLE optimisation
- Primary: **L-BFGS-B** with analytic bounds
- Fallback: **SLSQP** if L-BFGS-B fails
- **3 random starting points** to avoid local optima; best log-likelihood wins

### Bayesian MCMC
- **Independent-proposal Metropolis-Hastings**
- Proposal distribution: multivariate normal at MLE, covariance ∝ inverse
  observed Fisher information (from numerical Hessian)
- **5 000 burn-in draws** (discarded), **25 000 retained draws**
- Target acceptance rate: ~20–50 % (tuned via proposal scaling)
- Predictive density for day T+1: Monte Carlo average over posterior draws

### Newey-West lag selection
- `n_lags = floor(4 · (T/100)^(2/9))` where T = 600 out-of-sample forecasts

### CSL scoring rule
- For `y < τ`: `CSL = log f(y)`  (full log-density)
- For `y ≥ τ`: `CSL = log P(Y > τ)` (log survival probability at threshold)
- Threshold: **τ = −0.025** (−2.5 %)

---

## Data Sources

| Index | API Ticker | Notes |
|-------|-----------|-------|
| S&P 500 | `SPY` | SPDR S&P 500 ETF Trust; tracks the index closely since 1993 |
| Nikkei 225 | `EWJ` | iShares MSCI Japan ETF; used as a proxy (tracks broader Japanese market) |

API credentials: `MASSIVE_TOKEN` or `MASSIVE_API_KEY` environment variable.

---

## Running

```bash
# Set API key
export MASSIVE_TOKEN=your_key

# Install dependencies
pip install -r requirements.txt

# Fetch and verify data
python data/fetch_data.py

# Run experiments
python exp/exp_1_whole_density.py   # Experiment 1: whole density
python exp/exp_2_left_tail.py        # Experiment 2: left tail (CSL)

# Run test suite
pytest tests/ -v
```

---

## Expected Results

**Experiment 1** (whole density, KLIC log-score):
- Differences between frequentist and Bayesian are generally small and
  often statistically insignificant
- GJR and EGARCH tend to outperform GARCH(1,1) for the S&P 500 during the
  crisis period

**Experiment 2** (left tail, CSL):
- Bayesian methods significantly outperform frequentist MLE:
  DM t-statistics ≈ 5 for S&P 500 and Nikkei
- Asymmetric models (GJR, EGARCH) strongly outperform symmetric GARCH(1,1)
- Key finding: parameter uncertainty matters more in the tail than for the
  whole density

---

## Actual API Signatures (verified against src/)

| Function | Returns |
|----------|---------|
| `compute_score_differential(s_a, s_b)` | `(d, mean_d, t_stat, p_value)` — **4-tuple** |
| `build_comparison_table(scores_dict)` | `(mean_diff_df, tstat_df)` — **2-tuple** (no p-value DataFrame) |
| `csl_score(y, mu, h, nu, threshold)` | Uses `y <= threshold` (not strict `<`); at exactly threshold → log PDF |
| `dm_test(d)` | Raises `ZeroDivisionError` on a zero vector — never call with `d = s - s` |
| `frequentist_estimate(y, model)` | `{'params', 'loglik', 'h_path', 'h_forecast', 'success'}` — loglik can be **positive** for narrow densities |
| `bayesian_estimate(y, model, ...)` | `{'draws', 'acceptance_rate', 'posterior_mode', 'h_forecasts', 'param_names'}` |

EGARCH param names in `MODEL_PARAMS`: `['mu', 'omega', 'alpha', 'gamma', 'beta', 'nu']`.
The persistence parameter is **`'beta'`** (not `'phi'`).

Independence-chain MH acceptance rate can be low (e.g. ~1–2 %) for small training windows
(T ≈ 300) because the MVT proposal centred at MLE may not cover the posterior well.
Only assert `0 < ar < 1` in tests, not a specific range.

## estimation.py Implementation Details

### Hessian step-size formula
The key numerical challenge is that GARCH ω ≈ 1e-5 is tiny relative to
the default finite-difference step 1e-5.  Using a fixed step of 1e-5 for ω
evaluates the function 2–3× away from the mode, giving an inaccurate (and
often negative) Hessian eigenvalue.

The fix: `step_i = max(eps * |x_i|, 1e-8)` — proportional to the parameter
magnitude, floored at 1e-8 to prevent floating-point cancellation.

| Parameter | Typical mode | Step used |
|-----------|-------------|-----------|
| mu        | ≈ 0         | 1e-8      |
| omega     | ≈ 5e-6      | 1e-8      |
| alpha     | ≈ 0.10      | 1e-7      |
| beta      | ≈ 0.88      | 8.8e-6    |
| nu        | ≈ 8         | 8e-5      |

With these steps the GARCH/GJR Hessian is **positive-definite** without any
regularisation, giving proposal acceptance rates of ~45–55 %.

### Regularisation (negative Hessian eigenvalue fallback)
If any eigenvalue of H is ≤ 0 (can occur for EGARCH on misspecified data),
the ridge `|λ_min| × 1.01 + 1e-6` is added to the diagonal.  The resulting
proposal is wider in the badly-conditioned direction and acceptance can drop
to ~1–10 %; the chain is still correct (just slower mixing) and recovers on
well-specified data.

### MCMC chain initialisation
The chain always starts at the posterior mode, so the first acceptance is
guaranteed and burn-in converges faster.

### Acceptance-rate expectations by model (T=2520 window)
| Model  | Data type      | Typical rate |
|--------|----------------|-------------|
| GARCH  | GARCH-generated | ~50 %       |
| GJR    | GARCH-generated | ~45 %       |
| EGARCH | EGARCH-generated | ~50 %      |
| EGARCH | GARCH-generated (misspecified) | ~1–5 % |

---

## Common Pitfalls

1. **GJR alpha1/alpha2 naming** — do not confuse with `arch` library's
   `alpha/gamma` convention; the test suite in `test_models.py` verifies the
   asymmetry direction explicitly.

2. **EGARCH centering** — the paper does **not** subtract `E|z_t|` from the
   alpha term; adding centering changes results significantly.

3. **Return scaling** — always use decimal returns, not percentage returns.
   Mixing them breaks all parameter scales.

4. **h0 initialisation** — use the sample variance of the *current* window,
   not a global constant.

5. **Bayesian h_forecast** — compute the predictive variance for each MCMC
   draw and average them (or use the draw-specific variance path) to form the
   Bayesian predictive density correctly.
