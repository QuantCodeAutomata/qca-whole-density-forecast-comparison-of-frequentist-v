# GARCH Density Forecast Comparison: Frequentist vs Bayesian

A Python reproduction of the paper comparing **frequentist MLE** and **Bayesian
MCMC** estimation of GARCH-family volatility models for density forecasting on
equity index returns.

---

## Overview

The study evaluates one-step-ahead predictive densities produced by three
GARCH-family models estimated two ways:

| Model | Frequentist | Bayesian |
|-------|-------------|----------|
| GARCH(1,1) | MLE (L-BFGS-B) | Metropolis-Hastings MCMC |
| GJR(1,1)   | MLE (L-BFGS-B) | Metropolis-Hastings MCMC |
| EGARCH(1,1)| MLE (L-BFGS-B) | Metropolis-Hastings MCMC |

All innovations follow a **Student-t distribution**.  Two scoring rules are used:

- **Experiment 1** – Whole-density KLIC log-score (Table 1 of the paper)
- **Experiment 2** – Censored Likelihood Score (CSL) for the left tail below
  τ = −2.5 % (Table 2 of the paper)

### Data

| Index | Proxy ticker | Source |
|-------|-------------|--------|
| S&P 500 | `SPY` (SPDR S&P 500 ETF) | Massive API |
| Nikkei 225 | `EWJ` (iShares MSCI Japan ETF) | Massive API |

- **Full sample**: ~1997-01-01 – 2010-04-20 (daily log returns)
- **Estimation window**: rolling 10-year (≈ 2 520 trading days)
- **Out-of-sample period**: 2008-01-01 – 2010-04-19 (**600 forecasts**)

---

## Repository Structure

```
.
├── data/
│   ├── __init__.py
│   └── fetch_data.py          # Massive API data fetching (SPY, EWJ)
├── exp/
│   ├── __init__.py
│   ├── exp_1_whole_density.py # Experiment 1: KLIC log-score comparison
│   └── exp_2_left_tail.py     # Experiment 2: CSL left-tail comparison
├── src/
│   ├── __init__.py
│   ├── models.py              # Numba-JIT variance recursions & log-likelihoods
│   ├── estimation.py          # Frequentist MLE & Bayesian MCMC estimators
│   ├── forecasting.py         # Rolling-window forecast pipeline
│   ├── scoring.py             # Log-score, CSL, DM pairwise comparison tables
│   └── utils.py               # Newey-West SE, DM test
├── tests/
│   ├── __init__.py
│   ├── test_models.py         # Unit tests for variance recursions & likelihoods
│   ├── test_scoring.py        # Unit tests for scoring rules & DM test
│   └── test_estimation.py     # Integration tests for MLE & MCMC estimators
├── results/                   # Output CSV tables (created at runtime)
├── AGENTS.md                  # Repository knowledge base for AI agents
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd <repo-directory>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up the Massive API key

The data fetching module uses the [Massive](https://massive.io) REST API.
Export your key before running any data or experiment scripts:

```bash
export MASSIVE_TOKEN=your_api_key_here
```

Alternatively, the code also checks `MASSIVE_API_KEY`.

---

## Usage

### Fetch data

```bash
python data/fetch_data.py
```

This prints summary statistics for both return series and confirms the API
connection is working.

### Run both experiments (recommended)

```bash
# Production settings (n_burn=5000, n_draws=25000) — slow but accurate
python run_all.py --workers 4

# Fast mode (n_burn=500, n_draws=2000) — quick validation run
python run_all.py --fast --workers 4
```

Results are cached in `data/*.pkl` and output tables written to `results/`.

### Run Experiment 1 — Whole-density comparison (standalone)

```bash
python exp/exp_1_whole_density.py
```

Outputs per-forecast log-scores and pairwise Diebold–Mariano comparison
tables to `results/exp1_<index>_*.csv`.

### Run Experiment 2 — Left-tail comparison (standalone)

```bash
python exp/exp_2_left_tail.py
```

Outputs CSL scores and comparison tables to `results/exp2_<index>_*.csv`.

### Run the test suite

```bash
pytest tests/
```

---

## Models and Methodology

### Variance recursions (Student-t innovations)

**GARCH(1,1)**

```
h_{t+1} = ω + α ε_t² + β h_t
```

**GJR(1,1)**  *(leverage effect via indicator)*

```
h_{t+1} = ω + α₁ ε_t² + α₂ ε_t² 𝟙(ε_t < 0) + β h_t
```

**EGARCH(1,1)**  *(log-linear; no E|z_t| centering term per paper spec)*

```
log h_{t+1} = ω + α |z_t| + γ z_t + β log h_t,   z_t = ε_t / √h_t
```

### Scoring rules

**Log-score** (whole density, Experiment 1):

```
LS_t = log f(y_{t+1} | Ω_t)
```

**Censored Likelihood Score** (left tail, Experiment 2):

```
CSL_t = log f(y_{t+1}) · 𝟙(y_{t+1} < τ)  +  log P(Y > τ) · 𝟙(y_{t+1} ≥ τ)
```

where τ = −0.025 (−2.5 %).

### Bayesian estimation

A **random-walk Metropolis-Hastings (RWMH)** chain on unconstrained parameter
space is used:

- Priors: flat on model coefficients; translated-exponential `p(ν) = 0.05·exp[−0.05·(ν−2)]`
- Proposal covariance: 2.38²/p times the inverse numerical Hessian of the
  log-posterior at the MLE (Roberts & Rosenthal optimal scaling)
- Burn-in: 5 000 draws (discarded); retained: 25 000 draws  
  *(fast-mode defaults: 500 burn-in / 2 000 retained)*
- Predictive density: Monte Carlo mixture over posterior draws

Data is fetched via the **Massive REST API** with an automatic fallback to
**Yahoo Finance** (`yfinance`) when `MASSIVE_TOKEN` is absent or invalid.

### Statistical testing

Pairwise **Diebold–Mariano** tests with **Newey-West** HAC standard errors:

```
n_lags = floor(4 · (T/100)^(2/9)),  T = 600
```

---

## Expected Results

**Experiment 1 (whole density):** Differences between frequentist and Bayesian
are small and often statistically insignificant — both approaches capture the
conditional mean density well.

**Experiment 2 (left tail):** Bayesian methods significantly outperform
frequentist MLE (DM t-statistics ≈ 5 for S&P 500; similar for Nikkei).
Asymmetric models (GJR, EGARCH) outperform the symmetric GARCH(1,1), especially
in the left tail during the 2008–2010 crisis period.

---

## Citation

If you use this code, please cite the original paper this project reproduces
(details in the paper PDF included with the repository).
