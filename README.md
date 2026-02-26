# Bayesian Linear Regression — Evidence Model Selection

Small project for Bayesian model selection using **log-evidence** (marginal likelihood) in Bayesian Linear Regression.

## What it does
- **Section 2.1:** compares polynomial degrees (2–10) by log-evidence on 5 synthetic noisy functions and plots best vs worst fits.
- **Section 2.2:** finds the best **noise variance** by maximizing log-evidence on temperature data (Nov 16, 2024).

## Run
Requires `numpy`, `matplotlib`, and `ex3_utils.py` (provides `BayesianLinearRegression` + `polynomial_basis_functions`).

```bash
python ex3.py
