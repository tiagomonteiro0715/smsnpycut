# smsnpycut

A small Python port of the R package [`smsncut`](https://CRAN.R-project.org/package=smsncut): decision-theoretic optimal cutoff selection for a continuous diagnostic biomarker, modeled with **scale mixtures of skew-normal (SMSN)** distributions (Skew-Normal and Skew-*t*).

Given measurements from a healthy and a diseased group, it fits SMSN models by MLE and finds the cutoff that minimizes a weighted misclassification risk (accounting for disease prevalence and asymmetric false-positive/false-negative costs) — rather than just the symmetric Youden index. It also gives asymptotic confidence intervals for the cutoff, ROC/AUC, and Monte Carlo validation.

## Install

```bash
uv sync
```

## Quick start

```python
import numpy as np
import smsnpycut as sc

rng = np.random.default_rng(0)
healthy = sc.SkewNormal(xi=0.0, omega=1.0, alpha=2.0)
diseased = sc.SkewNormal(xi=2.5, omega=1.2, alpha=-1.0)

fit0 = sc.fit(healthy.rvs(300, random_state=rng), family="SN")
fit1 = sc.fit(diseased.rvs(300, random_state=rng), family="SN")

# 70% prevalence of healthy, false negatives 3x costlier than false positives
c_opt = sc.optimal_cutoff(fit0.dist, fit1.dist, pi0=0.7, pi1=0.3, lam0=1.0, lam1=3.0)
ci = sc.confidence_interval(c_opt, sc.variance(c_opt, fit0, fit1, 0.7, 0.3, 1.0, 3.0))
print(c_opt, ci, sc.auc(fit0.dist, fit1.dist))
```

See [main.py](main.py) for a full worked example.

## API

| Module | Purpose |
|---|---|
| `distributions` | `SkewNormal`, `SkewT` — pdf/cdf/rvs |
| `fit` | `fit(x, family)` — MLE via BFGS + Hessian |
| `cutoff` | `optimal_cutoff`, `youden_cutoff`, `admissible_interval`, `boundary_ok` |
| `inference` | `variance`, `confidence_interval`, `identifiability` |
| `roc` | `roc_curve`, `auc` |
| `simulate` | `mc_validate` — Monte Carlo check of the cutoff/CI recipe |

## Citation

See [CITATION.bib](CITATION.bib) for the original R package and its companion paper (de Paula, Mouriño & Dias Domingues, 2026).
