"""Monte Carlo validation of the plug-in cutoff and its confidence interval."""
import numpy as np

from .cutoff import optimal_cutoff
from .fit import fit
from .inference import confidence_interval, variance


def mc_validate(true0, true1, n0, n1, B=500, pi0=0.5, pi1=0.5, lam0=1.0, lam1=1.0,
                 family0="SN", family1="SN", seed=None, z=1.96):
    rng = np.random.default_rng(seed)
    c_true = optimal_cutoff(true0, true1, pi0, pi1, lam0, lam1)

    c_hats = np.full(B, np.nan)
    covered = np.zeros(B, dtype=bool)
    for b in range(B):
        x0, x1 = true0.rvs(n0, random_state=rng), true1.rvs(n1, random_state=rng)
        try:
            f0, f1 = fit(x0, family0), fit(x1, family1)
            c = optimal_cutoff(f0.dist, f1.dist, pi0, pi1, lam0, lam1)
            lo, hi = confidence_interval(c, variance(c, f0, f1, pi0, pi1, lam0, lam1), z)
            c_hats[b], covered[b] = c, lo <= c_true <= hi
        except Exception:
            pass

    return {
        "c_true": c_true,
        "success_rate": np.mean(~np.isnan(c_hats)),
        "bias": np.nanmean(c_hats) - c_true,
        "rmse": np.sqrt(np.nanmean((c_hats - c_true) ** 2)),
        "coverage": np.nanmean(covered),
    }
