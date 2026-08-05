"""Maximum-likelihood fitting of SMSN distributions."""
from dataclasses import dataclass

import numdifftools as nd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import skew

from .distributions import FAMILIES


@dataclass
class FitResult:
    dist: object          # fitted SkewNormal/SkewT instance
    theta: np.ndarray     # unconstrained MLE vector
    hess: np.ndarray      # observed Fisher information (Hessian of neg log-lik)
    loglik: float
    n: int
    family: str


def _neg_loglik(theta, x, Family):
    return -np.sum(Family.unpack(theta).logpdf(x))


def fit(x, family="SN", theta0=None):
    x = np.asarray(x, dtype=float)
    Family = FAMILIES[family]
    if theta0 is None:
        theta0 = np.zeros(Family.n_params)
        theta0[0], theta0[1] = x.mean(), np.log(x.std())
        theta0[2] = 1.0 if skew(x) >= 0 else -1.0  # alpha=0 is a gradient saddle when xi0 = mean(x)
        if family == "ST":
            theta0[3] = np.log(6.0)  # nu0 = 8

    res = minimize(_neg_loglik, theta0, args=(x, Family), method="BFGS")
    hess = nd.Hessian(_neg_loglik)(res.x, x, Family)
    return FitResult(Family.unpack(res.x), res.x, hess, -res.fun, len(x), family)
