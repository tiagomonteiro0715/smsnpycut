"""Asymptotic inference for the optimal cutoff: delta-method variance, CIs, identifiability."""
import numdifftools as nd
import numpy as np

from .cutoff import _risk_slope


def _slope_of_theta(c, fit0, fit1, pi0, pi1, lam0, lam1):
    def g(t0, t1):
        d0, d1 = type(fit0.dist).unpack(t0), type(fit1.dist).unpack(t1)
        return _risk_slope(c, d0, d1, pi0, pi1, lam0, lam1)
    return g


def variance(c_hat, fit0, fit1, pi0=0.5, pi1=0.5, lam0=1.0, lam1=1.0):
    """Delta-method variance of the plug-in cutoff estimator.

    fit0.hess / fit1.hess are Hessians of the per-group negative log-likelihood,
    so their inverses already estimate Cov(theta_hat) (no extra division by n).
    """
    g = _slope_of_theta(c_hat, fit0, fit1, pi0, pi1, lam0, lam1)
    dg_dc = nd.Derivative(lambda c: _risk_slope(c, fit0.dist, fit1.dist, pi0, pi1, lam0, lam1))(c_hat)
    dg_dt0 = nd.Gradient(lambda t: g(t, fit1.theta))(fit0.theta)
    dg_dt1 = nd.Gradient(lambda t: g(fit0.theta, t))(fit1.theta)

    dc_dt0 = -dg_dt0 / dg_dc
    dc_dt1 = -dg_dt1 / dg_dc
    cov0, cov1 = np.linalg.inv(fit0.hess), np.linalg.inv(fit1.hess)
    return dc_dt0 @ cov0 @ dc_dt0 + dc_dt1 @ cov1 @ dc_dt1


def confidence_interval(c_hat, var, z=1.96):
    se = np.sqrt(var)
    return c_hat - z * se, c_hat + z * se


def identifiability(c_hat, fit0, fit1, pi0=0.5, pi1=0.5, lam0=1.0, lam1=1.0, tol=1e-8):
    """True iff the risk slope crosses zero non-degenerately at c_hat (local identifiability)."""
    slope = nd.Derivative(lambda c: _risk_slope(c, fit0.dist, fit1.dist, pi0, pi1, lam0, lam1))(c_hat)
    return abs(slope) > tol
