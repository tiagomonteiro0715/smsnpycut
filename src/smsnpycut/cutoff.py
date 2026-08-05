"""Decision-theoretic and Youden cutoff selection.

Risk(c) = pi0*lambda0*P(X0 > c) + pi1*lambda1*P(X1 < c) is minimized directly
(grid search + bounded refinement) rather than root-finding on its derivative,
since skew densities can cross more than once and a root of the derivative may
be a local max rather than the global minimizer.
"""
import numpy as np
from scipy.optimize import minimize_scalar


def admissible_interval(dist0, dist1, width=10):
    s = width * max(dist0.omega, dist1.omega)
    return min(dist0.xi, dist1.xi) - s, max(dist0.xi, dist1.xi) + s


def _risk(c, dist0, dist1, pi0, pi1, lam0, lam1):
    return pi0 * lam0 * (1 - dist0.cdf(c)) + pi1 * lam1 * dist1.cdf(c)


def _risk_slope(c, dist0, dist1, pi0, pi1, lam0, lam1):
    """Derivative of Risk(c); its root is the first-order optimality condition used for inference."""
    return pi1 * lam1 * dist1.pdf(c) - pi0 * lam0 * dist0.pdf(c)


def optimal_cutoff(dist0, dist1, pi0=0.5, pi1=0.5, lam0=1.0, lam1=1.0, interval=None, n_grid=200):
    lo, hi = interval or admissible_interval(dist0, dist1)
    risk = lambda c: _risk(c, dist0, dist1, pi0, pi1, lam0, lam1)
    grid = np.linspace(lo, hi, n_grid)
    i = int(np.argmin([risk(c) for c in grid]))
    lo2, hi2 = grid[max(i - 1, 0)], grid[min(i + 1, n_grid - 1)]
    return minimize_scalar(risk, bounds=(lo2, hi2), method="bounded").x


def youden_cutoff(dist0, dist1, interval=None):
    """Cutoff maximizing sensitivity + specificity - 1 (equal-weight special case)."""
    return optimal_cutoff(dist0, dist1, 1.0, 1.0, 1.0, 1.0, interval)


def boundary_ok(dist0, dist1, pi0=0.5, pi1=0.5, lam0=1.0, lam1=1.0, interval=None):
    """True iff the optimal cutoff is interior, i.e. beats classifying everyone the same way."""
    lo, hi = interval or admissible_interval(dist0, dist1)
    risk = lambda c: _risk(c, dist0, dist1, pi0, pi1, lam0, lam1)
    c = optimal_cutoff(dist0, dist1, pi0, pi1, lam0, lam1, (lo, hi))
    return risk(c) < min(risk(lo), risk(hi))
