"""SMSN distributions: Skew-Normal (SN) and Skew-t (ST), Azzalini parametrization."""
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy import stats
from scipy.integrate import cumulative_trapezoid


@dataclass
class SkewNormal:
    xi: float
    omega: float
    alpha: float

    n_params = 3

    def _rv(self):
        return stats.skewnorm(self.alpha, loc=self.xi, scale=self.omega)

    def pdf(self, x):
        return self._rv().pdf(x)

    def logpdf(self, x):
        return self._rv().logpdf(x)

    def cdf(self, x):
        return self._rv().cdf(x)

    def rvs(self, size=1, random_state=None):
        return self._rv().rvs(size=size, random_state=random_state)

    def pack(self):
        return np.array([self.xi, np.log(self.omega), self.alpha])

    @classmethod
    def unpack(cls, theta):
        xi, log_omega, alpha = theta
        return cls(xi, np.exp(log_omega), alpha)


@dataclass
class SkewT:
    xi: float
    omega: float
    alpha: float
    nu: float

    n_params = 4

    def logpdf(self, x):
        z = (np.asarray(x, dtype=float) - self.xi) / self.omega
        tau = np.sqrt((self.nu + 1) / (self.nu + z**2))
        return (np.log(2) - np.log(self.omega)
                + stats.t.logpdf(z, df=self.nu)
                + stats.t.logcdf(self.alpha * z * tau, df=self.nu + 1))

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    @cached_property
    def _cdf_grid(self):
        """One dense integration grid per instance, reused by every `cdf` call
        (a fresh `quad` per point is too slow inside an optimization loop)."""
        grid = np.linspace(self.xi - 50 * self.omega, self.xi + 50 * self.omega, 5000)
        return grid, cumulative_trapezoid(self.pdf(grid), grid, initial=0.0)

    def cdf(self, x):
        grid, cum = self._cdf_grid
        return np.interp(x, grid, cum)

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        delta = self.alpha / np.sqrt(1 + self.alpha**2)
        u0, u1 = rng.standard_normal(size), rng.standard_normal(size)
        z = delta * np.abs(u0) + np.sqrt(1 - delta**2) * u1
        v = rng.chisquare(self.nu, size)
        return self.xi + self.omega * z / np.sqrt(v / self.nu)

    def pack(self):
        return np.array([self.xi, np.log(self.omega), self.alpha, np.log(self.nu - 2)])

    @classmethod
    def unpack(cls, theta):
        xi, log_omega, alpha, log_nu2 = theta
        return cls(xi, np.exp(log_omega), alpha, np.exp(log_nu2) + 2)


FAMILIES = {"SN": SkewNormal, "ST": SkewT}
