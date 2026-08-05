"""Parametric ROC curve and AUC induced by two fitted SMSN distributions."""
import numpy as np


def _support(dist0, dist1, width=8, n=4000):
    s = width * max(dist0.omega, dist1.omega)
    lo, hi = min(dist0.xi, dist1.xi) - s, max(dist0.xi, dist1.xi) + s
    return np.linspace(lo, hi, n)


def roc_curve(dist0, dist1, n=200):
    c = _support(dist0, dist1, n=n)
    fpr, tpr = 1 - dist0.cdf(c), 1 - dist1.cdf(c)
    order = np.argsort(fpr)
    return fpr[order], tpr[order]


def auc(dist0, dist1):
    """AUC = P(X1 > X0) = integral of F0(x) * f1(x) dx."""
    x = _support(dist0, dist1, width=15)
    return np.trapezoid(dist0.cdf(x) * dist1.pdf(x), x)
