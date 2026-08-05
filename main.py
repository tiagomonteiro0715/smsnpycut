import numpy as np

import smsnpycut as sc


def main():
    rng = np.random.default_rng(0)

    # simulate a biomarker: healthy (0) vs diseased (1) populations
    healthy = sc.SkewNormal(xi=0.0, omega=1.0, alpha=2.0)
    diseased = sc.SkewNormal(xi=2.5, omega=1.2, alpha=-1.0)
    x0 = healthy.rvs(300, random_state=rng)
    x1 = diseased.rvs(300, random_state=rng)

    # fit SMSN models to each group
    fit0 = sc.fit(x0, family="SN")
    fit1 = sc.fit(x1, family="SN")
    print("fitted healthy params (xi, log_omega, alpha):", fit0.theta)
    print("fitted diseased params (xi, log_omega, alpha):", fit1.theta)

    # decision-theoretic cutoff: 70% prevalence of healthy, false negatives cost 3x
    pi0, pi1, lam0, lam1 = 0.7, 0.3, 1.0, 3.0
    c_opt = sc.optimal_cutoff(fit0.dist, fit1.dist, pi0, pi1, lam0, lam1)
    c_you = sc.youden_cutoff(fit0.dist, fit1.dist)
    print(f"optimal cutoff: {c_opt:.3f}   Youden cutoff: {c_you:.3f}")

    # asymptotic uncertainty around the optimal cutoff
    var = sc.variance(c_opt, fit0, fit1, pi0, pi1, lam0, lam1)
    ci = sc.confidence_interval(c_opt, var)
    print(f"95% CI for optimal cutoff: ({ci[0]:.3f}, {ci[1]:.3f})")

    # discriminative power of the biomarker
    print(f"AUC: {sc.auc(fit0.dist, fit1.dist):.3f}")

    # Monte Carlo check: does the cutoff/CI recipe behave as expected?
    mc = sc.mc_validate(healthy, diseased, n0=300, n1=300, B=200,
                         pi0=pi0, pi1=pi1, lam0=lam0, lam1=lam1, seed=1)
    print("Monte Carlo validation:", mc)


if __name__ == "__main__":
    main()
