import itertools
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")   # save to file only, no windows
import matplotlib.pyplot as plt
from scipy.stats import norm

# same model as Bayesian_first_test.py:
# x_i ~ N(mu, sigma^2), sigma known, prior mu ~ N(mu0, tau0^2)

# parameter sweep
N_VALUES     = [1, 10, 100, 1000]
MU_VALUES    = [0.0, 2.0, 5.0]
SIGMA_VALUES = [0.5, 1.0, 2.0]

# prior (fixed for the whole sweep)
mu0, tau0 = 0.0, 1.0

OUT_DIR = os.path.join("Figures", "sweep")


def fmt(v):
    # 0.5 -> "0p5", so filenames have no extra dots
    return f"{v:.1f}".replace(".", "p").replace("-", "m")


def make_figure(n, mu_true, sigma, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(mu_true, sigma, n)

    # closed-form posterior, used to place a fine enough grid
    post_var = 1 / (n / sigma**2 + 1 / tau0**2)
    post_mean = post_var * (x.sum() / sigma**2 + mu0 / tau0**2)
    post_std = np.sqrt(post_var)
    lik_mean, lik_std = x.mean(), sigma / np.sqrt(n)

    lo = min(mu0 - 4 * tau0, lik_mean - 6 * lik_std, post_mean - 6 * post_std)
    hi = max(mu0 + 4 * tau0, lik_mean + 6 * lik_std, post_mean + 6 * post_std)
    mu_grid = np.linspace(lo, hi, 20001)
    dx = mu_grid[1] - mu_grid[0]

    log_prior = norm.logpdf(mu_grid, mu0, tau0)
    # sum_i log N(x_i | mu, sigma^2), written with sufficient statistics
    log_lik = (-0.5 * n * np.log(2 * np.pi * sigma**2)
               - (np.sum(x**2) - 2 * mu_grid * x.sum() + n * mu_grid**2) / (2 * sigma**2))
    log_post = log_prior + log_lik

    def to_density(logp):
        p = np.exp(logp - logp.max())
        return p / (p.sum() * dx)

    prior = to_density(log_prior)
    lik   = to_density(log_lik)
    post  = to_density(log_post)

    mu_map = mu_grid[np.argmax(log_post)]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4))

    ax1.hist(x, bins=min(40, max(5, n // 5)), density=True, alpha=0.6)
    ax1.plot(x, np.zeros_like(x), "|", color="k", ms=12, alpha=0.5)   # rug: shows each point for small n
    ax1.axvline(mu_true, color="g", ls="--", label=f"true μ = {mu_true}")
    ax1.set_title(f"Data x  (n = {n}, σ = {sigma})")
    ax1.legend()

    # full view: prior, likelihood and posterior together
    ax2.plot(mu_grid, prior, label=f"prior N({mu0}, {tau0}²)")
    ax2.plot(mu_grid, lik, "--", label="likelihood (normalised)")
    ax2.plot(mu_grid, post, label="posterior")
    ax2.axvline(mu_true, color="g", ls="--", label=f"true μ = {mu_true}")
    ax2.set_xlim(min(mu0 - 3 * tau0, lik_mean - 4 * lik_std),
                 max(mu0 + 3 * tau0, lik_mean + 4 * lik_std))
    ax2.set_xlabel("μ")
    ax2.legend(fontsize=8)
    ax2.set_title("Prior vs likelihood vs posterior")

    # zoom on the posterior, or for large n it is just a spike
    ax3.plot(mu_grid, lik, "--", color="C1", label="likelihood (normalised)")
    ax3.plot(mu_grid, post, color="C2", label="posterior")
    ax3.axvline(mu_true, color="g", ls="--", label=f"true μ = {mu_true}")
    ax3.axvline(mu_map, color="k", ls=":", label=f"MAP grid = {mu_map:.3f}")
    ax3.axvline(post_mean, color="r", ls=":", label=f"MAP formula = {post_mean:.3f}")
    wide = max(lik_std, post_std)
    ax3.set_xlim(min(lik_mean, post_mean) - 4 * wide, max(lik_mean, post_mean) + 4 * wide)
    ax3.set_xlabel("μ")
    ax3.legend(fontsize=8)
    ax3.set_title(f"Zoom on posterior  (posterior std = {post_std:.3g})")

    fig.suptitle(f"n = {n},  true μ = {mu_true},  σ = {sigma}")
    fig.tight_layout()

    fname = f"posterior_n{n:04d}_mu{fmt(mu_true)}_sigma{fmt(sigma)}.png"
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=120)
    plt.close(fig)
    return fname, mu_map, post_mean


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for n, mu_true, sigma in itertools.product(N_VALUES, MU_VALUES, SIGMA_VALUES):
        fname, mu_map, mu_formula = make_figure(n, mu_true, sigma)
        print(f"{fname}   MAP grid {mu_map:.4f}   formula {mu_formula:.4f}")
