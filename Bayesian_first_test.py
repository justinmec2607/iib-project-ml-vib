import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

rng = np.random.default_rng(0)


# random data: mean = 0, std = 1  
mu_true, sigma, n = 3.0, 1.0, 10
x = rng.normal(mu_true, sigma, n)


# grid of candidate mu values
mu_grid = np.linspace(-2, 8, 5000)

# prior
mu0,tau0 = 0.0, 1.0
log_prior = norm.logpdf(mu_grid, mu0, tau0)

#log likelihood
log_lik = np.array([np.sum(norm.logpdf(x, mu, sigma)) for mu in mu_grid])

# posterior
log_post = log_prior + log_lik

dx = mu_grid[1] - mu_grid[0]

def to_density(logp):
    p = np.exp(logp - logp.max())   # subtract max: avoids exp(huge negative) = 0
    return p / (p.sum() * dx)       # scale so area = 1

prior = to_density(log_prior)
lik   = to_density(log_lik)
post  = to_density(log_post)

mu_map = mu_grid[np.argmax(log_post)]
mu_formula = (tau0**2 * x.sum() + sigma**2 * mu0) / (n * tau0**2 + sigma**2)
tau_n = 1 / np.sqrt(1/tau0**2+n/sigma**2)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.hist(x, bins=40, density=True, alpha=0.6)
ax1.set_title("Data x")


ax2.plot(mu_grid, norm.pdf(mu_grid, mu_formula, tau_n), "k:", label="analytic posterior")


ax2.plot(mu_grid, prior, label="prior")
ax2.plot(mu_grid, lik, "--", label="likelihood (normalised)")
ax2.plot(mu_grid, post, label="posterior")
ax2.axvline(mu_map, color="k", ls=":", label=f"MAP grid = {mu_map:.3f}")
ax2.axvline(mu_formula, color="r", ls=":", label=f"MAP formula = {mu_formula:.3f}")
ax2.legend()
ax2.set_title("Prior vs likelihood vs posterior")

plt.show()



