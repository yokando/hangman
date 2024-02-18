#
# coin flip
##################################################################

import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import seaborn as sns
# import pymc3 as pm
plt.style.use('seaborn-darkgrid')


def posterior_grid(grid_points=100, heads=6, tosses=9):
  grid = np.linspace(0, 1, grid_points)
  print(grid)
  prior = np.repeat(5, grid_points)
  print(prior)
  likelihood = stats.binom.pmf(heads, tosses, grid)
  print(likelihood)
  unstd_posterior = likelihood * prior
  posterior = unstd_posterior / unstd_posterior.sum()
  return grid, posterior

points = 15
h, n = 1, 4
grid, posterior = posterior_grid(points, h, n)
plt.plot(grid, posterior, 'o-', label='heads = {}\ntosses = {}'.format(h, n))
plt.xlabel(r'$\theta$', fontsize=14)
plt.legend(loc=0, fontsize=14)
plt.savefig('coin1.png')