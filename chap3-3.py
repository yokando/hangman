#
# Bayesian Analysis with Python written by Osvaldo Marthin
# Chapter 3-3
##################################################################

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import pymc3 as pm
import pandas as pd
plt.style.use('seaborn-darkgrid')
np.set_printoptions(precision=2)
pd.set_option('display.precision', 2)


tips = sns.load_dataset('tips')
tips.tail()
print("start!!!!!!!!!!!")


y = tips['tip'].values
idx = pd.Categorical(tips['day']).codes

#
'''
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
'''
if __name__ == '__main__':
#
  with pm.Model() as comparing_groups:
    means = pm.Normal('means', mu=0, sd=10, shape=len(set(idx)))
    sds = pm.HalfNormal('sds', sd=10, shape=len(set(idx)))
    y = pm.Normal('y', mu=means[idx], sd=sds[idx], observed=y)
    trace_cg = pm.sample(5000)
  chain_cg = trace_cg[100::]

  dist = dist = stats.norm()
  _, ax = plt.subplots(3, 2, figsize=(16, 12))
 
  comparisons = [(i,j) for i in range(4) for j in range(i+1, 4)]
  pos = [(k,l) for k in range(3) for l in (0, 1)]

  for (i,j), (k,l) in zip(comparisons, pos):
    means_diff = chain_cg['means'][:,i]-chain_cg['means'][:,j]
    d_cohen = (means_diff / np.sqrt((chain_cg['sds'][:,i]**2 + chain_cg['sds'][:,j]**2) / 2)).mean()
    ps = dist.cdf(d_cohen/(2**0.5))
#
# AttributeError: 'Line2D' object has no property 'kde_plot'
#    pm.plot_posterior(means_diff, ref_val=0, ax=ax[k,l], color='skyblue', kde_plot=True)
    pm.plot_posterior(means_diff, ref_val=0, ax=ax[k,l], color='skyblue')
    ax[k,l].plot(0, label="Cohen's d = {:.2f}\nProb sup = {:.2f}".format(d_cohen, ps), alpha=0)
    ax[k,l].set_xlabel('$\mu_{}-\mu_{}$'.format(i,j), fontsize=18)
    ax[k,l].legend(loc=0, fontsize=14)
  plt.savefig('img312.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()



