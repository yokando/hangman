#
# Bayesian Analysis with Python written by Osvaldo Marthin
# Chapter 3-2
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
print("tips data - head")
print(tips.head())
print("tips data - tail")
print(tips.tail())

sns.violinplot(x='day', y='tip', data=tips)
plt.savefig('img310.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

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
#
# NameError: name 'chain_cg' is not defined
    chain_cg = trace_cg[100::]
    pm.traceplot(chain_cg)
    plt.savefig('img311.png', dpi=300, figsize=(5.5, 5.5))
    print("table_tips")
    print(pm.summary(chain_cg))

plt.figure()