#
# Bayesian Analysis with Python written by Osvaldo Marthin
# Chapter 3-4
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


N_samples = [30, 30, 30]
G_samples = [18, 18, 18]
print("start!!!!!!!!!!!")
group_idx = np.repeat(np.arange(len(N_samples)), N_samples)
data = []
for i in range(0, len(N_samples)):
  data.extend(np.repeat([1, 0], [G_samples[i], N_samples[i]- G_samples[i]]))

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
    with pm.Model() as model_h:
      alpha = pm.HalfCauchy('alpha', beta=10)
      beta = pm.HalfCauchy('beta', beta=10)
      theta = pm.Beta('theta', alpha, beta, shape=len(N_samples))
      y = pm.Bernoulli('y', p=theta[group_idx], observed=data)
      trace_h = pm.sample(2000)
    chain_h = trace_h[200:]
    pm.traceplot(chain_h)
    plt.savefig('img314.png', dpi=300, figsize=(5.5, 5.5))
    print("table_suishitu")
    print(pm.summary(chain_h))

    plt.figure()


    x = np.linspace(0, 1, 100)
    for i in np.random.randint(0, len(chain_h), size=100):
      pdf = stats.beta(chain_h['alpha'][i], chain_h['beta'][i]).pdf(x)
      plt.plot(x, pdf, 'g', alpha=0.05)

    dist = stats.beta(chain_h['alpha'].mean(), chain_h['beta'].mean())
    pdf = dist.pdf(x)
    mode = x[np.argmax(pdf)]
    mean = dist.moment(1)
    plt.plot(x, pdf, label='mode = {:.2f}\nmean = {:.2f}'.format(mode, mean))

    plt.legend(fontsize=14)
    plt.xlabel(r'$\theta_{prior}$', fontsize=16)
    plt.savefig('img315.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

