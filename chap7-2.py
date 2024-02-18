#
# Bayesian Analysis with Python written by Osvaldo Marthin
# Chapter 7-2
##################################################################

import pymc3 as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
#
import arviz as az

plt.style.use('seaborn-darkgrid')
np.set_printoptions(precision=2)

#https://stats.idre.ucla.edu/stat/data/fish.csv
fish_data = pd.read_csv('fish.csv')
fish_data.head()

with pm.Model() as ZIP_reg:
  psi = pm.Beta('psi', 1, 1)
  
  alpha = pm.Normal('alpha', 0, 10)
  beta = pm.Normal('beta', 0, 10, shape=2)
  lam = pm.math.exp(alpha + beta[0] * fish_data['child'] + beta[1] * fish_data['camper'])
  
  y = pm.ZeroInflatedPoisson('y', psi, lam, observed=fish_data['count'])
#
# ValueError: Unused step method arguments: {'njobs'}
# trace_ZIP_reg = pm.sample(2000, njobs=1
  trace_ZIP_reg = pm.sample(2000, chains=1)

chain_ZIP_reg = trace_ZIP_reg[100:]
pm.traceplot(chain_ZIP_reg);
plt.savefig('img710.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

children =  [0, 1, 2, 3, 4]
fish_count_pred_0 = []
fish_count_pred_1 = []
thin = 5
for n in children:
  without_camper = chain_ZIP_reg['alpha'][::thin] + chain_ZIP_reg['beta'][:,0][::thin] * n
  with_camper = without_camper + chain_ZIP_reg['beta'][:,1][::thin]
  fish_count_pred_0.append(np.exp(without_camper))
  fish_count_pred_1.append(np.exp(with_camper))

plt.plot(children, fish_count_pred_0, 'bo', alpha=0.01)
plt.plot(children, fish_count_pred_1, 'ro', alpha=0.01)

plt.xticks(children);
plt.xlabel('Number of children', fontsize=14)
plt.ylabel('Fish caught', fontsize=14)
plt.plot([], 'bo', label='without camper')
plt.plot([], 'ro', label='with camper')
plt.legend(fontsize=14)
plt.savefig('img711.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

iris = sns.load_dataset("iris")
df = iris.query("species == ('setosa', 'versicolor')")
y_0 = pd.Categorical(df['species']).codes
x_n = 'sepal_length' 
x_0 = df[x_n].values
y_0 = np.concatenate((y_0, np.ones(6)))
x_0 = np.concatenate((x_0, [4.2, 4.5, 4.0, 4.3, 4.2, 4.4]))
x_0_m = x_0 - x_0.mean()
plt.plot(x_0, y_0, 'o', color='k')
plt.savefig('img712.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

with pm.Model() as model_rlg:
  alpha_tmp = pm.Normal('alpha_tmp', mu=0, sd=100)
  beta = pm.Normal('beta', mu=0, sd=10)
  mu = alpha_tmp + beta * x_0_m
  theta = pm.Deterministic('theta', 1 / (1 + pm.math.exp(-mu)))
  
  pi = pm.Beta('pi', 1, 1)
  p = pi * 0.5 + (1 - pi) * theta
  
  alpha = pm.Deterministic('alpha', alpha_tmp - beta * x_0.mean())
  bd = pm.Deterministic('bd', -alpha/beta)
  
  yl = pm.Bernoulli('yl', p=p, observed=y_0)
#
# ValueError: Unused step method arguments: {'njobs'}
#  trace_rlg = pm.sample(2000, njobs=1)
  trace_rlg = pm.sample(2000, chains=1)

varnames = ['alpha', 'beta', 'bd', 'pi']
pm.traceplot(trace_rlg, varnames)
plt.savefig('img713.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

print("table7-2")
print(pm.summary(trace_rlg, varnames))

theta = trace_rlg['theta'].mean(axis=0)
idx = np.argsort(x_0)
plt.plot(x_0[idx], theta[idx], color='b', lw=3);
plt.axvline(trace_rlg['bd'].mean(), ymax=1, color='r')
#
# AttributeError: module 'pymc3' has no attribute 'hpd'
# bd_hpd = pm.hpd(trace_rlg['bd'])
bd_hpd = az.hdi(trace_rlg['bd'])

plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='r', alpha=0.5)

plt.plot(x_0, y_0, 'o', color='k')
#
# AttributeError: module 'pymc3' has no attribute 'hpd'
# theta_hpd = pm.hpd(trace_rlg['theta'])[idx]
theta_hpd = az.hdi(trace_rlg['theta'])[idx]
plt.fill_between(x_0[idx], theta_hpd[:,0], theta_hpd[:,1], color='b', alpha=0.5)

plt.xlabel(x_n, fontsize=16)
plt.ylabel('$\\theta$', rotation=0, fontsize=16)
plt.savefig('img714.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

