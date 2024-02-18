#
# Bayesian Analysis with Python written by Osvaldo Marthin
# Chapter 4-2
##############################################

import pymc3 as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
#
import arviz as az
#
plt.style.use('seaborn-darkgrid')
np.set_printoptions(precision=2)
pd.set_option('display.precision', 2)


N = 20
M = 8
idx = np.repeat(range(M-1), N)
idx = np.append(idx, 7)
np.random.seed(314)

alpha_real = np.random.normal(2.5, 0.5, size=M)
beta_real = np.random.beta(6, 1, size=M)
eps_real = np.random.normal(0, 0.5, size=len(idx))

y_m = np.zeros(len(idx))
x_m = np.random.normal(10, 1, len(idx))
y_m = alpha_real[idx] + beta_real[idx] * x_m  + eps_real

plt.figure(figsize=(16,8))
j, k = 0, N
for i in range(M):
  plt.subplot(2,4,i+1)
  plt.scatter(x_m[j:k], y_m[j:k])
  plt.xlabel('$x_{}$'.format(i), fontsize=16)
  plt.ylabel('$y$', fontsize=16, rotation=0)
  plt.xlim(6, 15)
  plt.ylim(7, 17)
  j += N
  k += N
plt.tight_layout()
plt.savefig('img417.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()

x_centered = x_m - x_m.mean()

with pm.Model() as unpooled_model:
  alpha_tmp = pm.Normal('alpha_tmp', mu=0, sd=10, shape=M)
  beta = pm.Normal('beta', mu=0, sd=10, shape=M)
  epsilon = pm.HalfCauchy('epsilon', 5)
  nu = pm.Exponential('nu', 1/30)
  y_pred = pm.StudentT('y_pred', mu= alpha_tmp[idx] + beta[idx] * x_centered, sd=epsilon, nu=nu, observed=y_m)
  alpha = pm.Deterministic('alpha', alpha_tmp - beta * x_m.mean()) 
    
  start = pm.find_MAP()
  step = pm.NUTS(scaling=start)
#
# ValueError: Unused step method arguments: {'njobs'}
#  trace_up = pm.sample(2000, step=step, start=start, njobs=1)
  trace_up = pm.sample(2000, step=step, start=start, chains=1)

varnames=['alpha', 'beta', 'epsilon', 'nu']
pm.traceplot(trace_up, varnames)
plt.savefig('img418.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()

with pm.Model() as hierarchical_model:
  alpha_tmp_mu = pm.Normal('alpha_tmp_mu', mu=0, sd=10)
  alpha_tmp_sd = pm.HalfNormal('alpha_tmp_sd', 10)
  beta_mu = pm.Normal('beta_mu', mu=0, sd=10)
  beta_sd = pm.HalfNormal('beta_sd', sd=10)

  alpha_tmp = pm.Normal('alpha_tmp', mu=alpha_tmp_mu, sd=alpha_tmp_sd, shape=M)
  beta = pm.Normal('beta', mu=beta_mu, sd=beta_sd, shape=M)
  epsilon = pm.HalfCauchy('epsilon', 5)
  nu = pm.Exponential('nu', 1/30)

  y_pred = pm.StudentT('y_pred', mu=alpha_tmp[idx] + beta[idx] * x_centered, sd=epsilon, nu=nu, observed=y_m)

  alpha = pm.Deterministic('alpha', alpha_tmp - beta * x_m.mean()) 
  alpha_mu = pm.Deterministic('alpha_mu', alpha_tmp_mu - beta_mu * x_m.mean())
  alpha_sd = pm.Deterministic('alpha_sd', alpha_tmp_sd - beta_mu * x_m.mean())
  
#TypeError: function() got an unexpected keyword argument 'njobs'
#  trace_hm = pm.sample(1000, njobs=1)
  trace_hm = pm.sample(1000, chains=1)

varnames=['alpha', 'alpha_mu', 'alpha_sd', 'beta', 'beta_mu', 'beta_sd', 'epsilon', 'nu']
pm.traceplot(trace_hm, varnames)
plt.savefig('img420.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()

plt.figure(figsize=(16,8))
j, k = 0, N
x_range = np.linspace(x_m.min(), x_m.max(), 10)
for i in range(M):
  plt.subplot(2,4,i+1)
  plt.scatter(x_m[j:k], y_m[j:k])
  plt.xlabel('$x_{}$'.format(i), fontsize=16)
  plt.ylabel('$y$', fontsize=16, rotation=0)
  alfa_m = trace_hm['alpha'][:,i].mean()
  beta_m = trace_hm['beta'][:,i].mean()
  plt.plot(x_range, alfa_m + beta_m * x_range, c='k', label='y = {:.2f} + {:.2f} * x'.format(alfa_m, beta_m))
  plt.xlim(x_m.min()-1, x_m.max()+1)
  plt.ylim(y_m.min()-1, y_m.max()+1)
  j += N
  k += N
plt.tight_layout()
plt.savefig('img421.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()

ans = sns.load_dataset('anscombe')
x_2 = ans[ans.dataset == 'II']['x'].values
y_2 = ans[ans.dataset == 'II']['y'].values
x_2 = x_2 - x_2.mean()
y_2 = y_2 - y_2.mean()

plt.scatter(x_2, y_2)
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.savefig('img422.png')
plt.figure()

with pm.Model() as model_poly:
  alpha = pm.Normal('alpha', mu=0, sd=10)
  beta1 = pm.Normal('beta1', mu=0, sd=1)
  beta2 = pm.Normal('beta2', mu=0, sd=1)
  epsilon = pm.HalfCauchy('epsilon', 5)

  mu = alpha + beta1 * x_2 + beta2 * x_2**2
  
  y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y_2)

#TypeError: function() got an unexpected keyword argument 'njobs'
#  trace_poly = pm.sample(2000, njobs=1)
  trace_poly = pm.sample(2000, chains=1)

pm.traceplot(trace_poly)
plt.tight_layout()
plt.savefig('img423.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()

x_p = np.linspace(-6, 6)
y_p = trace_poly['alpha'].mean() + trace_poly['beta1'].mean() * x_p + trace_poly['beta2'].mean() * x_p**2
plt.scatter(x_2, y_2)
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.plot(x_p, y_p, c='r')
plt.savefig('img424.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()