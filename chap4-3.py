﻿#
# Bayesian Analysis with Python written by Osvaldo Marthin
# Chapter 4-3
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

np.random.seed(314)
N = 100
alpha_real = 2.5
beta_real = [0.9, 1.5]
eps_real = np.random.normal(0, 0.5, size=N)

X = np.array([np.random.normal(i, j, N) for i,j in zip([10, 2], [1, 1.5])])
X_mean = X.mean(axis=1, keepdims=True)
X_centered = X - X_mean
y = alpha_real + np.dot(beta_real, X) + eps_real

def scatter_plot(x, y):
  plt.figure(figsize=(10, 10))
  for idx, x_i in enumerate(x):
    plt.subplot(2, 2, idx+1)
    plt.scatter(x_i, y)
    plt.xlabel('$x_{}$'.format(idx+1), fontsize=16)
    plt.ylabel('$y$', rotation=0, fontsize=16)

  plt.subplot(2, 2, idx+2)
  plt.scatter(x[0], x[1])
  plt.xlabel('$x_{}$'.format(idx), fontsize=16)
  plt.ylabel('$x_{}$'.format(idx+1), rotation=0, fontsize=16)

scatter_plot(X_centered, y)
plt.savefig('img425.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

with pm.Model() as model_mlr:
  alpha_tmp = pm.Normal('alpha_tmp', mu=0, sd=10)
  beta = pm.Normal('beta', mu=0, sd=1, shape=2)
  epsilon = pm.HalfCauchy('epsilon', 5)
  
  mu = alpha_tmp + pm.math.dot(beta, X_centered)
  alpha = pm.Deterministic('alpha', alpha_tmp - pm.math.dot(beta, X_mean)) 
  
  y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y)

#TypeError: function() got an unexpected keyword argument 'njobs'
#  trace_mlr = pm.sample(5000, njobs=1)
  trace_mlr = pm.sample(5000, chains=1)


varnames = ['alpha', 'beta','epsilon']
pm.traceplot(trace_mlr, varnames)
plt.savefig('img426.png', dpi=300, figsize=(5.5, 5.5))
print("table4-2")
print(pm.summary(trace_mlr, varnames))
plt.figure()

x1,y1 = (trace_mlr['beta'][:,0], trace_red['beta'][:,1])
sns.kdeplot(x=x1,y=y1)
plt.xlabel(r'$\beta_1$', fontsize=16)
plt.ylabel(r'$\beta_2$', fontsize=16, rotation=0)
plt.savefig('img426-1.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()

np.random.seed(314)
N = 100
x_1 = np.random.normal(size=N)
x_2 = x_1 + np.random.normal(size=N, scale=1)
y = x_1 + np.random.normal(size=N)
X = np.vstack((x_1, x_2))
scatter_plot(X, y)
plt.savefig('img427.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()

with pm.Model() as model_red:
  alpha = pm.Normal('alpha', mu=0, sd=1)
  beta = pm.Normal('beta', mu=0, sd=10, shape=2)
  epsilon = pm.HalfCauchy('epsilon', 5)
  
  mu = alpha + pm.math.dot(beta, X)
  y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y)
  
#TypeError: function() got an unexpected keyword argument 'njobs'
#  trace_red = pm.sample(5000, njobs=1)
  trace_red = pm.sample(5000, chains=1)

pm.traceplot(trace_red)
plt.savefig('img428.png', dpi=300, figsize=(5.5, 5.5))
print("table4-3")
print(pm.summary(trace_red))
plt.figure()

#
# TypeError: kdeplot() takes from 0 to 1 positional arguments but 2 were given
# sns.kdeplot(trace_red['beta'][:,0], trace_red['beta'][:,1])
x1,y1 = (trace_red['beta'][:,0], trace_red['beta'][:,1])
sns.kdeplot(x=x1,y=y1)
plt.xlabel(r'$\beta_1$', fontsize=16)
plt.ylabel(r'$\beta_2$', fontsize=16, rotation=0)
plt.savefig('img429.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()

x_2 = x_1 + np.random.normal(size=N, scale=0.01)
y = x_1 + np.random.normal(size=N)
X = np.vstack((x_1, x_2))
scatter_plot(X, y)
plt.tight_layout()
plt.savefig('img427-1.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()

with pm.Model() as model_red:
  alpha = pm.Normal('alpha', mu=0, sd=1)
  beta = pm.Normal('beta', mu=0, sd=10, shape=2)
  epsilon = pm.HalfCauchy('epsilon', 5)
  mu = alpha + pm.math.dot(beta, X)
  y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y)
  trace_red = pm.sample(5000, chains=1)
pm.traceplot(trace_red)
plt.tight_layout()
plt.savefig('img428-1.png', dpi=300, figsize=(5.5, 5.5))
print("table4-3-1")
print(pm.summary(trace_red))
plt.figure()

#
# TypeError: kdeplot() takes from 0 to 1 positional arguments but 2 were given
# sns.kdeplot(trace_red['beta'][:,0], trace_red['beta'][:,1])
x1,y2 = (trace_red['beta'][:,0], trace_red['beta'][:,1])
sns.kdeplot(x=x1,y=y2)
plt.xlabel(r'$\beta_1$', fontsize=16)
plt.ylabel(r'$\beta_2$', fontsize=16, rotation=0)
plt.savefig('img429-1.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()

#
# TypeError: DeprecationWarning() takes no keyword arguments
# pm.forestplot(trace_red, varnames=['beta'])
# plt.savefig('img430.png', dpi=300, figsize=(5.5, 5.5))
# plt.figure()

np.random.seed(314)
N = 100
r = 0.8

x_1 = np.random.normal(size=N)
x_2 = np.random.normal(loc=x_1 * r, scale=(1 - r ** 2) ** 0.5)
y = np.random.normal(loc=x_1 - x_2)
X = np.vstack((x_1, x_2))

scatter_plot(X, y)
plt.savefig('img431.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

with pm.Model() as model_ma:
  alpha = pm.Normal('alpha', mu=0, sd=10)
  beta = pm.Normal('beta', mu=0, sd=10, shape=2)
  epsilon = pm.HalfCauchy('epsilon', 5)

  mu = alpha + pm.math.dot(beta, X)
  y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y)


#TypeError: function() got an unexpected keyword argument 'njobs'
#  trace_ma = pm.sample(5000, njobs=1)
  trace_ma = pm.sample(5000, chains=1)

pm.traceplot(trace_ma)
plt.savefig('img432.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()

#
# TypeError: DeprecationWarning() takes no keyword arguments
# pm.forestplot(trace_ma, varnames=['beta']);
# plt.savefig('img433.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

