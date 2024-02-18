#
# Bayesian Analysis with Python written by Osvaldo Marthin
# Chapter 4-1
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

rates = [1, 2, 5]
scales = [1, 2, 3]

x = np.linspace(0, 20, 100)
f, ax = plt.subplots(len(rates), len(scales), sharex=True, sharey=True)
for i in range(len(rates)):
  for j in range(len(scales)):
    rate = rates[i]
    scale = scales[j]
    rv = stats.gamma(a=rate, scale=scale)
    ax[i,j].plot(x, rv.pdf(x))
    ax[i,j].plot(0, 0, label="$\\alpha$ = {:3.2f}\n$\\theta$ = {:3.2f}".format(rate, scale), alpha=0)
    ax[i,j].legend()

ax[2,1].set_xlabel('$x$')
ax[1,0].set_ylabel('$pdf(x)$')
plt.savefig('img401.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()

np.random.seed(314)
N = 100
alfa_real = 2.5
beta_real = 0.9
eps_real = np.random.normal(0, 0.5, size=N)

x = np.random.normal(10, 1, N)
y_real = alfa_real + beta_real * x
y = y_real + eps_real

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x, y, 'b.')
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.plot(x, y_real, 'k')
plt.subplot(1, 2, 2)
sns.kdeplot(y)
plt.xlabel('$y$', fontsize=16)
plt.savefig('img403.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()

with pm.Model() as model:
  alpha = pm.Normal('alpha', mu=0, sd=10)
  beta = pm.Normal('beta', mu=0, sd=1)
  epsilon = pm.HalfCauchy('epsilon', 5)

  mu = pm.Deterministic('mu', alpha + beta * x)
  y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y)

  start = pm.find_MAP()
  step = pm.Metropolis()
#
# ValueError: Unused step method arguments: {'njobs'}
#  trace = pm.sample(11000, step, start, njobs=1)
  trace = pm.sample(11000, step, start, chains=1)
trace_n = trace[1000:]
pm.traceplot(trace_n)
plt.savefig('img404.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()

varnames = ['alpha', 'beta', 'epsilon']
pm.autocorrplot(trace_n, varnames)
plt.savefig('img405.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()

#
# TypeError: kdeplot() takes from 0 to 1 positional arguments but 2 were given
# sns.kdeplot(trace_n['alpha'], trace_n['beta'])
x1,y1 = (trace_n['alpha'], trace_n['beta'])
sns.kdeplot(x=x1, y=y1)
plt.xlabel(r'$\alpha$', fontsize=16)
plt.ylabel(r'$\beta$', fontsize=16, rotation=0)
plt.savefig('img406.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()

plt.plot(x, y, 'b.');
alpha_m = trace_n['alpha'].mean()
beta_m = trace_n['beta'].mean()
plt.plot(x, alpha_m + beta_m * x, c='k', label='y = {:.2f} + {:.2f} * x'.format(alpha_m, beta_m))
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.legend(loc=2, fontsize=14)
plt.savefig('img407.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()

plt.plot(x, y, 'b.');
idx = range(0, len(trace_n['alpha']), 10)
plt.plot(x, trace_n['alpha'][idx] + trace_n['beta'][idx] * x[:,np.newaxis], c='gray', alpha=0.5);

plt.plot(x, alpha_m + beta_m * x, c='k', label='y = {:.2f} + {:.2f} * x'.format(alpha_m, beta_m))
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.legend(loc=2, fontsize=14)
plt.savefig('img408.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()

plt.plot(x, alpha_m + beta_m * x, c='k', label='y = {:.2f} + {:.2f} * x'.format(alpha_m, beta_m))

idx = np.argsort(x)
x_ord = x[idx]
#
# AttributeError: module 'pymc3' has no attribute 'hpd'
# sig = pm.hpd(trace_n['mu'], alpha=.02)[idx]
sig = az.hdi(trace_n['mu'])[idx]
plt.fill_between(x_ord, sig[:,0], sig[:,1], color='gray')

plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.savefig('img409.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()

#
# AttributeError: module 'pymc3' has no attribute 'sample_ppc'
# ppc = pm.sample_ppc(trace_n, samples=1000, model=model)
ppc = pm.sample_posterior_predictive(trace_n, samples=1000, model=model)
plt.plot(x, y, 'b.')
plt.plot(x, alpha_m + beta_m * x, c='k', label='y = {:.2f} + {:.2f} * x'.format(alpha_m, beta_m))

#
# AttributeError: module 'pymc3' has no attribute 'hpd'
# sig0 = pm.hpd(ppc['y_pred'], alpha=0.5)[idx]
# sig1 = pm.hpd(ppc['y_pred'], alpha=0.05)[idx]
sig0 = az.hdi(ppc['y_pred'], hdi_prob=0.5)[idx]
sig1 = az.hdi(ppc['y_pred'], hdi_prob=0.95)[idx]
plt.fill_between(x_ord, sig0[:,0], sig0[:,1], color='gray', alpha=1)
plt.fill_between(x_ord, sig1[:,0], sig1[:,1], color='gray', alpha=0.5)

plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.savefig('img410.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()

np.random.seed(314)
N = 100
alfa_real = 2.5
beta_real = 0.9
eps_real = np.random.normal(0, 0.5, size=N)

x = np.random.normal(10, 1, N)
y_real = alfa_real + beta_real * x
y = y_real + eps_real

#
#RuntimeError:
#        An attempt has been made to start a new process before the
#        current process has finished its bootstrapping phase.
#
#        This probably means that you are not using fork to start your
#        child processes and you have forgotten to use the proper idiom
#        in the main module:
#
#            if __name__ == '__main__':
#                freeze_support()
#                ...
#        The "freeze_support()" line can be omitted if the program
#        is not going to be frozen to produce an executable.
#
if __name__ == '__main__':
  with pm.Model() as model_n:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=1)
    epsilon = pm.HalfCauchy('epsilon', 5)

    mu = alpha + beta * x
    y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y)

    rb = pm.Deterministic('rb', (beta * x.std() / y.std()) ** 2)

    y_mean = y.mean()
    ss_reg = pm.math.sum((mu - y_mean) ** 2)
    ss_tot = pm.math.sum((y - y_mean) ** 2)
    rss = pm.Deterministic('rss', ss_reg/ss_tot)

    start = pm.find_MAP()
    step = pm.NUTS()
    trace_n = pm.sample(2000, step=step, start=start)
#
    print("table4-1")
    print(pm.summary(trace_n))
#
    pm.traceplot(trace_n)
    plt.savefig('img411.png', dpi=300, figsize=(5.5, 5.5))
    plt.figure()
    varnames = ['alpha', 'beta', 'epsilon', 'rb', 'rss']
#
# KeyError: 'var names: "[\'rb\' \'rss\'] are not present" in dataset'
#    varnames = ['alpha', 'beta', 'epsilon', 'rb', 'rss']
#    pm.summary(trace_n, varnames))
#    print("table4-1")
#    print(pm.summary(trace_n))

sigma_x1 = 1
sigmas_x2 = [1, 2]
rhos = [-0.99, -0.5, 0, 0.5, 0.99]

x, y = np.mgrid[-5:5:.1, -5:5:.1]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y

f, ax = plt.subplots(len(sigmas_x2), len(rhos), sharex=True, sharey=True)

for i in range(2):
  for j in range(5):
    sigma_x2 = sigmas_x2[i]
    rho = rhos[j]
    cov = [[sigma_x1**2, sigma_x1*sigma_x2*rho], [sigma_x1*sigma_x2*rho, sigma_x2**2]]
    rv = stats.multivariate_normal([0, 0], cov)
    ax[i,j].contour(x, y, rv.pdf(pos))
    ax[i,j].plot(0, 0, label="$\\sigma_{{x2}}$ = {:3.2f}\n$\\rho$ = {:3.2f}".format(sigma_x2, rho), alpha=0)
    ax[i,j].legend()
ax[1,2].set_xlabel('$x_1$')
ax[1,0].set_ylabel('$x_2$')
plt.savefig('img412.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()

np.random.seed(314)
N = 100
alfa_real = 2.5
beta_real = 0.9
eps_real = np.random.normal(0, 0.5, size=N)

x = np.random.normal(10, 1, N)
y_real = alfa_real + beta_real * x 
y = y_real + eps_real

data = np.stack((x, y)).T

with pm.Model() as pearson_model:
  mu = pm.Normal('mu', mu=data.mean(0), sd=10, shape=2)
  sigma_1 = pm.HalfNormal('simga_1', 10)
  sigma_2 = pm.HalfNormal('sigma_2', 10)
  rho = pm.Uniform('rho', -1, 1)
  
  cov = pm.math.stack(([sigma_1**2, sigma_1*sigma_2*rho], [sigma_1*sigma_2*rho, sigma_2**2]))
  
  y_pred = pm.MvNormal('y_pred', mu=mu, cov=cov, observed=data)
  
  start = pm.find_MAP()
  step = pm.NUTS(scaling=start)
#
# ValueError: Unused step method arguments: {'njobs'}
#  trace_p = pm.sample(1000, step=step, start=start, njobs=1)
  trace_p = pm.sample(1000, step=step, start=start, chains=1)

pm.traceplot(trace_p)
plt.savefig('img413.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()

ans = sns.load_dataset('anscombe')
x_3 = ans[ans.dataset == 'III']['x'].values
y_3 = ans[ans.dataset == 'III']['y'].values
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
beta_c, alpha_c = stats.linregress(x_3, y_3)[:2]
plt.plot(x_3, (alpha_c + beta_c* x_3), 'k', label='y ={:.2f} + {:.2f} * x'.format(alpha_c, beta_c))
plt.plot(x_3, y_3, 'bo')
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', rotation=0, fontsize=16)
plt.legend(loc=0, fontsize=14)
plt.subplot(1,2,2)
sns.kdeplot(y_3);
plt.xlabel('$y$', fontsize=16)
plt.tight_layout()
plt.savefig('img414.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()

with pm.Model() as model_t:
  alpha = pm.Normal('alpha', mu=0, sd=100)
  beta = pm.Normal('beta', mu=0, sd=1)
  epsilon = pm.HalfCauchy('epsilon', 5)
  nu = pm.Deterministic('nu', pm.Exponential('nu_', 1/29) + 1)
  
  y_pred = pm.StudentT('y_pred', mu=alpha + beta * x_3, sd=epsilon, nu=nu, observed=y_3)
  
  start = pm.find_MAP()
  step = pm.NUTS(scaling=start) 
#
# ValueError: Unused step method arguments: {'njobs'}
#  trace_t = pm.sample(2000, step=step, start=start, njobs=1)
  trace_t = pm.sample(2000, step=step, start=start, chains=1)

beta_c, alpha_c = stats.linregress(x_3, y_3)[:2]

plt.plot(x_3, (alpha_c + beta_c * x_3), 'k', label='non-robust', alpha=0.5)
plt.plot(x_3, y_3, 'bo')
alpha_m = trace_t['alpha'].mean()
beta_m = trace_t['beta'].mean()
plt.plot(x_3, alpha_m + beta_m * x_3, c='k', label='robust')

plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', rotation=0, fontsize=16)
plt.legend(loc=2, fontsize=12)
plt.savefig('img415.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()

# ppc = pm.sample_ppc(trace_n, samples=1000, model=model)
# ppc = pm.sample_ppc(trace_t, samples=200, model=model_t, random_seed=2)
ppc = pm.sample_posterior_predictive(trace_t, samples=200, model=model_t, random_seed=2)
for y_tilde in ppc['y_pred']:
  sns.kdeplot(y_tilde, alpha=0.5, c='g')

sns.kdeplot(y_3, linewidth=3)
plt.savefig('img416.png', dpi=300, figsize=(5.5, 5.5))
plt.figure()
