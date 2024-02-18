#
# Bayesian Analysis with Python written by Osvaldo Marthin
# Chapter 3-1
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

np.random.seed(123)
x = np.random.gamma(2, 1, 1000)
y = np.random.normal(0, 1, 1000)
print("start!!!!!!!!!!!")
data = pd.DataFrame(data=np.array([x, y]).T, columns=['$\\theta_1$', '$\\theta_2$'])
#
# AttributeError: 'PathCollection' object has no property 'stat_func'
# sns.jointplot(x='$\\theta_1$', y='$\\theta_2$', data=data, stat_func=None);
sns.jointplot(x='$\\theta_1$', y='$\\theta_2$', data=data);
plt.savefig('img301.png')

plt.figure()

data = np.array([51.06, 55.12, 53.73, 50.24, 52.05, 56.40, 48.45, 52.34,
55.65, 51.49, 51.86, 63.43, 53.00, 56.09, 51.93, 52.31, 52.33, 57.48, 
57.44, 55.14, 53.93, 54.62, 56.09, 68.58, 51.36, 55.47, 50.73, 51.94, 
54.95, 50.39, 52.91, 51.50, 52.68, 47.72, 49.73, 51.82, 54.99, 52.84, 
53.19, 54.52, 51.46, 53.73, 51.61, 49.81, 52.42, 54.30, 53.84, 53.16])

sns.kdeplot(data)
plt.savefig('img302.png')

plt.figure()

with pm.Model() as model_g:
  mu = pm.Uniform('mu', 40, 75)
  sigma = pm.HalfNormal('sigma', sd=10)
  y = pm.Normal('y', mu=mu, sd=sigma, observed=data)
#
# TypeError: function() got an unexpected keyword argument 'njobs'
#  trace_g = pm.sample(1100, njobs=1)
  trace_g = pm.sample(1100, chains=1)
chain_g = trace_g[100:]
pm.traceplot(chain_g)
plt.savefig('img304.png')

plt.figure()

#
# df = pm.summary(chain_g)
print("table3-1")
print(print(pm.summary(chain_g)))

#
# AttributeError: module 'pymc3' has no attribute 'sample_ppc'
# y_pred = pm.sample_ppc(chain_g, 100, model_g, size=len(data))
y_pred = pm.sample_posterior_predictive(chain_g, 100, model_g)
#ax = pm.kdeplot(data, color='C0')
#
#データセットの表示を省く100組の事後予測サンプルをうまく表示できるようにする
sns.kdeplot(data,c='b')
for i in y_pred['y']:
#    pm.kdeplot(i, color='C1', alpha=0.1, ax=ax)
    sns.kdeplot(i, c='r', alpha=0.1)
plt.xlim(35,75)
plt.title('Gaussian model', fontsize=16)
plt.xlabel('$x$', fontsize=16)
plt.savefig('img305.png')

plt.figure()

x_values = np.linspace(-10, 10,200)
for df in [1, 2, 5, 30]:
  distri = stats.t(df)
  x_pdf = distri.pdf(x_values)
  plt.plot(x_values, x_pdf, label=r'$\nu$ = {}'.format(df))

x_pdf = stats.norm.pdf(x_values)
plt.plot(x_values, x_pdf, label=r'$\nu = \infty$')
plt.xlabel('$x$')
plt.ylabel('$p(x)$')
plt.legend(loc=0, fontsize=14)
plt.xlim(-7, 7)
plt.savefig('img306.png')

plt.figure()

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
    with pm.Model() as model_t:
      mu = pm.Uniform('mu', 40, 75)
      sigma = pm.HalfNormal('sigma', sd=10)
      nu = pm.Exponential('nu', 1/30)
      y = pm.StudentT('y', mu=mu, sd=sigma, nu=nu, observed=data)
      trace_t = pm.sample(1100)
      chain_t = trace_t[100:]
#
#NameError: name 'chain_t' is not defined
      pm.traceplot(chain_t)
      plt.savefig('img308.png')

      plt.figure()
#
# NameError: name 'chain_t' is not defined
      print("table3-2")
      print(pm.summary(chain_t))

# AttributeError: module 'pymc3' has no attribute 'sample_ppc'
# y_pred = pm.sample_ppc(chain_t, 100, model_t, size=len(data))
      y_pred = pm.sample_posterior_predictive(chain_t, 100, model_t)
#
#データセットの表示を省く100組の事後予測サンプルをうまく表示できるようにする
      sns.kdeplot(data, c='b')
      for i in y_pred['y']:
#   sns.kdeplot(i, c='r', alpha=0.1)
#        sns.kdeplot(i, c='r', alpha=0.1, ax=ax)
        sns.kdeplot(i, c='r', alpha=0.1)
        plt.xlim(35,75)
        plt.title("Student's t model", fontsize=16)
        plt.xlabel('$x$', fontsize=16)
        plt.savefig('img309.png')

plt.figure()

