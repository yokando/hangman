#!/usr/bin/env python
# coding: utf-8

# # PyMC3による線形回帰モデルの推定

# $$
# y_i = \alpha + \beta x_i \\
# y \sim N(\mu = \alpha + \beta x, \sigma = \varepsilon ) \\
# \alpha \sim N (\mu_{\alpha}, \sigma_{\alpha} ) \\
# \beta \sim N(\mu_{\beta}, \sigma_{\beta}) \\
# \varepsilon \sim U(0, h_s)
# $$

# In[ ]:


# -*- coding: utf-8 -*-
import pymc3 as pm
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
#
import arviz as az

# get_ipython().run_line_magic('matplotlib', 'inline')


# #### 単回帰モデルを用いたデータ生成

# In[ ]:


np.random.seed(123)
num = 100
alpha_real = 2.5
beta_real  = 0.9
x = np.random.normal(10, 1, num)
y_real = alpha_real + beta_real*x
eps_real = np.random.normal(0, 0.5, size=num)
y = y_real + eps_real


# In[ ]:


print(x.mean(), x.std(), x.min(), x.max(), x.shape)
print(y.mean(), y.std(), y.min(), y.max(), y.shape)


# In[ ]:


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
#axs[0].plot(x, y_real, c='k')
axs[0].scatter(x, y, c='k')
axs[0].grid()
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')

sns.kdeplot(data=y, ax=axs[1], color='k')
axs[1].set_xlabel('y')
axs[1].set_ylabel('density')

#plt.savefig('fig_Bayes_PyMC3_Regre_01.png')


# #### Bounded Variables
# https://docs.pymc.io/api/bounds.html <br>
# HalfCauchy: https://docs.pymc.io/api/distributions/continuous.html

# #### deterministic変数
# Deterministic transforms: https://docs.pymc.io/notebooks/api_quickstart.html<br>
# Model: https://docs.pymc.io/api/model.html

# #### Half Cauchy分布
# PyMC3 API: https://docs.pymc.io/api/distributions/continuous.html#pymc3.distributions.continuous.HalfCauchy<br>
# Wikipedia: https://ja.wikipedia.org/wiki/コーシー分布<br>
# Wolfram Mathworld – Cauchy distribution: https://mathworld.wolfram.com/CauchyDistribution.html
# 

# ### パラメータのmuを確率にする

# In[ ]:


# 線形回帰モデル
with pm.Model() as model_linear:
    alpha = pm.Normal('alpha', mu=0.0, sd=10.0)
    beta  = pm.Normal('beta',  mu=1.0, sd=1.0)
    epsilon = pm.HalfCauchy('epsilon', 5)
#    mu = pm.Deterministic('mu', alpha+beta*x)
    y_rv = pm.Normal('y_rv', mu=alpha+beta*x, sd=epsilon, observed=y)


# In[ ]:


tree_model = pm.model_to_graphviz(model_linear)
tree_model
#木構造の図をPNGで保存したい場合には，次の行のコメントをはずす
#tree_model.render(filename='fig_Bayes_PyMC3_Regre_TreeModel', format='png') # DOT言語ファイルと画像ファイルの保存


# In[ ]:


burnin = 1000

# RuntimeError:
#       An attempt has been made to start a new process before the
#        current process has finished its bootstrapping phase.
if __name__ == '__main__':

  with model_linear:
    start = pm.find_MAP()
    step  = pm.NUTS()
    trace = pm.sample(draws=5000+burnin, tune=1000, start=start, step=step, cores=2)


# In[ ]:


    trace_burned = trace[burnin:]
    pm.traceplot(trace_burned, legend=True)
    plt.savefig('fig_Bayes_PyMC3_Regre_traceplot.png')
    plt.figure()

# In[ ]:


    print(pm.summary(trace_burned))


# #### autocorrplot()
# https://docs.pymc.io/api/plots.html

# In[ ]:

    varnames = ['alpha', 'beta', 'epsilon']
#
    pm.autocorrplot(trace_burned, varnames)

    plt.savefig('fig_Bayes_PyMC3_Regre_autocorr.png')
    plt.figure()

# #### $\alpha$と$\beta$の相関を見るために位相平面を描く。ここに，等高線で表す

# In[ ]:

# TypeError: `data2` has been removed (replaced by `y`); please update your code.
#    sns.kdeplot(data=trace_burned['alpha'], data2=trace_burned['beta'])
    sns.kdeplot(x=trace_burned['alpha'], y=trace_burned['beta'])
    plt.xlabel('alpha')
    plt.ylabel('beta')

    plt.savefig('fig_Bayes_PyMC3_Regre_phaseplane.png')
    plt.figure()


# ### 事後分布の考察と視覚化

# In[ ]:


    alpha_mean = trace_burned['alpha'].mean()
    beta_mean  = trace_burned['beta'].mean()
    print(alpha_mean, beta_mean)
    y_rv_mean = alpha_mean + beta_mean*x


# In[ ]:


    plt.plot(x,y_rv_mean, c='k')
#plt.plot(x,y_real, c='r')
    plt.scatter(x,y, s=10)

    plt.xlabel('x')
    plt.ylabel('y')

    plt.savefig('fig_Bayes_PyMC3_Regre_mean.png')
    plt.figure()

# #### 回帰直線の不確実性を表す

# In[ ]:


    for i in range(100):
      y_rv_pred = trace_burned['alpha'][i] + trace_burned['beta'][i]*x
      plt.plot(x, y_rv_pred, c='g', alpha=0.1)
    plt.scatter(x,y, s=10, c='k')

    plt.xlabel('x')
    plt.ylabel('y')

    plt.savefig('fig_Bayes_PyMC3_Regre_uncertainty.png')
    plt.figure()

# In[ ]:


    print(x.min(), x.max())


# ### numpy.argsortの使い方を知る
# 配列の要素に対するソートを行うとき，値そのものをソートするのではなく，ソートされたときの要素番号（インデックス）を返す。

# In[ ]:


    xxx = np.random.randint(0,100,5)
    xxx


# In[ ]:


    np.argsort(xxx)


# #### sample_posterior_predictive()
# https://docs.pymc.io/api/inference.html

# In[ ]:


    ppc = pm.sample_posterior_predictive(trace_burned, samples=1000, model=model_linear)


# #### 関数の使い方
# 統計と診断に関する関数はArviZライブラリに任されているため，pymc3の関数はArviZ関数のエイリアスとなっている。<br>このため，使い方はAriviZ関数のドキュメントを見ることとなる。<br>
# pymc3.stats.hpd(): https://docs.pymc.io/api/stats.html <br>
# 
# 次はmaplotlibの関数である。<br>
# plt.fill_between(): https://matplotlib.org/api/_as_gen/matplotlib.pyplot.fill_between.html

# In[ ]:


    idx = np.argsort(x)
    x_ord = x[idx]
#    hpd50 = pm.hpd(ppc['y_rv'], hdi_prob=0.5)[idx]  # 50%HPD
#    hpd95 = pm.hpd(ppc['y_rv'], hdi_prob=0.95)[idx] # 95%HPD
    hpd50 = az.hdi(ppc['y_rv'], hdi_prob=0.5)[idx]  # 50%HPD
    hpd95 = az.hdi(ppc['y_rv'], hdi_prob=0.95)[idx] # 95%HPD

    plt.fill_between(x_ord, hpd50[:,0], hpd50[:,1], color='gray', alpha=0.8)
    plt.fill_between(x_ord, hpd95[:,0], hpd95[:,1], color='gray', alpha=0.3)
    plt.scatter(x,y, s=10, c='k')
    plt.plot(x,y_rv_mean, c='k')

    plt.xlabel('x')
    plt.ylabel('y')

    plt.savefig('fig_Bayes_PyMC3_Regre_HPD.png')
    plt.figure()

# In[ ]:




