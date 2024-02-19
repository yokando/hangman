#!/usr/bin/env python
# coding: utf-8

# # PyMC3、正規分布の推定

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np
import pymc3 as pm
import scipy.stats as stats
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')


# #### 観測データは2つの正規分布の連結とする
# この事実は，まだ，知らないものとする

# In[ ]:


np.random.seed(123)
n1, n2 = 100, 100
y1 = np.random.normal(loc=5, scale=10, size=n1)
y2 = np.random.normal(loc=8, scale= 6, size=n2)
y = np.concatenate([y1, y2])


# In[ ]:


print(y1.min(), y2.min(), y.min())
print(y1.max(), y2.max(), y.max())


# In[ ]:


bins=20
range = (-30, 30)
plt.hist(y1, bins=bins, range=range, histtype='step')
plt.hist(y2, bins=bins, range=range, histtype='step', alpha=0.5)


# #### 観測データの情報

# In[ ]:


num = len(y)
print(num, y.mean(), y.std())


# In[ ]:


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
axs[0].plot(y)
axs[1].hist(y, bins=30, range=(-30, 30))

axs[0].set_xlabel('k')
axs[0].set_ylabel('y')
axs[1].set_xlabel('y')
axs[1].set_ylabel('frequency')

#plt.savefig('fig_Bayes_PyMC3_Normal_01.png', bbox_inches='tight') #xlabeの欠落を防ぐ方法2


# In[ ]:


with pm.Model() as model_norm:
    mu = pm.Uniform('mu', 2, 10)
    sigma = pm.HalfNormal('sigma', sd=20)
    y_rv = pm.Normal('y_rv', mu=mu, sd=sigma, observed=y)


# In[ ]:


tree_model = pm.model_to_graphviz(model_norm)
tree_model
#木構造の図をPNGで保存したい場合には，次の行のコメントをはずす
#tree_model.render(filename='fig_Bayes_PyMC3_Normal_TreeModel', format='png') # DOT言語ファイルと画像ファイルの保存


# In[ ]:


burnin = 100

# RuntimeError:
#        An attempt has been made to start a new process before the
#        current process has finished its bootstrapping phase.
if __name__ == '__main__':
               
  with model_norm:
#    start = pm.find_MAP()
#    step = pm.Metropolis() #離散型変数
    trace = pm.sample(draws=1000+burnin, tune=100, cores=2)


# In[ ]:


    trace_burned = trace[burnin:]
    pm.traceplot(trace_burned, legend=True)
    plt.savefig('fig_Bayes_PyMC3_Normal_02.png', bbox_inches='tight') #xlabeの欠落を防ぐ方法2


# In[ ]:


    print(len(trace_burned['mu']), len(trace_burned['sigma']))


# In[ ]:


    pm.plot_posterior(trace_burned)
    plt.savefig('fig_Bayes_PyMC3_Normal_posterior.png', bbox_inches='tight') #xlabeの欠落を防ぐ方法2


# In[ ]:


    print(pm.summary(trace_burned))


# #### posterior predictive checks
# https://docs.pymc.io/api/inference.html<br>
# https://docs.pymc.io/notebooks/posterior_predictive.html

# In[ ]:


    ppc = pm.sample_posterior_predictive(trace_burned, samples=20, model=model_norm)


# In[ ]:


    np.array(ppc['y_rv']).shape


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html

# In[ ]:


    plt.figure()
    Xlow, Xhigh = -30,50
    X = np.arange(Xlow, Xhigh, 0.1)
#    plt.xlim(Xlow, Xhigh)
    kde = stats.gaussian_kde(y) # yのKDE
    plt.plot(X, kde(X), linestyle='solid', color='k', label='kde of data')

    for yy in ppc['y_rv']:
      kde_ppt = stats.gaussian_kde(yy)
      plt.plot(X, kde_ppt(X), color='r', alpha=0.2)

    plt.title("Posterior predictive of Normal model")
    plt.legend()
    plt.grid(True)

#    plt.savefig('fig_Bayes_PyMC3_Normal_ppc.png')
    plt.show()


# In[ ]:




