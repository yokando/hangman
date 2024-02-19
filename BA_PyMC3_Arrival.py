#!/usr/bin/env python
# coding: utf-8

# # PyMC3, 到着数の変化検出
# 本例題は，C.D.Pilon;Pythonで体験するベイズ推論，森北出版，1.4節にヒントを得ている。<br>
# 到着数は，メールの受信数，待ち行列（銀行のATMなど），交通事故数などを表す。

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np
import pymc3 as pm
import scipy.stats as stats
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')


# #### 到着数の人工データの生成

# In[ ]:


np.random.seed(123)
n1, n2 = 50, 50
lambda_1_real, lambda_2_real = 10, 20
y1 = np.random.poisson(lambda_1_real, n1)
y2 = np.random.poisson(lambda_2_real, n2)
y  = np.concatenate([y1, y2])


# #### 本文は，観測データ yを得たところから話が始まる

# In[ ]:


num = len(y)
days = np.arange(num)


# In[ ]:


fig = plt.subplots(figsize=(12,4))
plt.bar(days, y)
plt.xlabel('day (k)')
plt.ylabel('the number of arrival y')
#plt.savefig('fig_Bayes_PyMC3_Arrival_01.png')
plt.show()


# #### 確率モデルの作成
# Continuous<br>
# pm.Exponential(): https://docs.pymc.io/api/distributions/continuous.html<br>
# Discrete<br>
# pm.DiscreteUniform(), pm.Poisson():https://docs.pymc.io/api/distributions/discrete.html

# #### pymc3でのモデル関数が条件分岐を含む場合の書き方
# switch: https://docs.pymc.io/api/math.html<br>

# In[ ]:


alpha = 1/y.mean() # 初期値
with pm.Model() as model:
    tau  = pm.DiscreteUniform('tau', lower=0, upper=num)
    lambda_1 = pm.Exponential('lambda_1', alpha)
    lambda_2 = pm.Exponential('lambda_2', alpha)
    # days<tau未満:lambda_1, >=tau以上:lambda_2を返す、lambda_を確定変数として扱う
    lambda_ = pm.math.switch(days < tau, lambda_1, lambda_2)

    y_rv = pm.Poisson('y_rv', lambda_, observed=y)


# In[ ]:


tree_model = pm.model_to_graphviz(model)
tree_model
#木構造の図をPNGで保存したい場合には，次の行のコメントをはずす
#tree_model.render(filename='fig_Bayes_PyMC3_Arrival_tree', format='png') # DOT言語ファイルと画像ファイルの保存


# In[ ]:


burnin = 1000

# RuntimeError:
#        An attempt has been made to start a new process before the
#        current process has finished its bootstrapping phase.
if __name__ == '__main__':

  with model:
    trace = pm.sample(draws=5000+burnin, tune=1000, cores=2)


# In[ ]:


    trace_burned = trace[burnin:]
    pm.traceplot(trace_burned, legend=True)
#plt.savefig('fig_Bayes_PyMC3_Arrival_traceplot.png', bbox_inches='tight') #xlabeの欠落を防ぐ方法2


# In[ ]:


    print(pm.summary(trace_burned))


# In[ ]:


    pm.plot_posterior(trace_burned, var_names=('lambda_1','lambda_2'))
#plt.savefig('fig_Bayes_PyMC3_Arrival_posterior_lambda.png', bbox_inches='tight')


# In[ ]:


    pm.plot_posterior(trace_burned['tau'])
#plt.savefig('fig_Bayes_PyMC3_Arrival_posterior_tau.png', bbox_inches='tight')


# #### posterior predictive samples
# https://docs.pymc.io/api/inference.html<br>
# https://docs.pymc.io/notebooks/posterior_predictive.html

# In[ ]:

    ppc = pm.sample_posterior_predictive(trace_burned, samples=20, model=model)
    print("ppc : ", ppc)


# In[ ]:

    plt.figure()
    Xlow, Xhigh = -10,50
    X = np.arange(Xlow, Xhigh, 0.1)
    plt.xlim(Xlow, Xhigh)
    kde = stats.gaussian_kde(y) # DataのKDE
    plt.plot(X, kde(X), linestyle='solid', color='k', label='kde of data')

    for yy in ppc['y_rv']:
#      plt.hist(yy, bins=60, range=(Xlow, Xhigh), histtype='step', density=True, alpha=0.5)
      kde_ppt = stats.gaussian_kde(yy)
      plt.plot(X, kde_ppt(X), color='r', alpha=0.2)

# Poissonのpmf（確率質量関数）をプロット，平均値は上記の結果を使用する
# 本来は離散確率であるが，実線でプロットしている
    XX = np.arange(0, Xhigh) 
    plt.plot(XX, stats.poisson.pmf(XX, 9.8), c='b')
    plt.plot(XX, stats.poisson.pmf(XX, 20.0), c='b')

    plt.ylabel('Probability')
    plt.title("Posterior predictive")
    plt.legend()
    plt.grid(True)

    plt.savefig('fig_Bayes_PyMC3_Arrive_ppc.png')
#    plt.show()

# In[ ]:


# サンプリング結果の分離
    lambda_1_samples = trace_burned.lambda_1
    lambda_2_samples = trace_burned.lambda_2
    tau_samples = trace_burned.tau


# In[ ]:


#lambdaのヒストグラム
    fig= plt.subplots(figsize=(18, 4))
    plt.hist(lambda_1_samples, density=True, bins=50, label='lambda_1')
    plt.hist(lambda_2_samples, density=True, bins=50, label='lambda_2')
    plt.xlim(7, 24)
    plt.legend()

    plt.savefig('fig_Bayes_PyMC3_Arrival_lambda.png', bbox_inches='tight') #
#    plt.show()


# In[ ]:


# tau
    fig = plt.subplots(figsize=(18, 4))
    plt.hist(tau_samples, bins=np.arange(44, 56, 1), rwidth=0.9, density=True, align='left', label='tau')
    plt.xticks(np.arange(44, 56, 1))
    plt.legend()

    plt.savefig('fig_Bayes_PyMC3_Arrival_tau.png', bbox_inches='tight') #
#    plt.show()


# In[ ]:




