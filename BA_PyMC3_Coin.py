#!/usr/bin/env python
# coding: utf-8

# # PyMC3，コイン投げ
# $\theta$: 表のでる確率，$N$:試行数，$k$:表の出た回数，$y$:ベルヌーイ試行したときの結果（例：y=(0,1,1,0,1)）。<br>
# 尤度関数は次である。<br>
# $$
#  f\left( {Y = y | \theta } \right)   =  {}_N C_k    \, \theta^k  \left( {1 - \theta} \right)^{N - k}
# $$

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as stats
import pymc3 as pm
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:

np.random.seed(123) # 再現性を得るため，数字123は任意
num = 100 # 試行数
theta_real = 0.35   # 表の出る確率
y = stats.bernoulli.rvs(p=theta_real, size=num)
total_front = y.sum() #　表の出た回数


# #### 確率モデルの作成
# with pm.Model() as model:で，"model"という名前でModelオブジェクトが作成される。これはコンテナとして，後続で宣言される確率変数を含む。
# 参照：<br>
# ・Model Specification: https://docs.pymc.io/notebooks/getting_started.html<br>
# ・Model: https://docs.pymc.io/api/model.html<br>
# 宣言される確率変数<br>
# $\theta$，$y$は，それぞれ，ベータ分布，二項分布に従う確率変数と定義される。<br>
# 参照：<br>
# ・ベータ分布: https://docs.pymc.io/api/distributions/continuous.html<br>
# ・二項分布: https://docs.pymc.io/api/distributions/discrete.html<br>
# モデルのコンパイルはwithブロックを抜けるときに行っている。

# In[ ]:


with pm.Model() as model:
    theta = pm.Beta('theta', alpha=1, beta=1)
    y_rv = pm.Binomial('y_rv', n=num, p=theta, observed=total_front) # rv:random variable
#    y_rv = pm.Bernoulli('y_rv', p=theta, observed=data) # ベルヌーイ分布を用いる場合の使用法


# PyMC3のGraphing Models: https://docs.pymc.io/api/model_graph.html<br>
# このベースとなるgraphvizのUser Guide:https://graphviz.readthedocs.io/en/stable/manual.html<br>
# 下記のg1.render()は，graphvizの文法となる。

# In[ ]:

tree_model = pm.model_to_graphviz(model)
tree_model
#木構造の図を保存したい場合には，次の行のコメントをはずす
#tree_model.render(filename='fig_Bayes_PyMC3_Binomial_TreeModel', format='png') # DOT言語ファイルと画像ファイルの保存


# sample():https://docs.pymc.io/api/inference.html#pymc3.sampling.sample<br>
# draws:ステップ数<br>
# start:確率変数として定義したパラメータの初期値を設定する。MAPは設定法の1つ。設定しない場合には，自動的にランダムな値が設定される。<br>
# step:MCMCにおけるステップ法を設定する。設定しない場合には，自動的に適切な方法が設定される。<br>
# cores:CPUコア数，この数だけMCMCを並列的に実行する。

# In[ ]:


#
# RuntimeError:
#        An attempt has been made to start a new process before the
#        current process has finished its bootstrapping phase.
if __name__ == '__main__':

  with model:
    start = pm.find_MAP()
    step = pm.Metropolis() #離散型変数
    trace = pm.sample(draws=1000, tune=100, start=start, step=step, cores=4)


# pymc3.plots.traceplot: https://docs.pymc.io/api/plots.html

# In[ ]:


    pm.traceplot(trace, figsize=(10,4), legend=True)
#plt.savefig('fig_Bayes_PyMC3_ex1.png', bbox_inches='tight') #xlabeの欠落を防ぐ方法2
    plt.show()


# In[ ]:


    print(trace['theta'])
    len(trace['theta'])


# https://docs.pymc.io/api/plots.html

# In[ ]:


#pm.plot_posterior(trace, ref_val=0.3)
    pm.plot_posterior(trace)
#plt.savefig('fig_Bayes_PyMC3_ex1_posterior.png', bbox_inches='tight') #xlabeの欠落を防ぐ方法2


# HPD:https://docs.pymc.io/api/stats.html

# In[ ]:


    print(pm.summary(trace))


# In[ ]:




