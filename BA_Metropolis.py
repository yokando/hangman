#!/usr/bin/env python
# coding: utf-8

# # メトロポリス（Metropolis）アルゴリズム

# #### 目標分布，標準正規分布
# $$
# f(x) = \frac{1}{\sqrt{2 \pi}} \exp \left( -\frac{x^2}{2} \right)
# $$

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# 目標分布に比例する確率分布
def f(x): 
    return 1/math.sqrt(2*math.pi)*math.exp(-x**2/2)


# In[ ]:


np.random.seed(123)
ndata = 1000 # the number of data
nskip = 10 # skip
loopmax = ndata*nskip
theta = -10 # 初期値
sdata = np.zeros(ndata) # サンプルデータ格納用の配列
cnt = 0  # カウンタ
for k in range(loopmax):
    # Proposal distribution
    xi = stats.uniform.rvs(loc=-1, scale=2, size=1) # 提案分布 範囲[loc, loc+scale]
    theta_new = theta + xi
    alp = min(1, f(theta_new)/f(theta))
    r = np.random.rand()
    if r > alp:
        theta_new = theta
    theta = theta_new
    if k%nskip==0:
        sdata[cnt] = theta
        cnt += 1


# In[ ]:


plt.plot(sdata[0:500], color='k')
plt.title("Chain")
plt.xlabel('k')
plt.ylabel('Posterior')
#plt.savefig('fig_Bayes_Metropolis01.png')
plt.show()


# #### Pythonとカーネル密度推定(KDE)について調べたまとめ
# KDE関数を提供しているのは，SciPy, seaborn, pandas, statsmodels, sklearnなど幾つかある。<br>
# https://vaaaaaanquish.hatenablog.com/entry/2017/10/29/181949<br>
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html<br>
# https://seaborn.pydata.org/generated/seaborn.kdeplot.html<br>
# https://ja.wikipedia.org/wiki/カーネル密度推定
# 

# In[ ]:


kde = stats.gaussian_kde(sdata)


# ヒストグラムでdensity=Trueは，確率密度関数に合せて全面積を1に正規化する。<br>
# https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.hist.html

# In[ ]:


X = np.arange(-5.0, 5.0, 0.1)
Y = stats.norm.pdf(X, 0.0, 1.0)
plt.plot(X, Y, linewidth=2, color='k', label='target')
plt.plot(X, kde(X), linestyle='dashed', color='b', label='kde')
plt.hist(sdata, bins=50, density=True, color=(0.6, 0.6, 0.6), label='sdata')
plt.title("Metropolis")
plt.xlabel('theta')
plt.ylabel('f(theta)')
plt.legend()
plt.grid(True)
print("end")
# plt.savefig('fig_Bayes_Metropolis02.png')
plt.show()


# In[ ]:




