#!/usr/bin/env python
# coding: utf-8

# # ポアソン到着モデルのシミュレーション
# ポアソン分布, $t=1$とおく
# $$
# f(k; \lambda) = \exp (-\lambda) \frac{(\lambda )^k}{k !}
# $$
# 指数分布
# $$
# f(x; \lambda) = \lambda \exp(-\lambda x)\, , \,\, x \ge 0 \\
# E[x] = 1/\lambda
# $$
# scipy.stats.poisson<br>
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html<br>
# scipy.stats.expon<br>
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html<br>
# stats.expon(scale, loc)の意味，ディフォルト loc=0, scale=1
# $$
# y = \frac{(x-\mathrm{loc})}{\mathrm{scale}}
# $$

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')


# #### 指数分布のプロット
# 
# $\lambda$の値による場合分け<br>

# In[ ]:


np.random.seed(123)
lamb = [1, 2, 4]
num = 1000
x = np.linspace(0.01, 10.0, num)

fig, axs = plt.subplots(ncols=3, figsize=(14,4))

for i in range(3):
    y_pdf = stats.expon(scale=1/lamb[i]).pdf(x)
    y_rvs = stats.expon(scale=1/lamb[i]).rvs(size=num)
    axs[i].plot(x,y_pdf, c='k', label='pdf')
    axs[i].hist(y_rvs, bins=50, density=True, label='rvs', color='c', alpha=0.3)
    print(y_rvs.max(), y_rvs.min(), y_rvs.mean())
    axs[i].set_xlabel('x (lamb='+str(lamb[i])+')')
    axs[i].set_xlim(0,5)
    axs[i].grid()
    axs[i].legend()

#plt.savefig('fig_MM_QueueingExponent.png', bbox_inches='tight')
plt.show()


# In[ ]:


Num=20 # the number of arrival, Num人分の到着時刻を得る


# In[ ]:


np.random.seed(123)
lamb = 2

t_interval = stats.expon(scale=1/lamb).rvs(size=Num) # 到着時間の間隔
t_arrival  = np.zeros(Num) # 到着時間

t_arrival[0] = t_interval[0]
for i in range(1, len(t_interval)):
    t_arrival[i] = t_arrival[i-1] + t_interval[i]

fig = plt.subplots(figsize=(6,3))
plt.vlines(t_arrival, ymin=0, ymax=1)
plt.xlim(0,10)
plt.xlabel('time[k]')

#plt.savefig('fig_MM_QueueingExponentArrival.png', bbox_inches='tight')
plt.show()


# In[ ]:


np.max(t_arrival)


# In[ ]:


np.random.seed(123)
lamb = 3

t_interval = stats.expon(scale=1/lamb).rvs(size=Num) # 到着時間の間隔
t_arrival  = np.zeros(Num) # 到着時間

t_arrival[0] = t_interval[0]
for i in range(1, len(t_interval)):
    t_arrival[i] = t_arrival[i-1] + t_interval[i]

fig = plt.subplots(figsize=(6,3))
plt.vlines(t_arrival, ymin=0, ymax=1)
plt.xlim(0,10)
plt.xlabel('time[k]')

#plt.savefig('fig_MM_QueueingExponentArrival.png', bbox_inches='tight')
plt.show()


# In[ ]:





# In[ ]:




