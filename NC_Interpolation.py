#!/usr/bin/env python
# coding: utf-8

# # 補間

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import scipy.stats

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# numpyのみならずscipyの確率の関数の初期値を定める
np.random.seed(123)


# カーブフィッティング，多項式使用

# In[ ]:


num = 20
x = np.arange(0,num,1)
y = 1.2*x + sp.stats.uniform(loc=-10.0, scale=10.0).rvs(num)

plt.scatter(x, y, color='k')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')

plt.savefig('fig_NC_Interpolation_01.png')


# #### カーブフィッティング
# 多項式を用いる。関数内を他の非線形関数に置き換えることも可能である。

# In[ ]:


p5  = np.polyfit(x, y, deg=5)
p15 = np.polyfit(x, y, deg=13)
xx = np.linspace(np.min(x), np.max(x), 200)
y5 = np.polyval(p5, xx)
y15 = np.polyval(p15, xx)


# In[ ]:


plt.scatter(x, y, color='k')
plt.plot(xx, y5,  label='deg=5')
plt.plot(xx, y15, label='deg15')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.savefig('fig_NC_Interpolation_02.png')


# #### 付録：ルンゲ現象
# https://en.wikipedia.org/wiki/Runge%27s_phenomenon

# In[ ]:


num = 11
x2 = np.linspace(-1, 1, num)
y2 = (2/(1 + 16 * x2 ** 2)) - 1

p = np.polyfit(x2, y2, deg=10)
xx = np.linspace(-1, 1, 200)
yy = np.polyval(p, xx)

plt.scatter(x2, y2, c='k')
plt.plot(xx, yy, c='g')

plt.xlabel('x')
plt.ylabel('y')
plt.grid()


# In[ ]:




