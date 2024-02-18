#!/usr/bin/env python
# coding: utf-8

# # 捕食・被捕食モデル(predator–prey)，Lotka–Volterra equations

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np  
from scipy.integrate import solve_ivp  

import matplotlib.pyplot as plt  

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def odePre(t, y, a, b, c, d):
    dx =  a*y[0] - b*y[0]*y[1]
    dy = -c*y[1] + d*y[0]*y[1]
    return [dx, dy]


# In[ ]:


a, b, c, d = 2, 1, 3, 1
x0, y0 = 5, 1
Tend = 10.0

sol = solve_ivp(fun=odePre, t_span=[0, Tend], y0=[x0, y0], args=[a, b, c, d], dense_output=True) 
tc = np.linspace(0, Tend, 100)  
yc = sol.sol(tc)


# In[ ]:


fig, axs = plt.subplots(ncols=2, figsize=(14,4))
axs[0].plot(tc, yc[0].T, label='x', c='k')
axs[0].plot(tc, yc[1].T, label='y', c='b')
axs[0].set_xlabel('t')
axs[0].legend()
axs[0].grid()

axs[1].plot(yc[0].T, yc[1].T, c='k')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].grid()
meanx = c/d
meany = a/b
axs[1].scatter(meanx, meany, s=30, c='r', marker='o')

#plt.savefig('fig_NS_PredatorPrey_01.png', bbox_inches='tight')
plt.show()


# ## 釣りを考慮した例

# In[ ]:


a, b, c, d = 2, 1, 3, 1
eps = 0.1
Tend = 10.0
x0, y0 = 5, 1


# In[ ]:


def odePre2(t, y, a, b, c, d, eps):
    dx =  a*y[0] - b*y[0]*y[1] - eps*y[0]
    dy = -c*y[1] + d*y[0]*y[1] - eps*y[1]
    return [dx, dy]


# In[ ]:


sol2 = solve_ivp(fun=odePre2, t_span=[0, Tend], y0=[x0, y0], args=[a, b, c, d, eps], dense_output=True) 
tc = np.linspace(0, Tend, 100)  
yc2 = sol2.sol(tc)


# In[ ]:


fig, axs = plt.subplots(ncols=2, figsize=(14,4))
axs[0].plot(tc, yc2[0].T, label='x', c='k')
axs[0].plot(tc, yc2[1].T, label='y', c='b')
axs[0].set_xlabel('t')
axs[0].legend()
axs[0].grid()

axs[1].plot(yc2[0].T, yc2[1].T, c='k')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].grid()

meanx = (c+eps)/d
meany = (a-eps)/b
axs[1].scatter(meanx, meany, s=30, c='r', marker='o')


#plt.savefig('fig_NS_PredatorPrey_02.png', bbox_inches='tight')
plt.show()


# In[ ]:




