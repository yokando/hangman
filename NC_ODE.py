#!/usr/bin/env python
# coding: utf-8

# # solve_ivpを用いた常微分方程式の解法
# solve_ivp:Solving initial value problems<br>
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html<br>
# 解法のうちRadauの説明：https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np  
from scipy.integrate import solve_ivp  
import matplotlib.pyplot as plt  

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def dFunc(time, x, mass, damper, spring, u):
    dx0 = x[1]
    dx1 = (-1/mass)*(damper*x[1] + spring*x[0] - u) 
    return [dx0, dx1]


# In[ ]:


x0 = [0.0, 0.0]
u = 1.0
mass, damper, spring = 4.0, 1.0, 1.0
T_END = 20


# In[ ]:


# Solve ODE  
sol = solve_ivp(fun=dFunc, t_span=[0, T_END], y0=x0, method='RK45', args=[mass, damper, spring, u], dense_output=True)  


# In[ ]:


print(type(sol))
print(sol)


# In[ ]:


#時間の出力
print(sol.t.size)
print(sol.t)


# In[ ]:


#yの出力
print(sol.y[0].size, sol.y[1].size)
print(sol.y)


# In[ ]:


tc = np.linspace(0, T_END, 100)  
yc = sol.sol(tc)
print(len(tc), len(yc))


# In[ ]:


fig, ax = plt.subplots(ncols=2, figsize=(12,4))

ax[0].plot(sol.t, sol.y.T)
ax[0].set_xlabel('t')
ax[0].set_ylabel('x(t), v(t)')
ax[0].grid()

ax[1].plot(tc, yc[0].T, label='x')
ax[1].plot(tc, yc[1].T, label='v')
ax[1].set_xlabel('t')
ax[1].set_ylabel('x(t), v(t)')
ax[1].grid()
ax[1].legend()


plt.savefig('fig_NC_ODE.png')
# plt.show()


# In[ ]:




