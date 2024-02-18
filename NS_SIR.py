#!/usr/bin/env python
# coding: utf-8

# # SIRモデル
# https://ja.wikipedia.org/wiki/SIRモデル<br>
# https://www.ms.u-tokyo.ac.jp/~inaba/inaba_science_2008.pdf

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np  
from scipy.integrate import solve_ivp  

import matplotlib.pyplot as plt  

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def odeSIR(t, y, beta, gamma):
    dS = -beta*y[0]*y[1]
    dI =  beta*y[0]*y[1] - gamma*y[1]
    dR =  gamma*y[1]
    return [dS, dI, dR]


# In[ ]:


beta, gamma = 0.01, 0.5
Tend = 10.0
coef = 1 # 1, 3, 5
S0 = coef*gamma/beta
I0 = 10
R0 = 0

sol = solve_ivp(fun=odeSIR, t_span=[0, Tend], y0=[S0, I0, R0], args=[beta, gamma], dense_output=True) 


# In[ ]:


#print(sol)
#print(sol.t.size)


# In[ ]:


#yの出力
print(sol.y[0].size, sol.y[1].size)
print(sol.y)


# In[ ]:


# 連続値を取得する　今回は区間を100分割してる  
tc = np.linspace(0, Tend, 100)  
yc = sol.sol(tc)


# In[ ]:


print(len(tc), len(yc))


# In[ ]:





# In[ ]:


plt.plot(tc, yc.T)


# In[ ]:


S0 = 1*gamma/beta
sol1 = solve_ivp(fun=odeSIR, t_span=[0, Tend], y0=[S0, I0, R0], args=[beta, gamma], dense_output=True) 
S0 = 3*gamma/beta
sol3 = solve_ivp(fun=odeSIR, t_span=[0, Tend], y0=[S0, I0, R0], args=[beta, gamma], dense_output=True) 
S0 = 5*gamma/beta
sol5 = solve_ivp(fun=odeSIR, t_span=[0, Tend], y0=[S0, I0, R0], args=[beta, gamma], dense_output=True) 


# In[ ]:


print(gamma/beta, 3*gamma/beta, 5*gamma/beta)


# In[ ]:


tc = np.linspace(0, Tend, 100)  
yc1 = sol1.sol(tc)
yc3 = sol3.sol(tc)
yc5 = sol5.sol(tc)


# In[ ]:


fig, axs = plt.subplots(nrows=2, figsize=(10,6))

axs[0].plot(tc, yc1[0].T, label='S0=gamma/beta', c='k')
axs[0].plot(tc, yc3[0].T, label='S0=3*gamma/beta', c='b')
axs[0].plot(tc, yc5[0].T, label='S0=5*gamma/beta', c='r')
axs[0].set_xlabel("$t$")  
axs[0].set_ylabel("$S$")
axs[0].grid()
axs[0].legend()

axs[1].plot(tc, yc1[1].T, label='S0=gamma/beta', c='k')
axs[1].plot(tc, yc3[1].T, label='S0=3*gamma/beta', c='b')
axs[1].plot(tc, yc5[1].T, label='S0=5*gamma/beta', c='r')
axs[1].set_xlabel("$t$")  
axs[1].set_ylabel("$I$")
axs[1].grid()
axs[1].legend()

#plt.savefig('fig_NS_SIR_01.png', bbox_inches='tight')
plt.show()


# In[ ]:


fig = plt.subplots(figsize=(7,5))
plt.plot(yc1[0].T, yc1[1].T, label='S0=gamma/beta', c='k')
plt.plot(yc3[0].T, yc3[1].T, label='S0=3*gamma/beta', c='b')
plt.plot(yc5[0].T, yc5[1].T, label='S0=5*gamma/beta', c='r')
plt.xlabel("$S$")  
plt.ylabel("$I$")
plt.grid()
plt.legend()

#plt.savefig('fig_NS_SIR_02.png', bbox_inches='tight')
plt.show()


# In[ ]:





# In[ ]:


beta = 0.01
gamma = 0.5

I0 = 10
R0 = 0
color = ['k', 'b', 'r']
color_id = 0


for coef in [1, 3, 5]:
    S0 = coef*(gamma/beta)
    s_val = np.arange(1., S0, 0.1)
    const = (S0+I0 - (gamma/beta)*np.log(S0))
    print(coef, S0, const)
    i_val = -s_val + (gamma/beta)*np.log(s_val) + const
    plt.plot(s_val, i_val, label='coef='+str(coef), c=color[color_id])
    color_id+=1

plt.xlabel('$S$')
plt.ylabel('$I$')
plt.ylim(0.0, 140.)
plt.grid()
plt.legend()

plt.vlines(x=gamma/beta, ymin=0, ymax=140, linestyle='dashed')


#plt.savefig('fig_NS_SIR_Analysis.png', bbox_inches='tight')
plt.show()


# In[ ]:




