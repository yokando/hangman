#!/usr/bin/env python
# coding: utf-8

# # 人口成長，Population growth

# ### 非線形関数のカーブフィッティング（パラメータ推定）
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np  
import scipy.optimize as opt
import matplotlib.pyplot as plt  

# get_ipython().run_line_magic('matplotlib', 'inline')


# #### 計算の簡略化のため，xのデータは西暦でなく，0から始まる整数とする。

# In[ ]:


#1980 -  2010, 5年毎の日本の総人口，単位：千人
y = np.array([117060, 121049, 123611, 125570, 126926, 127768, 128057])
x = np.arange(0,len(y))
# 2012年から総人口は減少に転じている


# In[ ]:


def func(t, p0, pinf, gamma):
    p = pinf/( 1.0 + (pinf/p0 - 1.0)*np.exp(-gamma*t))
    return p


# In[ ]:


popt, pcov = opt.curve_fit(func, x, y, p0=[110000, 130000, 0.6])
popt


# In[ ]:


pest= popt[1]/( 1.0 + (popt[1]/popt[0] - 1.0)*np.exp(-popt[2]*x))

x2 = np.arange(-10, 20, 1)
pest2= popt[1]/( 1.0 + (popt[1]/popt[0] - 1.0)*np.exp(-popt[2]*x2))


# In[ ]:


fig, axs = plt.subplots(ncols=2, figsize=(14,4))

axs[0].plot(x, pest)
axs[0].scatter(x, y)
axs[0].set_xlabel('Year')
axs[0].set_ylabel('P')
axs[0].grid()

axs[1].plot(x2, pest2)
axs[1].set_xlabel('Year')
axs[1].set_ylabel('P')
axs[1].grid()

#plt.savefig('fig_NS_Population_01.png', bbox_inches='tight')
plt.show()


# In[ ]:




