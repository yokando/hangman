#!/usr/bin/env python
# coding: utf-8

# # ロジスティック方程式のカオス性
# https://ja.wikipedia.org/wiki/ロジスティック写像

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


a_list = [1.25, 2.85, 3.1, 3.5, 3.7, 4]
num = 60
x0 = 0.01
xarr = np.zeros((6,num))

for i in range(6):
    xarr[i,0] = x0
    a = a_list[i]
    for k in range(1,num):
        xarr[i,k] = a*xarr[i,k-1]*(1-xarr[i,k-1])


# In[ ]:


fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(14,12))
#titles=['a='+str(a[0]),'Norm(2,1)','Binom(10,0.3)','Poisson(4)']

for i in range(6):
    ncol=i//2; nrow=i%2 # 切捨て除算，剰余
    ax[ncol, nrow].plot(xarr[i,1:], color='k')
    ax[ncol, nrow].set_title('a='+str(a_list[i]))

#plt.savefig('fig_NS_Chaos_LogisticBehavior.png', bbox_inches='tight')
plt.show()


# ## 初期値への敏感性

# In[ ]:


a = 4.0
xarr2 = np.zeros((2,num))
x0_list = [0.01, 0.010001]
for i in range(2):
    xarr2[i,0] = x0_list[i]
    for k in range(1,num):
        xarr2[i,k] = a*xarr2[i,k-1]*(1-xarr2[i,k-1])

fig, ax = plt.subplots(ncols=2, figsize=(14,4))
for i in range(2):
    ax[i].plot(xarr2[i,1:], color='k')
    ax[i].set_title('x0='+str(x0_list[i]))

    
#plt.savefig('fig_NS_LogisticInitVal.png', bbox_inches='tight')
plt.show()


# ## ロジスティック写像（Logistic map）
# https://qiita.com/jabberwocky0139/items/33add5b3725204ad377f
# 
# http://nworks.hateblo.jp/entry/2016/12/04/152143
# https://ja.wikipedia.org/wiki/ロジスティック写像

# In[ ]:


fig = plt.subplots(figsize=(7,4))

n = 200
npick = 50
for a in np.linspace(1, 4, 1000):
    x = [0.01]
    for i in range(n):
        x.append(a * x[i] * (1-x[i]))
    plt.plot([a]*npick, x[-npick:], 'b.', markersize=0.5)

plt.xlabel('a')
plt.ylabel('x[npick]')
plt.title('n='+str(n)+'  npick='+str(npick))

#plt.savefig('fig_NS_LogisticMap.png', bbox_inches='tight')
plt.show()


# In[ ]:


fig = plt.subplots(figsize=(7,6))

n = 800
npick = 200
for a in np.linspace(3.5, 4, 1000):
    x = [0.01]
    for i in range(n):
        x.append(a * x[i] * (1-x[i]))
    plt.plot([a]*npick, x[-npick:], 'c.', markersize=0.2, alpha=0.5)

plt.xlabel('a')
plt.ylabel('x[npick]')
plt.title('n='+str(n)+'  npick='+str(npick))

#plt.savefig('fig_NS_LogisticMap2.png', bbox_inches='tight')
plt.show()


# In[ ]:




