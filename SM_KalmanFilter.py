#!/usr/bin/env python
# coding: utf-8

# # 離散時間線形のカルマンフィルタ
# 1次系と2次系のシミュレーション

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


np.random.seed(123)
# System Information
a, b, c = 1, 1, 1
sigma_v = 1.0
sigma_w = np.sqrt(2.0)

# Initial values
x = 0.0
xhat = 0.0
P = 0.001

num = 500 # the number of simulation step
x_data = np.zeros(num)
y_data = np.zeros(num)
xhat_data = np.zeros(num)
P_data    = np.zeros(num)

x_data[0] = x
xhat_data[0] = xhat
P_data[0] = P

for k in range(1, num):
    # System
    x = a*x + stats.norm.rvs(loc=0.0, scale=sigma_v, size=1)
    y = c*x + stats.norm.rvs(loc=0.0, scale=sigma_w, size=1)
    x_data[k] = x
    y_data[k] = y
# Kalman Filter
    # Prediction
    xhat = a*xhat
    P = a*P*a + (sigma_v**2)*(b*b)
    P_data[k] = P
    # Filterling
    g = P*c/(c*P*c+sigma_w**2)
    xhat = xhat + g*(y-c*xhat)
    xhat_data[k] = xhat
    P = (1 - g*c)*P


# In[ ]:


fig, ax = plt.subplots(ncols=2, figsize=(14,4))

nn = 50

ax[0].plot(x_data[:nn], c='k', label='x')
ax[0].plot(y_data[:nn], c='c', label='y', linestyle='dashed')
ax[0].plot(xhat_data[:nn], c='r', label='xhat')
ax[0].set_xlabel('k')
ax[0].legend()

ax[1].plot( (x_data[:nn]-xhat_data[:nn]), c='k', label='x - xhat')
ax[1].set_xlabel('k')
ax[1].legend()
ax[1].grid()

#plt.savefig('fig_SM_KalmanFilter_1st.png', bbox_inches='tight')
plt.show()


# In[ ]:


nn=20
plt.plot(P_data[:nn])
plt.xlabel('k')
plt.ylabel('P')
plt.xticks(np.arange(0,nn+1,5))

#plt.savefig('fig_SM_KalmanFilter_1st_gain.png', bbox_inches='tight')
plt.show()


# In[ ]:


#定常ゲインを見る
print(g)


# ## 2次系

# #### 対象システムの出力$y(k)$を先に計算して，配列に格納する

# In[ ]:


np.random.seed(123)

A = np.array([[-1.2, -0.52], [1.0, 0.0]])
b = np.array([[1.0], [0.0]]) # column vector
c = np.array([[1.0], [0.0]]) # column vector
sigma_v = np.sqrt(2)
sigma_w = 1.0

# Initial values
x = np.array([[0.5],[1.0]])

num = 200
x0 = np.zeros(num) # store x[0]
x0[0] = x[0]
y = np.zeros(num)
y[0] = np.dot(c.T, x) + stats.norm.rvs(loc=0.0, scale=sigma_w, size=1)

g0 = np.zeros(num) # store g[0]
g1 = np.zeros(num) # store g[1]

for k in range(1,num):
    x = np.dot(A,x) + b*stats.norm.rvs(loc=0.0, scale=sigma_v, size=1)
    x0[k] = x[0]
    y[k] = np.dot(c.T, x) + stats.norm.rvs(loc=0.0, scale=sigma_w, size=1)


# #### カルマンフィルタ

# In[ ]:


# Initial values
P  = np.eye(2,2)*0.1
UnitMat = np.eye(2,2) # Unit Matrix
xhat = np.array([[0.0], [0.0]])
xhat_data = np.zeros(num)
xhat_data[0] = xhat[0]

for k in range(1, num):
    # Prediction
    xhat = np.dot(A, xhat)
    P = np.dot( np.dot(A,P), A.T) + (sigma_v**2)*np.dot(b, b.T)
    # Filterling
    g = np.dot(P,c)/(np.dot( np.dot(c.T,P),c)  + (sigma_w**2) )
    g0[k] = g[0]
    g1[k] = g[1]
    xhat = xhat + g*(y[k] - np.dot(c.T,xhat))
    xhat_data[k] = xhat[0]
    P = np.dot( (UnitMat - np.dot(g,c.T)), P)


# In[ ]:


fig, ax = plt.subplots(ncols=2, figsize=(14,4))

nn = 20
ax[0].plot(x0[:nn], c='k', label='x[0]')
ax[0].plot(y[:nn], c='c', label='y', linestyle='dashed')
ax[0].plot(xhat_data[:nn], c='r', label='xhat')
ax[0].set_xlabel('k')
ax[0].set_xticks(np.arange(0,nn+1,5))
ax[0].legend()

ax[1].plot( (g0[:nn]), c='k', label='g[0]')
ax[1].plot( (g1[:nn]), c='c', label='g[1]')
ax[1].set_xlabel('k')
ax[1].set_ylabel('gain')
ax[1].set_xticks(np.arange(0,nn+1,5))
ax[1].legend()

#plt.savefig('fig_SM_KalmanFilter_2nd.png', bbox_inches='tight')
plt.show()


# #### 行列Aの固有値を調べる。

# In[ ]:


eig_val, eig_vec = np.linalg.eig(A)
print(eig_val)
print(abs(eig_val))

