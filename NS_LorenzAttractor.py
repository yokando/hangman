#!/usr/bin/env python
# coding: utf-8

# # Lorenz Attractor
# https://en.wikipedia.org/wiki/Lorenz_system<br>
# 
# 
# 差分方程式を用いたプログラムは次を参照：https://matplotlib.org/3.1.0/gallery/mplot3d/lorenz_attractor.html<br>
# This is an example of plotting Edward Lorenz's 1963 "Deterministic Nonperiodic Flow" in a 3-dimensional space using mplot3d.<br>
# https://journals.ametsoc.org/jas/article/20/2/130/16956/Deterministic-Nonperiodic-Flow

# ## 注意：
# グラフの描画の都合上，1セル毎に実行してください。

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


def lorenz(x, y, z, s=10, r=28, b=8/3):
    '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    '''
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return [x_dot, y_dot, z_dot]


# #### 差分方程式で解く

# In[ ]:


dt = 0.01
xyz_ini = [0.0, 1.0, 1.05]
#xyz_ini = [0.0001, 1.0, 1.05]
num_steps = 10000

xs = np.empty(num_steps)
ys = np.empty(num_steps)
zs = np.empty(num_steps)
xs[0], ys[0], zs[0] = xyz_ini # initial values

for i in range(1, num_steps):
    x_dot, y_dot, z_dot = lorenz(xs[i-1], ys[i-1], zs[i-1])
    xs[i] = xs[i-1] + (x_dot * dt)
    ys[i] = ys[i-1] + (y_dot * dt)
    zs[i] = zs[i-1] + (z_dot * dt)


# ### マジックコマンド notebookの説明
# Interactivityを有効にするには，matplotlibのnotebookバックエンドを使用する必要があります。<br>
# %matplotlib notebookを実行することでこれを行うことができます。<br>
# https://ipython.readthedocs.io/en/stable/interactive/magics.html

# In[ ]:

#
# get_ipython().run_line_magic('matplotlib', 'notebook')

# Plot
fig = plt.figure()

# TypeError: FigureBase.gca() got an unexpected keyword argument 'projection'
# ax = fig.gca(projection='3d')
ax = fig.add_subplot(projection='3d')

ax.plot(xs, ys, zs, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()


# In[ ]:


#plt.savefig('fig_NS_LorenzAttractor_Init00.png', bbox_inches='tight')


# In[ ]:





# In[ ]:


fig = plt.subplots(figsize=(6,3))
plt.plot(xs, c='k')
plt.title('Initial value ='+str(xyz_ini))
#plt.savefig('fig_NS_LorenzAttractor_xInit00.png', bbox_inches='tight')
plt.show()


# #### ODE法を用いて解く

# In[ ]:


from scipy.integrate import solve_ivp


# In[ ]:


def lorenz2(t, x, s, r, b):
    x_dot = s*(x[1] - x[0])
    y_dot = r*x[0] - x[1] - x[0]*x[2]
    z_dot = x[0]*x[1] - b*x[2]
    return [x_dot, y_dot, z_dot]


# In[ ]:


xyz0 = xyz_ini # initial values
s, r, b =10, 28, 8/3
T_END = 100
sol = solve_ivp(fun=lorenz2, t_span=[0, T_END], y0=xyz0, method='RK45', args=(10, 28, 8/3,), dense_output=True)
t = np.linspace(0, T_END, 10000)  
xyz = sol.sol(t)
print(len(t), len(xyz))


# In[ ]:


#get_ipython().run_line_magic('matplotlib', 'notebook')

# Plot
fig = plt.figure()
# ax = fig.gca(projection='3d')
ax = fig.add_subplot(projection='3d')


ax.plot(xyz[0], xyz[1], xyz[2], lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()


# In[ ]:





# ### x成分だけの時系列をプロットする
# 1番目は差分方程式の結果

# In[ ]:


# get_ipython().run_line_magic('matplotlib', 'inline')

fig = plt.subplots(figsize=(6,3))
plt.plot(xs, c='k')
plt.title('Initial values='+str(xyz_ini))
#plt.savefig('fig_NS_LorenzAttractor_x.png', bbox_inches='tight')
plt.show()


# 2番目はsolve_ivpの結果

# In[ ]:


fig = plt.subplots(figsize=(6,3))
plt.plot(xyz[0], c='k')
plt.title('Initial values='+str(xyz_ini))
plt.show()


# In[ ]:




