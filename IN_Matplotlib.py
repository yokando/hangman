#!/usr/bin/env python
# coding: utf-8

# # matplotlibを用いて複数のプロット，グラフを描く
# subplotsのパラメータ:<br>
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html <br>
# 
# このパラメータのうち，更なるもの（figsize[inch, inch], facecolor, edgecolorなど）は次：<br>
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.figure.html<br>
# 

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


x = np.linspace(-3, 3, 20)
y1 = x
y2 = x ** 2
y3 = x ** 3
y4 = x ** 4


# In[ ]:


fig = plt.subplots(figsize=(10,3)) # size [inch,inch]
plt.plot(x,y2, c='k', label='y2')
plt.plot(x,y3, c='r', label='y1', linestyle='dashed')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Example')
plt.grid()
plt.legend(loc='lower right')

#plt.tight_layout() # xlabelの欠落を防ぐ方法１
#plt.savefig('fig_Intro_MultipleGraph_1.png', bbox_inches='tight')
plt.show()


# #### マルチプロット　１番目の方法

# In[ ]:


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
 
axs[0,0].plot(x, y1, label='y1') # upper left
axs[0,1].plot(x, y2) # upper right
axs[1,0].plot(x, y3) # lower left
axs[1,1].plot(x, y4) # lower right

axs[0,0].set_xlabel('x')
axs[0,0].set_ylabel('y')
axs[0,0].grid()
axs[0,0].legend()


#plt.savefig('fig_Intro_MultipleGraph_2.png', bbox_inches='tight')
plt.show()


# #### マルチプロット　2番目の方法

# In[ ]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(6,4))

ax1.plot(x, y1) # upper left
ax2.plot(x, y2) # upper right
ax3.plot(x, y3) # lower left
ax4.plot(x, y4) # lower right
 
#plt.savefig('fig_Intro_MultipleGraph_3.png', bbox_inches='tight')
plt.show()


# ## 文字コードの説明
# https://docs.python.org/2.4/lib/standard-encodings.html<br>
# PEP 263 -- Defining Python Source Code Encodings<br>
# https://www.python.org/dev/peps/pep-0263/

# ## Built-in magic commands
# https://ipython.readthedocs.io/en/stable/interactive/magics.html

# In[ ]:


# get_ipython().run_line_magic('lsmagic', '')


# In[ ]:




