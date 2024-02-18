#!/usr/bin/env python
# coding: utf-8

# # 注意：
# 本Notebookはアニメーションを用いている。このため，JupyterLabでエラーが生じた場合（IPythonとのインタフェースが不備のもよう，2020年9月時点），いったん，JupyterLabを終了して，Jupyter Notebookを立ち上げてから，再実行してください。

# # モンテカルロ法，円周率を求める
# 実行上の注意：<br>
# ・1セル毎に実行すること

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.animation as animation

# get_ipython().run_line_magic('matplotlib', 'nbagg')


# #### 円周率
# 単位円（unit circle）内に，2次元一様乱数(x,y)がこの中に入る比率を用いる。<br>
# アニメーションを用いて処理速度が遅いので，データ数は高々数百点までが望ましい。

# In[ ]:


np.random.seed(123)
fig, ax = plt.subplots(figsize=(4,4))
ucircle = pat.Circle(xy = (0.0, 0.0), radius = 1.0, edgecolor = 'black', fill=False)
ax.add_patch(ucircle)
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)

count=0

def func_pi(N, fig_title, dummy):
    global count
    x,y = np.random.uniform(low=-1, high=1, size=2)
    judge = x*x + y*y <= 1.0
    if judge:
        plt.scatter(x, y, marker='o', color='green')
    else:
        plt.scatter(x, y, marker='x', color='red')
    count += judge
    pi_est = 4*count/N
    plt.title(fig_title+' N='+str(N)+' est pi='+'{:1.5f}'.format(pi_est))

    
ani = animation.FuncAnimation(fig, func_pi, fargs = ('Monte Carlo',1.0), \
    interval = 10, frames = 500, repeat=False)

plt.show()


# In[ ]:


#plt.savefig('fig_SM_MonteCarlo.png', bbox_inches='tight')


# 円周率を4桁合せるにもデータ数は10**7程度必要<br>
# core i7プロセッサレベルを用いても時間を要する。<br>
# この高速化は，Numbaを用いるかマルチプロセスを用いる方法がある。<br>
# この高速化の詳細は「Pythonデータエンジニアリング入門 高速化とデバイスデータアクセスの基本と応用」（オーム社）参照。

# In[ ]:


print(np.pi)


# In[ ]:


np.random.seed(123)


# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 3 -n 1', "N = 10**8\nx,y = np.random.uniform(low=-1, high=1, size=(2,N))\ninside = (x**2 + y**2) <= 1\npi = inside.sum()*(4/np.float(N))\nprint('N=',N, 'Calculation pi =',pi)\n")


# In[ ]:




