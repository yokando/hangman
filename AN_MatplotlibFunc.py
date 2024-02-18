#!/usr/bin/env python
# coding: utf-8

# # 実行時の注意
# JupyterLabを起動してからアニメーションが現れないときは，いったん終了して，<br>
# Jupyter notebookを起動してから１セル毎に実行してみください。<br>
# 2020年9月現在，JupyterLabのグラフィックス表示機能やIPythonはまだ完全に作動していないようです。

# # matplotlibを用いたアニメーション
# 内容：Axesのレンジ（範囲）を固定して，波形が流れるようなアニメーション<br>
# matplotlib.animation.FuncAnimation: https://matplotlib.org/api/_as_gen/matplotlib.animation.FuncAnimation.html

# #### 注意：
# このNotebookの実行は、1セル毎のstep by stepとしてください。アニメーションの実行でcallbackの解除などの問題があるため、"Run All"は描画が正しく行われないことがあります。

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#
from matplotlib.animation import FuncAnimation

# get_ipython().run_line_magic('matplotlib', 'nbagg')


# #### plt.cla(), plt.clf(), plt.close()の違い
# https://matplotlib.org/api/pyplot_api.html <br>
# plt.cla()   # Clear Axes，現在のFigureの現在のアクティブなAxesをクリアする。<br>
# plt.clf()   # Clear Figure，現在の図形全体をクリアする。<br>
# plt.close() # Clear a Figure window，現在のグラフ用ウィンドウをクローズする。<br>

# ### FuncAnimationの使用
# Doc:https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.animation.FuncAnimation.html<br>

# In[ ]:


fig = plt.figure(figsize=(6,2))
x = np.arange(0, 10, 0.01)

def update(k, fig_title, Amp):
    plt.cla()       # Figureの中のアクティブなAxesをクリア, eg. plt.clf() clear figure.
    y = Amp*np.sin(x-k)
    plt.plot(x, y, 'r')
    plt.xlim(0, 10)
    plt.ylim(-1.2*Amp, 1.2*Amp)
    plt.title(fig_title + 'Frame k=' + str(k))

# interval [ms],  frames: 描くフレーム数, repeat: Falseは1回のみ，Trueは繰返しプロット
ani = animation.FuncAnimation(fig, update, fargs = ('Animation, ', 1.8), \
    interval = 100, frames = 32, repeat=False)
#
#plt.show()
ani.save('fig_AN_MatplotlibFunc01.mp4', writer='ffmpeg')

# In[ ]:

plt.savefig('fig_AN_MatplotlibFunc01.png')

# In[ ]:


#plt.clf()


# In[ ]:


fig = plt.figure(figsize=(6,2))

xstart, xend, xdt = 0, 10, 0.05
x = np.arange(xstart, xend, xdt)

def update2(k, fig_title, coef):
    plt.cla()    # Axesをクリア , Figureの中のアクティブなAxesをクリア
    x = np.arange(xstart+k*xdt, xend+k*xdt, xdt)
#    x = x+xdt  #これにすると描画が行われない。
    y = np.exp( -coef*(np.sin(x)**2))*np.cos(x)
    plt.plot(x, y, c='r')
    plt.title(fig_title + 'Frame k=' + str(k))
    plt.ylim(-1.2, 1.2)


ani = animation.FuncAnimation(fig, update2, fargs = ('Animation  ', 5.0), \
    interval = 10, frames = 150, repeat=False)

#
#plt.show()
ani.save('fig_AN_MatplotlibFunc02.mp4', writer='ffmpeg')



# In[ ]:


plt.savefig('fig_AN_MatplotlibFunc02.png')


# In[ ]:


#plt.close()


# In[ ]:




