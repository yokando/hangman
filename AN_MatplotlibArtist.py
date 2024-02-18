#!/usr/bin/env python
# coding: utf-8

# # 実行時の注意
# JupyterLabを起動してからアニメーションが現れないときは，いったん終了して，<br>
# Jupyter notebookを起動してから１セル毎に実行してみてください。<br>
# 2020年9月現在，JupyterLabのグラフィックス表示機能やIPythonはまだ完全に作動していないようです。

# # matplotlibを用いたアニメーション
# matplotlib.animationを用いる<br>
# ドキュメント https://matplotlib.org/api/animation_api.html<br>
# ここに，2種のアニメーション方法があり，ここでは1番目のArtistAnimationを説明する。<br>
# 1. 全部描画したフレームを複数枚、順次表示する。<br>
# ArtistAnimationを用いる。<br>
# https://matplotlib.org/api/_as_gen/matplotlib.animation.ArtistAnimation.html<br>
# 2. 逐次的にプロット<br>
# FuncAnimationを用いる。<br>
# https://matplotlib.org/api/_as_gen/matplotlib.animation.FuncAnimation.html<br>
# 

# ### 幾つかのアニメーション基本文法

# #### マジックコマンド nbagg
# IPython Notebook上でインタラクティブな画像表示を実現するための機能 https://matplotlib.org/3.1.3/users/prev_whats_new/whats_new_1.4.html<br>
# マジックコマンドリスト：https://ipython.readthedocs.io/en/stable/interactive/magics.html<br>
# マジックコマンド一覧の表示：%lsmagic

# #### plt.cla(), plt.clf(), plt.close()の違い
# https://matplotlib.org/api/pyplot_api.html <br>
# plt.cla()   # Clear Axes，現在のFigureの現在のアクティブなAxesをクリアする。<br>
# plt.clf()   # Clear Figure，現在の図形全体をクリアする。<br>
# plt.close() # Clear a Figure window，現在のグラフ用ウィンドウをクローズする。<br>

# #### ArtistAnimationとFuncAnimationの引数について共通的なものの説明
# interval: 描画に関する手続き（関数呼び出しなど）間のインターバル時間[ms]<br>
# frames:描画するフレームの数<br>
# repeat:Trueはフレームの最後を描画した後に繰返す。Falseは繰返さない<br>
# blit:Trueでオン，Falseでオフ。CGでは古くからある描画の高速化手法である。幾つかの要件があり，また，コールバックの仕方などが複雑になる<br>
#  ref. https://matplotlib.org/3.2.1/api/animation_api.html<br>
#  ref. Animating selected plot elements in SciPy Cookbook, https://scipy-cookbook.readthedocs.io/items/Matplotlib_Animations.html

# ### アニメーションの保存
# この機能は照会だけに留め，ここでは用いない。作成したアニメーションをgifや mp4に保存できる。ただし，それぞれ外部ソフトウェア（例えば，gifの場合はImagemagickなど）を用意する必要がある。
# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.animation.Animation.html#matplotlib.animation.Animation.save

# ## 例：マークのランダム発生
# 注意：このNotebookの実行は、1セル毎のstep by stepとしてください。アニメーションの実行でcallbackの解除などの問題があるため、"Run All"は描画が正しく行われないことがあります。

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# get_ipython().run_line_magic('matplotlib', 'nbagg')


# #### matplotlib.pyplot.plot 
# https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.plot.html<br>
# matplotlib.markers<br>
# https://matplotlib.org/api/markers_api.html#module-matplotlib.markers

# In[ ]:


fig = plt.figure()
imgBuffer = []

mark_list = ["4", "8", "p", "*", "x", "D", "4", "8", "p", "*"]
loop = len(mark_list)
num = 8

for i in range(loop):
    x = np.random.randn(num)
    y = np.random.randn(num)
    img = plt.plot(x, y, linewidth=0, marker=mark_list[i], markersize=15)
    imgBuffer.append(img)


# ts [ms]毎にnum毎のグラフの表示を切替えるアニメーション
ts = 300
ani = animation.ArtistAnimation(fig, imgBuffer, interval=ts, repeat=False)
plt.show()


# In[ ]:


#plt.savefig('fig_AN_MatplotlibArtist.png')


# In[ ]:




