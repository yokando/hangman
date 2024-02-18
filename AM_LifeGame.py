#!/usr/bin/env python
# coding: utf-8

# # ライフゲーム
# 
# ・離散的環境で動作するエージェントのシミュレーション<br>
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


# #### 初期設定
# 
# フィールドの大きさと初期状態でセルを黒にする確率

# In[2]:

#
# TypeError: 'NoneType' object is not callable


ROWS = 20
COLS = 20
p=0.3


# #### ライフゲームのクラス
# 
# ・checkメソッド：周囲の生存セルを数える<br>
# ・evolutionメソッド：周囲の状態から次の状態を決める

# In[3]:


class LifeGame():
    def __init__(self):
        self.field = np.random.choice([0,1],[ROWS,COLS],p=[1-p,p])
    print("start0")
    def check(self, y, x):
        # 周囲の生存セルを数える
        cnt = 0
        tbl = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
        for t in tbl:
            xx, yy = [x + t[0], y + t[1]]
            if xx==COLS:xx=0
            if yy==ROWS:yy=0
            if self.field[yy][xx]: cnt += 1
        return cnt
    print("start1")
    def evolution(self, t=-1):
        if t == 0:return self.field
        next_field = np.zeros([ROWS,COLS], dtype=int)
        for y in range(ROWS):
            for x in range(COLS):
                n = self.check(y,x)
                s = self.field[y,x]
                if s==0:
                    if n==3:
                        next_field[y,x] = 1
                    else:
                        next_field[y,x] = 0
                else:
                    if n<=1 or n>=4:
                        next_field[y,x] = 0
                    else:
                        next_field[y,x] = 1
        self.field = next_field
        return self.field
    print("start2")


# #### ライフゲームの実行
# 
# ・表示の設定<br>
# ・シミュレーションの実行（for文）<br>
# ・アニメーションの表示

# In[4]:


np.random.seed( 0 )
L = LifeGame()
print("start3")

figsize =  (6,6*ROWS/COLS) if ROWS<COLS else (6*COLS/ROWS,6)
makersize = 300/COLS if ROWS<COLS else 300/ROWS
fig, ax = plt.subplots(1, 1, figsize=figsize)
ax.set_xlim(-0.5,COLS-0.5)
ax.set_ylim(-0.5,ROWS-0.5)
for i in range(COLS+1):
    ax.plot((i-0.5,i-0.5),(-0.5,ROWS-0.5),color='lightgrey', linewidth=0.5)
for i in range(ROWS+1):
    ax.plot((-0.5,COLS-0.5),(i-0.5,i-0.5),color='lightgrey', linewidth=0.5)
ims = []
print("start5")

for t in range(100):
    field_data = L.evolution(t)
    plot_data = np.array(np.where(field_data == 1))
    im=ax.plot(plot_data[1],plot_data[0],marker="s",linestyle="none", ms=makersize,c="k")
    ims.append(im)
print("start6")
ani = animation.ArtistAnimation(fig, ims, interval=200, repeat=False)
HTML(ani.to_jshtml())
print("start7")

#ani.save("LifeGame.gif")#,writer="imagemagick")
ani.save("LifeGame.gif")#,writer="imagemagick")
