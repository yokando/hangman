#!/usr/bin/env python
# coding: utf-8

# # 実行時の注意
# JupyterLabを起動してからアニメーションが現れないときは，いったん終了して，<br>
# Jupyter notebookを起動してから１セル枚に実行してみください。<br>
# 2020年9月現在，JupyterLabのグラフィックス表示機能やIPythonはまだ完全に作動していないようです。

# # VPythonを用いたアニメーション
# ビリヤードの衝突問題

# 座標系：右手系<br>
# 左右はx軸（右方向がプラス，左方向はマイナス），奥行きはz軸（遠方方向がマイナス），高さはy軸<br>
# テーブル面はxz平面，テーブル面のy座標は0<br>
# テーブル面の端の中心のxz座標は(0,0)とする<br>
# VPythonのpos.x, pos.y, pos.z -> Length, Height, Width

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import vpython as vs

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Create Scene
scene = vs.canvas(width=600, height=400, title='Animation') # Enable to restart
scene.camera.pos  = vs.vector(0, 6, 10)
scene.camera.axis = vs.vector(0, -2, -8) - scene.camera.pos

# 床の作成，床の真ん中を原点とする
floor = vs.box(length=40, height=0.8, width=60, color=vs.color.green)
floor.pos = vs.vector( 0, -(floor.height/2), -floor.width/2)

# 球の作成
ball_radius = 0.5 # 半径

# 的球の初期位置のx座標
b2_x_init = 2*ball_radius - ball_radius  # theta2 -> 30 [deg]
#b2_x_init = 2*ball_radius - 0.59*ball_radius # theta2 -> 45 [deg]
#b2_x_init = 2*ball_radius - 0.0076*ball_radius # almost theta2 -> 85 [deg]

# 手球b1, 的球b2
b1 = vs.sphere(pos=vs.vector(0, ball_radius, 0), radius=ball_radius, \
               color=vs.color.white)
b2 = vs.sphere(pos=vs.vector(b2_x_init, ball_radius, -8), radius=ball_radius, \
               color=vs.color.red)
# 各パラメータ
v1, v2 = 0.4, 0   # 初期速度
theta1, theta2=0, 0 # 初期角度
c_rest = 0.8 # 反発係数，coefficient of restitution
flag = False  # 衝突判定フラグ，Trueは衝突検知

def ball_step(ball, v, theta):
    coef1 = 1
    x = coef1*v*np.sin(theta)
    z = -coef1*v*np.cos(theta) # 奥行きがマイナスｚ軸方向ゆえ"-"がつく
    ball.pos += vs.vector(x, 0, z)

def check_collision():
#def check_collision(b1, b2, flag):
    global b1, b2, flag, v1, v2, theta1, theta2, c_rest
    dx = b1.pos.x - b2.pos.x
    dz = b1.pos.z - b2.pos.z
    d = np.sqrt(dx*dx + dz*dz)
    if d <= (b1.radius + b2.radius):
        flag = True
        theta2 = np.arcsin(-(b1.pos.x-b2.pos.x)/(2*b1.radius) )
        theta1 = theta2 - np.pi/2
        v2 = 0.5*v1*(1.0 + c_rest)
        v1 = 0.5*v1*(1.0 - c_rest)
    
    
for k in range(100):
    vs.sleep(0.01)
    ball_step(b1, v1, theta1)
    ball_step(b2, v2, theta2)
    if not flag:
        check_collision()


# In[ ]:


# https://www.glowscript.org/docs/VPythonDocs/canvas.html
#アニメーション図の保存
#scene.capture('fig_AN_Billiard.png')


# #### 的球の進路の計算
# $ x = 2r \sin \theta_2 $より，$d = 2r \left( 1 - \sin \theta_2 \right)$ を得る。このグラフを次に示す。

# In[ ]:


theta_2 = np.arange(0, 90+1, 5)
r = 1
d = 2*r*(1-np.sin( theta_2*np.pi/180 ))
plt.plot(d, theta_2, c='k')
plt.xlim(0, 2)
plt.ylim(0, 90)
plt.grid()
plt.xlabel('$d$')
plt.ylabel('$ \\theta_2 $ [deg]')
#plt.savefig('fig_Anima_Billi_Relation.png')


# In[ ]:


for i in range(len(d)):
    print(theta_2[i], d[i])


# In[ ]:




