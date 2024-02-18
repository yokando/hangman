#!/usr/bin/env python
# coding: utf-8

# # 実行時の注意
# JupyterLabを起動してからアニメーションが現れないときは，いったん終了して，<br>
# Jupyter notebookを起動してから１セル枚に実行してみください。<br>
# 2020年9月現在，JupyterLabのグラフィックス表示機能やIPythonはまだ完全に作動していないようです。

# # VPythonを用いた単振り子のアニメーション
# 
# ガリレオが発見したとされる振り子の等時性<br>
# https://www2.nhk.or.jp/school/movie/clip.cgi?das_id=D0005300756_00000<br>
# 当時は，糸の長さが長く，少しの振れ角しか与えられなかった。しかも，精度の良い時計が無い時代に，周期性を確かめるために脈を利用するという知恵には敬服するものがある。
# 
# 小学校理科では，振り子の等時性：周期は，おもりの重さや振れ角に依存せず，糸の長さに依存すると説明している。<br>
# https://www.mext.go.jp/a_menu/shotou/new-cs/senseiouen/1304651.html<br>
# 
# しかし，実際には振れ角により周期は変わる。<br>
# このことをアニメーションで示す。

# 座標系：右手系<br>
# 視点（view point）は$(0, y_{v0}, z_{v0})$，$(y_{v0}, z_{v0} > 0)$とする。<br>
# ここを起点とした視線方向を定めるための注視点を$(0, y_{v1}, z_{v1})$，$(0<y_{v1}<y_{v0}, z_{v1}<0)$とする。<br>
# 左右はx軸（右方向がプラス方向），奥行きはz軸（遠方方向がマイナス方向），高さはy軸<br>
# 床面はxz平面，床面のy座標は0<br>
# 床面の端の中心のxz座標は(0,0)とする<br>
# VPythonのpos.x, pos.y, pos.z は次に対応する： Length, Height, Width<br>

# SciPy.orgのIntegratin and ODEs(scipy.ntegrate) https://docs.scipy.org/doc/scipy/reference/integrate.html<br>
# は古典的APIを持つodeint()よりもsolve_ivp()を勧めている。<br>
# scipy.integrate.solve_ivp: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html<br>
# この解法は幾つかあり，中でもRadauが良く用いられている。<br> https://ja.wikipedia.org/wiki/ルンゲ＝クッタ法のリスト<br>
# しかし，slove_ivp()の数値解は，数値だけでなく様々な計算情報も含んでおり，本計算には使いにくいため，odeint()を用いる。これはAPIが古いと言っているだけで，数値計算の能力は高いと述べている。
# 

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np
import vpython as vs
import matplotlib.pyplot as plt 
from scipy.integrate import odeint
from scipy.special import ellipk

# get_ipython().run_line_magic('matplotlib', 'inline')


# #### 単振り子の運動方程式
# Lagrangeの運動方程式より，単振り子の運動方程式は次で与えられる。
# $$
# m  \frac{d^2 l \theta}{dt^2}+ k  \frac{d l \theta}{dt} + mg \sin \theta = 0
# $$
# ここに，$l$: 糸の長さ$[m]$，$m$: 質量$[kg]$，$\theta$: 振れ角[rad]，$g$: 重力加速度$[m/s^2]$, $k$:粘性減衰係数<br>
# $l \theta$は弧の長さを表す。<br>
# ここでは，$k$は無いものとし，次の連立の1階微分方程式をモデルとして用いる。
# $$
# \frac{d}{{dt}}\theta (t) = \omega (t) \\
#  \frac{d}{{dt}}\omega (t)  =  - \frac{g}{l} \sin(t) 
# $$

# In[ ]:


def dFunc(x, time, string_len, g):
    dx0 = x[1]
    dx1 = -(g/string_len)*np.sin(x[0]) # string_len:糸の長さ
    return [dx0, dx1]


# In[ ]:


string1_len = 10
theta1_ini = 20*(np.pi/180) # deg -> rad
v1_ini= 0

string2_len = 10
theta2_ini = 60*(np.pi/180) # deg -> rad
v2_ini = v1_ini


# #### 初めに，odeintを用いて，周期に関する数値解を求める。
# グラフをみて，2つの差を認識しにくい

# In[ ]:


g = 9.80665 # 重力加速度
#time=np.linspace(0,6.9,100)
time = np.arange(0, 10, 0.01)
sol_1 = odeint(dFunc, [theta1_ini, v1_ini], time, args=(string1_len,g,))
sol_2 = odeint(dFunc, [theta2_ini, v2_ini], time, args=(string2_len,g,))


# In[ ]:


plt.plot(time, sol_1[:,0])
plt.plot(time, sol_2[:,0])
plt.legend(["string1", "string2"])
plt.xlabel("time")
plt.ylabel("theta [rad]")


# ##### 2つの波形を見て周期が異なることがわかるが，その差はわずかである。

# #### 次のセルを実行すると，振り子CGの初期画面が静止状態で表示される。

# In[ ]:


# Create Scene
scene = vs.canvas(width=600, height=400, title='Pendulum Animation') # Enable to restart
scene.camera.pos  = vs.vector(0, 6, 20)
scene.camera.axis = vs.vector(0, 2, -100) - scene.camera.pos

floor = vs.box(pos=vs.vector(0,0,0),length=40, height=0.1, width=60, color=vs.color.green)
#floor.pos = vs.vector( 0, -(floor.height/2), floor.width/2) # 床の端の真ん中を原点とする
#print(floor.pos)


origin = vs.sphere(pos=vs.vector(0, 0, 0), radius=0.5, color=vs.color.red) # show original point of 3D space
#org    = vs.sphere(pos=vs.vector(1, 0, 1), radius=0.5, color=vs.color.cyan) # exam. position

bar_height = string1_len+2.0
bar = vs.cylinder(pos=vs.vector(0, bar_height, 0), axis=vs.vector(0,0,-10), radius=0.5, color=vs.color.white)

node1 = vs.vector(0, bar.pos.y, 0)
end1 = vs.vector( string1_len*np.sin(theta1_ini),  bar.pos.y-string1_len*np.cos(theta1_ini), 0)
str1 = vs.cylinder(pos=node1, axis=-(node1-end1), radius=0.02, color=vs.color.yellow)
#m1 = vs.sphere(pos=end1, radius=1, color=vs.color.white)
m1 = vs.cone(pos=end1, axis=vs.vector(0, -1, 0), radius=0.5, color=vs.color.white)

node2 = vs.vector(0, bar.pos.y, -5)
end2 = vs.vector( string2_len*np.sin(theta2_ini),  bar.pos.y-string2_len*np.cos(theta2_ini), node2.z)
str2 = vs.cylinder(pos=node2, axis=-(node2-end2), radius=0.02, color=vs.color.yellow)
#m2 = vs.sphere(pos=end2, radius=1, color=vs.color.white)
m2 = vs.cone(pos=end2, axis=vs.vector(0, -1, 0), radius=0.5, color=vs.color.white)


# In[ ]:


# https://www.glowscript.org/docs/VPythonDocs/canvas.html
#scene.capture('fig_Anima_Pandulum_01.png')


# #### 次のセルを実行すると上に示すアニメーションが開始する。

# In[ ]:


tstart, tend = 0.0, 0.0
dt = 0.1
nstep = 5
theta1 = theta1_ini
v1 = v1_ini

theta2 = theta2_ini
v2 = v2_ini

for k in range(200):
    vs.sleep(0.01)
    tstart = k*dt
    tend = (k+1)*dt
    time = np.linspace(tstart, tend, nstep)
    sol_1 = odeint(dFunc, [theta1, v1], time, args=(string1_len,g,))
    sol_2 = odeint(dFunc, [theta2, v2], time, args=(string2_len,g,))

    #    print(k,sol_1[k][0])
    theta1 = sol_1[nstep-1][0] # angle
    v1     = sol_1[nstep-1][1] # angular velocity

    theta2 = sol_2[nstep-1][0]
    v2     = sol_2[nstep-1][1]
#    print(k,theta1)
    end1 = vs.vector( string1_len*np.sin(theta1),  bar.pos.y-string1_len*np.cos(theta1), 0)
#    print(end1)
    str1.axis = -(node1-end1)
    m1.pos = end1
    
    end2 = vs.vector( string2_len*np.sin(theta2),  bar.pos.y-string2_len*np.cos(theta2), node2.z)
    str2.axis =-(node2-end2)
    m2.pos = end2  


# ### 完全楕円積分（Complete Elliptic Integral）を用いた周期の計算
# ref.<br>
# https://en.wikipedia.org/wiki/Pendulum_(mathematics)<br>
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipk.html

# In[ ]:


deg_list = np.arange(5, 95, 5) # 0 - 90, 5 deg step
periodT = np.zeros(np.size(deg_list))
k = 0
for deg in deg_list:
    periodT[k] = (2/np.pi)*ellipk( np.sin( (deg*np.pi/180)/2. )**2 )
    print(k, deg, periodT[k])
    k += 1


# In[ ]:


plt.plot(deg_list, periodT, marker='x')
plt.xlabel('theta_M [deg]')
plt.ylabel('T')
plt.grid()
#plt.savefig('fig_Anima_Pendulum_Analysis.png')

plt.show()


# In[ ]:




