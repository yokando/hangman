#!/usr/bin/env python
# coding: utf-8

# # 実行時の注意
# JupyterLabを起動してからアニメーションが現れないときは，いったん終了して，<br>
# Jupyter notebookを起動してから１セル枚に実行してみください。<br>
# 2020年9月現在，JupyterLabのグラフィックス表示機能やIPythonはまだ完全に作動していないようです。

# # VPythonを用いたアニメーション
# 立方体・バネ系，微分方程式を用いず，単なる単振動運動とした。<br>

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np
import vpython as vp

# get_ipython().run_line_magic('matplotlib', 'inline')


# #### VPython 文法
# ・ワールド座標系は右手系<br>
# ・canvas: あるキャンバス（sceneとも言う）を作成して，この中でアニメーション用のオブジェﾄを描画する。呼出す毎に新たなキャンバスが作成されるので，アニメーションのリスタート時にこれを呼ぶこと<br>
# ・https://www.glowscript.org/docs/VPythonDocs/canvas.html<br>
# ・box:箱を描く，https://www.glowscript.org/docs/VPythonDocs/box.html<br>
# ・helix:バネを描く，https://www.glowscript.org/docs/VPythonDocs/helix.html<br>
# ・color: https://www.glowscript.org/docs/VPythonDocs/color.html<br>

# #### アニメーションの設計
# ・視点はデフォルトのままとする。すなわち，右が+x, 上が+y，奥行きが-z軸方向<br>
# ・立方体の底面の中心を原点（0,0,0）に一致させ，これに適するように床，壁，スプリングを配置する。

# In[ ]:


# キャンバスの作成
scene = vp.canvas(width=600, height=300, title='Cube-Spring') # Enable to restart
# 3次元空間内の原点を見るために，球を原点に配置。
orig = vp.sphere(pos=vp.vector(0,0,0), radius=0.1, color=vp.color.red)
# オブジェクトの作成
# 立方体
cube_size = 1
cube = vp.box(size=vp.vector(cube_size, cube_size, cube_size), color=vp.color.orange)
cube.pos = vp.vector(0, cube.height/2, 0)
# 床
floor = vp.box(length=5.0, height=0.1, width=cube_size+0.2, color=vp.color.green)
floor.pos = vp.vector(0, -floor.height/2, 0)
# 壁
wall  = vp.box(length=0.1, height=1.5, width=floor.width, color=vp.color.yellow)
wall.pos = vp.vector((-wall.length/2-floor.length/2), (wall.height/2 - floor.height) , 0)

#equi_length = box_size/2+floor.height/2 #バネの自然長
#wall_surface_pos = cube_size/2+floor.height/2

# バネ
spring_pos_wall = vp.vector((wall.pos.x+wall.length/2), cube.pos.y, cube.pos.z )
spring_pos_cube = vp.vector( (cube.pos.x-cube.length/2), cube.pos.y, cube.pos.z)
spring = vp.helix(pos=spring_pos_wall, axis=(spring_pos_cube - spring_pos_wall), 
                  radius=0.2,     # バネ径の半径
                  thickness=0.05, # バネ寸法
                  coils=8,        # バネ巻数
                  color=vp.vector(0, 1, 1) # cyan
                 )

def func_pos(k):
    return np.sin(k*0.1)

for k in range(100):
    vp.sleep(0.1)
    x_pos = func_pos(k)
    cube.pos = vp.vector(x_pos, cube.pos.y, 0)
    spring_pos_cube = vp.vector( (cube.pos.x-cube.length/2), cube.pos.y, cube.pos.z)
    spring.axis = spring_pos_cube - spring_pos_wall


# In[ ]:


# https://www.glowscript.org/docs/VPythonDocs/canvas.html
#アニメーション図の保存
#scene.capture('fig_AN_CubeSpring.png')


# In[ ]:




