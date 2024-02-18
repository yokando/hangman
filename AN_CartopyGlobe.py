#!/usr/bin/env python
# coding: utf-8

# # 回転する地球儀のアニメーション
# ## cartopy
# HP: https://scitools.org.uk/cartopy/<br>
# Install: https://scitools.org.uk/cartopy/docs/latest/installing.html  
# 投影：https://scitools.org.uk/cartopy/docs/latest/crs/projections.html

# In[ ]:


# -*- coding: utf-8 -*-
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#
# set the animation.embed_limit rc parameter to a larger value (in MB)
plt.rcParams['animation.embed_limit'] = 50

#
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# get_ipython().run_line_magic('matplotlib', 'nbagg')


# In[ ]:


fig = plt.figure(figsize=(4,4))

def update(k, fig_title):
    plt.cla()
    ax = plt.axes(projection=ccrs.Orthographic(central_longitude=(-k), central_latitude=0))
    ax.stock_img() # 地球に色を付ける。しかし，かなり遅くなる。
    ax.coastlines()
    plt.title(fig_title)
#

# ani = animation.FuncAnimation(fig, update, fargs=('Globe',), interval=1, frames=270, repeat=False)
# intervalは表示間隔で、単位はミリ秒
# figはinterval=10、fig_2はinterval=1、fig_3はinterval=100（時間がかかりすぎて途中で中断）
ani = animation.FuncAnimation(fig, update, fargs=('Globe',), interval=10, frames=270, repeat=False)
HTML(ani.to_jshtml())

#plt.show()
#
# 動画として保存
ani.save('fig_Anima_CartopyEarth.mp4', writer='ffmpeg')

# In[ ]:


plt.savefig('fig_Anima_CartopyEarth.png')


# In[ ]:




