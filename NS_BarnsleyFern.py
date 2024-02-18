#!/usr/bin/env python
# coding: utf-8

# # バーンスレイのシダ，The Barnsley fern
# 
# Wiki
# https://en.wikipedia.org/wiki/Barnsley_fern<br>

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.transforms as trans

from IPython.display import Image

# get_ipython().run_line_magic('matplotlib', 'inline')


# ### シダの写真
# Wikimedia Commons, https://commons.wikimedia.org/wiki/Main_Page　からキーワード"Fern"で検索したURLを下記で用いている。<br>
# この作品は次にある。<br>
# https://commons.wikimedia.org/wiki/File:Ostrich_fern_at_Myrstigen_trail_1.jpg

# In[ ]:


url = 'https://upload.wikimedia.org/wikipedia/commons/d/db/Ostrich_fern_at_Myrstigen_trail_1.jpg'
Image(url, width=300, height=300)


# In[ ]:


f1 = lambda x,y: (0., 0.16*y)
f2 = lambda x,y: (0.85*x + 0.04*y, -0.04*x + 0.85*y + 1.6)
f3 = lambda x,y: (0.2*x - 0.26*y, 0.23*x + 0.22*y + 1.6)
f4 = lambda x,y: (-0.15*x + 0.28*y, 0.26*x + 0.24*y + 0.44)
fs = [f1, f2, f3, f4]


# numpy.random.choice: https://docs.scipy.org/doc//numpy-1.10.4/reference/generated/numpy.random.choice.html

# In[ ]:


num = 50000
# Canvas size (pixels)
width, height = 300, 300
image = np.zeros((width, height))
image2 = np.zeros((width, height))

x, y = 0, 0
for i in range(num):
    # Pick a random transformation and apply it
    f = np.random.choice(fs, p=[0.01, 0.85, 0.07, 0.07])
    x, y = f(x,y)
    # Map (x,y) to pixel coordinates.
    # NB we "know" that -2.2 < x < 2.7 and 0 <= y < 10
    if abs(x) >= 2.7:
        print('over x')
    if abs(y) >= 10.0:
        print('over y')
    ix, iy = int(width / 2 + x * width / 10), int(y * height / 12)
    # Set this point of the array to 1 to mark a point in the fern
    image[iy, ix] = 1
    image2[ix, iy] = 1   


# #### 緑色を使用するためmatplotlib.cmを使用
# https://matplotlib.org/3.2.2/api/cm_api.html

# #### スライスシング start, end, step
# https://qiita.com/okkn/items/54e81346d8f35733ab5e<br>
# https://deepage.net/features/numpy-slicing.html

# 絵の天地を逆向きにするためにマイナスstepのスライシングを用いている

# In[ ]:


fig = plt.subplots(figsize=(6,6))
plt.imshow(image[::-1,:], cmap=cm.Greens)
#plt.savefig('fig_NS_BarnsleyFern_01.png', bbox_inches='tight')
plt.show()


# In[ ]:


fig = plt.subplots(figsize=(6,6))
plt.imshow(image2, cmap=cm.Greens)
plt.show()


# In[ ]:


fig = plt.subplots(figsize=(6,6))
plt.imshow(image, cmap=cm.Greens)
plt.show()


# In[ ]:


plt.imshow(image[0:60,130:190], cmap=cm.Greens)


# In[ ]:


from scipy import ndimage

rotated_img = ndimage.rotate(image[0:60,130:190], 180)
plt.imshow(rotated_img[:,::-1], cmap=cm.Greens)
#plt.savefig('fig_NS_BarnsleyFern_02.png', bbox_inches='tight')
plt.show()


# In[ ]:




