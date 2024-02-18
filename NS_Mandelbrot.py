#!/usr/bin/env python
# coding: utf-8

# # マンデルブロ集合
# 

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def mandelbrot(arr, width, height, xmin, ymin, xcoef, ycoef, maxIt):
    for ky in range(height):
        cy = ycoef*(height-ky) + ymin
        for kx in range(width):
            cx = xcoef*kx + xmin
            c = complex(cx, cy)
            z = complex(0.0, 0.0)
            flag = True
            for i in range(maxIt):
                count = i
                z = z * z + c
                if abs(z) >= 2.0:
                    flag = False
                    break
            if flag:
                arr[ky, kx] = ( int(255), int(255), int(255) )
            else:
                if count <= 1:
                    b_color = 0
                elif count <= 3:
                    b_color = 60
                elif count <= 5:
                    b_color = 150
                else:
                    b_color = 255
                arr[ky, kx] = ( int(0), int(0), int(b_color) )


# In[ ]:


#全体を見るには次の2行を活かし，下の2行をコメントアウト
xmin, xmax, ymin, ymax = -3.0, 1.0, -1.5, 1.5
WIDTH  = 800 ; HEIGHT = 600 # propotional to (x,y) range

#一部を見るには次の2行を活かし，上の2行をコメントアウト
#xmin, xmax, ymin, ymax = 0.1, 0.5, 0.4, 0.8
#WIDTH  = 800 ; HEIGHT = 800 # propotional to (x,y) range

xwidth = xmax - xmin
ywidth = ymax - ymin
maxIt = 256

x_coeff = xwidth/np.float64(WIDTH)
y_coeff = ywidth/np.float64(HEIGHT)
array = np.zeros((HEIGHT, WIDTH, 3), dtype='int32')  # 3 => RGB array
mandelbrot( array, WIDTH, HEIGHT, xmin, ymin, x_coeff, y_coeff, maxIt )


# In[ ]:


plt.imshow(array, vmin=0, vmax=255, interpolation='none')
plt.title('Mandelbrot Set')
plt.xlabel('Re(z)')
plt.ylabel('Im(z)')

X0 = (np.abs(xmin)/xwidth)*WIDTH
plt.xticks([0, X0, WIDTH], [xmin, 0, xmax])

Y0 = (np.abs(ymin)/ywidth)*HEIGHT
plt.yticks([0, Y0, HEIGHT], [ymax, 0, ymin])

#plt.savefig('fig_NS_Mandelbrot_2.png', bbox_inches='tight')
plt.show()


# In[ ]:




