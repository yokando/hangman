#!/usr/bin/env python
# coding: utf-8

# # SimPy
# SimPy HP:https://simpy.readthedocs.io/<br>
# env.timeout: https://simpy.readthedocs.io/en/latest/api_reference/simpy.events.html<br>
# env.nou: https://simpy.readthedocs.io/en/latest/api_reference/simpy.core.html<br>
# 
# https://stefan.sofa-rockers.org/downloads/simpy-ep14.pdf<br>
# https://en.wikipedia.org/wiki/SimPy

# #### イベント駆動型プログラムとは何かを知るスクリプト，上記のWikipedia参照

# In[ ]:


# -*- coding: utf-8 -*-
import simpy


# In[ ]:


def clock(env, name, tick): 
    while True:
        print(name, env.now)
        yield env.timeout(tick)


# In[ ]:


env = simpy.Environment()
env.process(clock(env, 'fast', 0.5)) 
env.process(clock(env, 'slow', 1)) 


# In[ ]:


env.run(until=2) 


# ### リストを用いたFIFO

# In[ ]:


q=[0.2, 0.3, 0.5, 0.8]
q.append(1.1)
q = q[1:]
q


# In[ ]:




