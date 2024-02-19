#!/usr/bin/env python
# coding: utf-8

# # 待ち行列 M/M/1
# Wikipedia M/M/1 queue:https://en.wikipedia.org/wiki/M/M/1_queue
# SimPy: https://simpy.readthedocs.io/

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np
import simpy
import random

import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')


# ### 評価指標
# $\lambda$: 平均到着人数(mean rate of arrival)[人/時間]<br>
# $1/\lambda$: 平均到着時間間隔[時間/人]<br>
# $\mu$: 平均サービス件数(mean rate of service)[人/時間]<br>
# $1/\mu$: 平均サービス時間[時間/人]<br>
# $\rho$: 利用率, $\rho > 1(\lambda > \mu)$の場合，待ち行列は収束しない。
# $$
# \rho = \frac{\lambda}{\mu}
# $$
# $L$:平均系内客数(mean number of customers in the system)
# $$
# L = \frac{\rho}{1-\rho}
# $$
# $L_q$:平均待ち行列客数(mean number of customers in the queue)
# $$
# L_q = \rho L
# $$
# $W$:平均系内滞在時間(mean waiting time in the system)
# $$
# W = \frac{L}{\lambda} = \frac{\rho}{\lambda(1-\rho)} = \frac{1}{\mu - \lambda}
# $$
# $W_q$:平均待ち時間(mean waiting time in the queue)
# $$
# W_q = \frac{1}{\mu} L
# $$
# このシミュレーションでは，$L=L_q$, $W=W_q$と考える。
# 

# In[ ]:


#Lambda, Mu = 0.8 , 1.0
Lambda, Mu = 1/8, 1/4
rho = Lambda/Mu
L = rho/(1 - rho)
W = 1/(Mu - Lambda)
print('rho =',rho,' L = ',L,' W = ',W)


# In[ ]:


# 変数や配列の設定
numq = 0  #待ち行列人数の初期値
flag_do = True # 関数queue()のwhileブロックで処理する間，他からの処理を重複して実行しないようにするためのフラグ
qtime_arrv = [] #客が到着したときの時刻のキュー（queue）, FIFO方式

list_stay_time = [] # 滞在時間
list_t = [] # 客の到着時間
list_q = [] # 待ち行列の人数


# In[ ]:


# 到着イベント
def arrive():
    global qtime_arrv, numq, flag_do
    while True:
        yield env.timeout(random.expovariate(Lambda)) # 平均到着時間間隔
        envnow = env.now # 現在のシミュレーション時間を取得（単位時間のため，単位は無い）
        qtime_arrv.append(envnow)
        list_t.append(envnow)
        list_q.append(numq)
        numq += 1 # 待ち行列に追加
        if(flag_do):
            env.process(queue())

# 待ち行列に並ぶ
def queue():
    global qtime_arrv, numq, flag_do, list_stay_time
    flag_do = False
    while len(qtime_arrv) > 0:
        yield env.timeout(random.expovariate(Mu)) # 平均サービス時間
        list_stay_time.append(env.now - qtime_arrv[0])
        numq -= 1 # 待ち行列から出る
        qtime_arrv = qtime_arrv[1:] # リストの左シフトにより出た人の到着時刻をqueueから除く, FIFO
    flag_do = True


# In[ ]:


# シミュレーション環境の作成
random.seed(123)
Stime = 10**6 # シミュレーション時間
env = simpy.Environment()
env.process(arrive()) # ジェネレータarrive()を配置
env.run(until = Stime)  # 実行

print('calculated L =',np.sum(list_stay_time)/Stime, ' W =',np.mean(list_stay_time))
print('Clients:',len(list_stay_time))
#print('Min queueing:',min(list_q), ', Max queueing:',max(list_q))
print('Min queueing:',min(list_q), ', Max queueing:',max(list_q), ', No queuing:', list_q.count(0))
#print('Approximate simulattion time:',len(list_t)*(1/Lambda))


# In[ ]:


fig, axs = plt.subplots(ncols=2, figsize=(14,4))
axs[0].scatter(list_t, list_q, marker='.', s=5, c='k')
axs[0].set_xlabel('time')

start, end = 0, 100
axs[1].scatter(list_t[start:end], list_q[start:end], marker='.', s=10, c='k')
axs[1].set_xlabel('time')

#plt.savefig('fig_MM_Queueing_MM1_results.png', bbox_inches='tight')
plt.show()


# #### 初めの5人の到着時刻と待ち行列人数

# In[ ]:


print(list_t[:5])
print(list_q[:5])


# #### 最後の5人の到着時刻と待ち行列人数

# In[ ]:


print(list_t[-5:])
print(list_q[-5:])

