#!/usr/bin/env python
# coding: utf-8

# # 在庫管理

# In[ ]:


# -*- coding: utf-8 -*- 
import numpy as np
import scipy, scipy.stats

import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')


# #### 仕入れ条件

# In[ ]:


Cyear, Cone, Ccost, Csell = 5, 40, 50, 80
LeadTime = 6
alp = 0.05
ka = scipy.stats.norm.ppf(1-alp, loc=0, scale=1)


# #### ある15日間の需要量

# In[ ]:


demand = np.array([3,6,9,10,5,8,2,6,11,3,5,3,6,2,5])
Mean = demand.mean()
Std  = demand.std(ddof=1) # ddof=1 means unbiased std
print(Mean, Std)


# #### シミュレーション条件

# In[ ]:


Kmax = 120 # シミュレーション回数
Invent0 = 220 # 在庫量の初期値
randLow, randHigh = 0, 15


# ### 発注点法，order point system

# In[ ]:


np.random.seed(123)

#
# AttributeError: module 'numpy' has no attribute 'int'.
# OrderingPoint = np.int(LeadTime*Mean+ka*Std*np.sqrt(LeadTime))
# TotalDemand = np.int(Mean*365 + 0.5) # 四捨五入
# EOQ = np.int(np.sqrt( (2*Cone*TotalDemand)/Cyear ) + 0.5)
OrderingPoint = int(LeadTime*Mean+ka*Std*np.sqrt(LeadTime))
TotalDemand = int(Mean*365 + 0.5) # 四捨五入
EOQ = int(np.sqrt( (2*Cone*TotalDemand)/Cyear ) + 0.5)
print(ka,OrderingPoint, TotalDemand, EOQ)


# #### 次のセルだけを実行すると，実行毎に結果が異なる。
# これは，乱数発生の仕組みによる。テキストの10回のシミュレーション結果は，次のセルだけの実行を何回も行った結果である。

# #### アルゴズムの説明
# 初めてx[k] <= OrderingPointとなった日からLeatTime期間は納入できない。<br>
# そこで，この日を起点としてLeadTime期間中であることをflag=Trueが示す。<br>
# lagは，LeaｄTime期間の日のカウンタであり，lag ==0 はLeadTime期間の終了を示す。<br>
# LeadTimeを“今日”を起点とするならば，lag = LeadTime + 1，“明日”を起点とするなばLeadTime + 2となる。

# In[ ]:


x = np.zeros(Kmax) # 在庫量
x[0] = Invent0
flag = False
dsum = 0

for k in range(1,Kmax):
#    d = np.int(max(scipy.stats.norm.rvs(loc = Mean, scale = Std, size=1), 0))
    d = scipy.stats.randint.rvs(low=randLow, high=randHigh, size=1)
    dsum += d
    x[k] = x[k-1] - d
    #初めてx[k] <= OrderingPointとなった日からLeatTime期間は納入できない
    if x[k] <= OrderingPoint and flag == False:
        flag = True
        lag = LeadTime + 1
    if flag == True:
        lag = lag - 1
        if lag == 0:
            x[k] = x[k] + EOQ
            flag = False
print('Total amount dsum = %d' % (dsum))
print('Mean of inventory x = %f' % (x.mean()))

plt.plot(x)
plt.ylim(0, 250)
plt.xlabel('Day')
plt.ylabel('Volume of inventories')
plt.title('Ordering Point Method')
plt.show()


# ### 定期発注法，periodical ordering system

# In[ ]:


np.random.seed(3456) # 発注点法と確率変数のseedを変えることで日々の需要量を異ならせる
# CycleT = np.int(np.sqrt( (2*Cone*TotalDemand)/Cyear )/Mean )
# POS_Q = np.int((CycleT + LeadTime)*Mean + ka*Std*np.sqrt(CycleT + LeadTime) + 0.5)
CycleT = int(np.sqrt( (2*Cone*TotalDemand)/Cyear )/Mean )
POS_Q = int((CycleT + LeadTime)*Mean + ka*Std*np.sqrt(CycleT + LeadTime) + 0.5)

print(CycleT, POS_Q)


# #### 次のセルだけを実行すると，実行毎に結果が異なる。
# これは，乱数発生の仕組みによる。テキストの10回のシミュレーション結果は，次のセルだけの実行を何回も行った結果である。

# In[ ]:


y = np.zeros(Kmax)
y[0] = Invent0
count = CycleT
dsum = 0

for k in range(1,Kmax):
#    d = np.int(max(scipy.stats.norm.rvs(loc = Mean, scale = Std, size=1), 0))
    d = scipy.stats.randint.rvs(low=randLow, high=randHigh, size=1)
    dsum += d
    y[k] = y[k-1] - d
    count = count - 1
    if count == 0:
        y[k] = POS_Q
        count = CycleT

print('Total amount dsum = %d' % (dsum))
print('Mean of inventory ypo = %f' % (y.mean()))

plt.plot(y)
plt.ylim(0, 250)
plt.xlabel('Day')
plt.ylabel('Volume of inventories')
plt.title('Regular Order Method')
plt.show()


# #### 1回目だけ，2つのグラフを描画するためのスクリプト

# In[ ]:


plt.plot(x, c='k', label='OR')
plt.plot(y, c='b', linestyle='dashed', label='PO')

plt.ylim(0, 250)
plt.xlabel('Day')
plt.ylabel('Volume of inventories')
plt.legend()
plt.title('Order Point(line) and Periodical Ordering(dashed)')

#plt.savefig('fig_MM_Inventory_Comparison.png', bbox_inches='tight')
plt.show()


# In[ ]:




