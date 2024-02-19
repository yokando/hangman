#!/usr/bin/env python
# coding: utf-8

# # 経営モデル，簡単な例

# #### リボ払い（revolving payment)
# この英訳は，https://support.rakuten-card.jp/faq/show/15101?category_id=6&site_domain=guest#:~:text=Revolving%20payment%20means%20that%20you,on%20the%20balance%20each%20month.\n"<br>
# 払い方は幾つか種類がある。https://www.smbc-card.com/nyukai/magazine/knowledge/revo_3method.jsp<br>
# 定額払いを例にとる
# 
# <br>
# 計算の評価は次を参照：<br>
# AEON CARD, ご返済シミュレーション結果：https://www.aeon.co.jp/NetBranch/cashingSimRevoInit.do
# 

# In[ ]:


# -*- coding: utf-8 -*- 
import numpy as np
import scipy.optimize

import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


debt = 200000   # 借入金額
inter_a = 0.15      # 金利／年　annual interest rate of 15 percent
inter_m = inter_a/12      # 金利／月 monthly interest
repay = 5000     # 月々の返済金 monthly repayment

num = 500  # 500か月分の残高のデータの記録が可能
zandaka   = np.zeros(num)
risokubun = np.zeros(num)
gankinbun = np.zeros(num)


# In[ ]:


sum = repay   # 返済金の合計
month = 0

zandaka[month] = debt
risokubun[month] = 0
gankinbun[month] = 0

while debt > 0: # 借入金があるならば（＞０）
    month = month + 1
#
# AttributeError: module 'numpy' has no attribute 'int'.
#    repay_i = np.int(debt*inter_m) # 今月の利息払い分
    repay_i = int(debt*inter_m) # 今月の利息払い分
    ret = repay - repay_i             # 返済金のうち元金分　（利息分を引く）
    debt = debt - ret            # 借入残高

    # グラフ用
    zandaka[month] = debt
    risokubun[month] = repay_i
    gankinbun[month] = ret
    print("month: %d   RisokuBun: %d  GankinBun: %d  Zandaka: %d" % (month, repay_i, ret, debt))

    if debt < repay :  #借入残高が返済金より少なくなった場合
#
# AttributeError: module 'numpy' has no attribute 'int'.
#        inter = np.int(debt*inter_m)
        inter = int(debt*inter_m)
        ret = debt + inter
        debt = 0
        sum = sum + ret
        month = month + 1
        
        # グラフ用
        zandaka[month] = debt
        risokubun[month] = inter
        gankinbun[month] = repay
#        print("month: %d   RisokuBun: %d  GankinBun: %d  Zandaka: %d" % (month, repay_i, ret, debt))

        break    # while文を脱出
    sum = sum + repay
    
print("Total month : %d  Amount paid: %d" % (month, sum))


# In[ ]:


fig, axs = plt.subplots(ncols=2, figsize=(14, 4))
axs[0].plot(zandaka[1:month], c='k', label='zandaka')
axs[0].set_xlabel('month')
axs[0].grid()
axs[0].legend()

axs[1].plot(risokubun[1:month], c='r', label='risokubun')
axs[1].plot(gankinbun[1:month], c='g', label='gankinbun')
axs[1].set_xlabel('month')
axs[1].grid()
axs[1].legend()

#plt.savefig('fig_MM_SimpleEx_Revo.png')
plt.show()


# In[ ]:





# #### 損益分岐点
# ・https://ja.wikipedia.org/wiki/損益分岐点<br>
# より<br>
# 横軸：販売数量，縦軸：費用，売上高<br>
# 
# 固定費を次の３つに分類<br>
# ・能力費は、リース料、減価償却費、固定資産税、保険料などのように物的設備の導入に伴って発生する費用です。<br>
# ・組織費は、人件費など人事計画に伴って発生する費用です。<br>
# ・政策費は、広告宣伝費、試験研究費などで経営者の意思決定に伴って一定額が発生する費用です。<br>

# #### 非線形変動費
# ・https://core.ac.uk/download/pdf/228685601.pdf<br>
# ・西川，他: 非線形損益分岐点分析に関する考察, 20(1), 59/65, 日本経営システム学会誌(2003)
# 

# 固定費：fixed cost
# 変動費：variable cost
# 非線形変動費：nonlinear variable cost

# In[ ]:


x = np.arange(10)
cf = np.ones(10)*10
cv = 2*x
cn = np.array([0, 1.5, 2.8, 2.9, 1.6, 0.8, 1.4, 2.6, 3.2, 3.4])*5
cost = cf+cv+cn
print(cost)


# #### 下記のparmに多項式の係数が格納されている。

# In[ ]:


parm = np.polyfit(x, cost, 3)

fig, axs = plt.subplots(ncols=2, figsize=(14,5))

axs[0].plot(x, cf, c='b', marker='x', label='fixed')
axs[0].plot(x, cv, c='g', marker='^', label='variable')
axs[0].plot(x, cn, c='r', marker='D', label='nonlinear')
axs[0].plot(x, cost, c='k', marker='o', label='cost')
axs[0].set_xlabel('x')
axs[0].set_ylim([0,50])
axs[0].legend()
axs[0].grid()


axs[1].scatter(x, cost, c='k', label='cost')
axs[1].plot(x, np.poly1d(parm)(x), c='b', label='f(x)')
axs[1].set_xlabel('x')
axs[1].set_ylim([0,50])
axs[1].legend()
axs[1].grid()

#plt.savefig('fig_MM_SimpleEx_BEP_01.png')
plt.show()


# In[ ]:





# #### 損益分岐点，例1
# 求めたparmに基づく多項式と，y = a xの交点（分岐点）を求める。初めに、aを適当に与える。

# In[ ]:


a = 5.0

fig = plt.subplots(figsize=(6,4))

plt.scatter(x, cost, c='k', label='cost')
plt.plot(x, np.poly1d(parm)(x), c='b', label='f(x)')
plt.plot(x, a*x, c='g', label='sales')
plt.xlabel('x')
plt.ylim([0,50])
plt.legend()
plt.grid()

#plt.savefig('fig_MM_SimpleEx_BEP_02.png')
plt.show()


# ### 交点を求める。

# #### 多項式の根，fsolve
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html<br>

# In[ ]:


def fun1(x):
    return np.poly1d(parm)(x) - a*x

root = scipy.optimize.fsolve(fun1, x0=[6, 8])
print(root)


# ### 接点を求める。

# #### 連立非線形方程式，root
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html<br>
# default method is "hybr", 修正 Powell Hybrid 法，ヤコビ行列を作成して用いている。<br>
# これはMINPACK（非線形方程式のライブラリ）を用いている。<br>

# In[ ]:


def fun2(x):
    f = [  parm[0]*x[0]**3 + parm[1]*x[0]**2 + parm[2]*x[0] + parm[3] - x[1]*x[0],
         3*parm[0]*x[0]**2 + 2*parm[1]*x[0] + parm[2] - x[1]]
    return f
result = scipy.optimize.root(fun2, x0=[7, 5])
print(result)
print(result.x)


# In[ ]:


fig = plt.subplots(figsize=(6,4))

plt.scatter(x, cost, c='k', label='cost')
plt.plot(x, np.poly1d(parm)(x), c='b', label='f(x)')
plt.plot(x, result.x[1]*x, c='g', label='sales')
plt.xlabel('x')
plt.grid()
plt.legend()
plt.ylim([0,50])


#plt.savefig('fig_MM_SimpleEx_BEP_03.png')
plt.show()


# In[ ]:




