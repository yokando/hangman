#!/usr/bin/env python
# coding: utf-8

# # ベイズ統計、確率密度関数の推定
# コイン投げの例<br>
# 第1回目の観測：$N=5, k=3$

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np
import sympy  # 数式処理パッケージ　https://www.sympy.org/en/index.html
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# 推定する確率変数pの変数定義
p = sympy.Symbol('p')
likelihood = p**3 * (1-p)**2


# In[ ]:


# 正規化定数を求めるための積分
const = sympy.integrate(likelihood, (p, 0, 1))
const


# In[ ]:


#事後分布を求め，区間[0,1]での面積（＝全確率）を求める
pos_1 = likelihood/const
print(pos_1)
all_prob = sympy.integrate(pos_1, (p,0,1))
print(all_prob)


# In[ ]:


fig, axs = plt.subplots(figsize=(10, 3))
prob = np.linspace(0, 1, 100)
# SymPyのsubs()メソッド、変数に値を代入 https://note.nkmk.me/python-sympy-factorization-solve-equation/
plt.plot(prob, [pos_1.subs(p, i) for i in prob])
plt.xlabel('p')
plt.ylabel('posterior')
#ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
axs.set_xticks(np.arange(0,1.1,0.1).tolist())
plt.minorticks_on()
plt.grid(which='both') # both be major or minor, see matplotlib.pyplot.grid
#splt.savefig('fig_Bayes_Coin_Distribution.png', bbox_inches='tight')
plt.show()


# In[ ]:


# MAP(Maximum a posterior)を求める
eq2 = sympy.diff(likelihood)
print(eq2)
eq3 = sympy.factor(eq2)
print(eq3)
sol = sympy.solve(eq3, p)
print(sol)


# In[ ]:


#事後分布の平均値を求める
mean_val = sympy.integrate(p * pos_1, (p,0,1))
print(mean_val,'=', float(mean_val))


# In[ ]:


# 0.5 <= p <= 0.7　の確率を求める
sympy.integrate(pos_1, (p,0.5,0.7))


# ### 次の観測
# $N=9, k=5$　とする。次の事後分布を表す関数pos_2(p)は正規化されていないことに注意されたい。

# In[ ]:


def pos_2(p):
    return p**(3+5) * (1-p)**(2+4)

fig, axs = plt.subplots(figsize=(10, 3))
prob = np.linspace(0, 1, 100)
plt.plot(prob, pos_2(prob))
plt.xlabel('p')
plt.ylabel('posterior')
axs.set_xticks(np.arange(0,1.1,0.1).tolist())
plt.minorticks_on()
plt.grid(which='both') # both be major or minor, see matplotlib.pyplot.grid
#splt.savefig('fig_Bayes_Coin_Distribution.png', bbox_inches='tight')
plt.show()


# #### 事後分布が正規化されていなくても，平均値と最頻値には影響しない。

# In[ ]:


mean_val = 9/(9+7)
mode_val = (9-1)/((9-1)+(7-1))
print(mean_val, mode_val)


# In[ ]:




