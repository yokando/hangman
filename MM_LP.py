#!/usr/bin/env python
# coding: utf-8

# # LP　( Linear Programming)

# In[ ]:


# -*- coding: utf-8 -*- 
import numpy as np
import pulp

import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')


# #### 例題
# 目的関数
# $$
# \arg \,\,\min \limits_{x,y} \,f\left( {x,y} \right) =  - 4x + y
# $$
# 
# 制約条件
# $$
# \mathrm{subject}\,\mathrm{to}\,\,\left\{ {\begin{array}{*{20}{c}}
# 	{0 \le x \le 3}\\
# 	{0 \le y \le 1}\\
# 	{x + y \le 2}
# 	\end{array}} \right.
# $$

# In[ ]:


prob = pulp.LpProblem("myProblem", pulp.LpMinimize)
x = pulp.LpVariable("x", 0, 3, cat="Continuous")
y = pulp.LpVariable("y", 0, 1, cat="Continuous")
prob += x + y <= 2
prob += -4*x + y
#status = prob.solve(GLPK(msg = 0))
status = prob.solve()
print(pulp.LpStatus[status]) #Display the status of the solution
print(prob)
#print pulp.value(x)
#print pulp.value(y)
print("x=", x.value())
print("y=", y.value())
print("f(x,y)= %f" % prob.objective.value())


# In[ ]:





# #### 例1　ミックスジュース

# In[ ]:


prob = pulp.LpProblem("Juice", pulp.LpMaximize)
x1 = pulp.LpVariable("x1", 0, None, cat="Integer")
x2 = pulp.LpVariable("x2", 0, None, cat="Integer")
x3 = pulp.LpVariable("x3", 0, None, cat="Integer")

prob += 40*x1 + 5*x2 + 20*x3 <= 2000
prob += 30*x1 + 50*x2 + 10*x3 <= 3000
prob += 5*x1 + 10*x2 + 50*x3 <= 1000
prob += 300*x1 + 280*x2 + 420*x3

status = prob.solve()
print(pulp.LpStatus[status])
print("x1=", x1.value())
print("x2=", x2.value())
print("x3=", x3.value())
print("f(x)= %f" % (pulp.value(prob.objective)))


# #### 例2：輸送問題

# In[ ]:


prob = pulp.LpProblem("Transpose", pulp.LpMinimize)
x1 = pulp.LpVariable("X1", 0, None, cat="Integer")
x2 = pulp.LpVariable("X2", 0, None, cat="Integer")
x3 = pulp.LpVariable("X3", 0, None, cat="Integer")
x4 = pulp.LpVariable("X4", 0, None, cat="Integer")
x5 = pulp.LpVariable("X5", 0, None, cat="Integer")
x6 = pulp.LpVariable("X6", 0, None, cat="Integer")

prob += x1 + x4 == 18
prob += x2 + x5 == 14
prob += x3 + x6 == 10
prob += x1 + x2 + x2 <= 20
prob += x4 + x5 + x6 <= 22
prob += 125*x1 + 160*x2 + 175*x3 + 145*x4 + 92*x5 + 125*x6

status = prob.solve()
print(pulp.LpStatus[status])
print("x1=", x1.value())
print("x2=", x2.value())
print("x3=", x3.value())
print("x4=", x4.value())
print("x5=", x5.value())
print("x6=", x6.value())
print("f(x)= %f" % (pulp.value(prob.objective)))

