#!/usr/bin/env python
# coding: utf-8

# # 数値を用いた問題（数値最適化）
# 
# ・ある条件の下で評価値を最大(または最小)にする問題の解法
# 
# DEAP
# https://deap.readthedocs.io/en/master/

# In[7]:


from deap import base, creator, tools, algorithms
import numpy as np
import random


# In[8]:


RANGE = 10#初期個体の生成だけでなく突然変異の範囲指定にも使用するため定数で設定


# DEAPを使用するための設定（解析手法と初期値）

# In[9]:


#価値の最大化
creator.create( "Fitness", base.Fitness, weights=(1.0,) )
#遺伝子の各要素に重複を許すさないためsetを設定
creator.create("Individual", list, fitness = creator.Fitness )
 
toolbox = base.Toolbox()
#遺伝子の属性の設定
toolbox.register( "attribute", random.randrange, 0, RANGE  )
#初期個体の生成
toolbox.register( "individual", tools.initRepeat, creator.Individual, toolbox.attribute, 2 )
#初期個体群を作成
toolbox.register( "population", tools.initRepeat, list, toolbox.individual )


# 評価関数

# In[10]:


def myEvaluation( individual ):
    x0 = individual[0]
    x1 = individual[1]
   
    if 2*x0+2*x1>7:
        return -100,
    if 3*x0+5*x1>14:
        return -100,
    
    return 4*x0+5*x1,


# DEAPを使用するための設定（評価，選択，交叉，突然変異）

# In[11]:


toolbox.register("evaluate", myEvaluation)
toolbox.register( "mate", tools.cxBlend, alpha =0.2)
toolbox.register( "mutate", tools.mutUniformInt, indpb=0.05, low=0, up=RANGE  )
toolbox.register("select", tools.selTournament, tournsize=3)


# シミュレーション中に表示する情報の設定

# In[12]:


hof = tools.ParetoFront()
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)


# シミュレーションの実行と最もよい個体の表示

# In[13]:


pop = toolbox.population(n=500)#個体数50
algorithms.eaSimple( pop, toolbox, cxpb = 0.8, mutpb=0.1, ngen=100, stats=stats, halloffame=hof, verbose=True)#世代数100

#最もよい個体の表示
best_ind = tools.selBest(pop, 1)[0]
print(best_ind)
print(myEvaluation(best_ind))


# In[ ]:




