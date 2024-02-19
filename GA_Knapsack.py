#!/usr/bin/env python
# coding: utf-8

# # 組合せに関係した問題（ナップザック問題）
# 
# ・ナップザック問題を例題にした組み合わせ問題の解法<br>
# ・DEAPの基本的な使い方の説明
# 
# DEAP
# https://deap.readthedocs.io/en/master/

# In[1]:


from deap import base, creator, tools, algorithms
import numpy as np
import random


# 物の重さと価値の設定

# In[2]:


items = ((8,10),(7,13), (6,7),(5,4), (4,9),(3,3),(2,3),(1,2))


# DEAPを使用するための設定（解析手法と初期値）

# In[3]:


#価値の最大化
creator.create( "Fitness", base.Fitness, weights=(1.0,) )
#遺伝子の各要素に重複を許すためlistを設定（順序など重複のない要素の場合はsetを用いる)
creator.create("Individual", list, fitness = creator.Fitness )
 
toolbox = base.Toolbox()
#遺伝子の属性の設定
toolbox.register( "attribute", random.randint, 0, 1 )
#初期個体の生成
toolbox.register( "individual", tools.initRepeat, creator.Individual, toolbox.attribute, len(items) )
#初期個体群を作成
toolbox.register( "population", tools.initRepeat, list, toolbox.individual )


# 評価関数

# In[4]:


MAX_WEIGHT = 10
def evalKnapsack( individual ):
    weight = 0.0
    value = 0.0
    for i in range(len(items)):#individual:
        s = individual[i]
        if s == 1:
            weight += items[ i ][0]
            value += items[ i ][1]
    if weight > MAX_WEIGHT:
        return 0, 
    return value,


# DEAPを使用するための設定（評価，選択，交叉，突然変異）

# In[5]:


toolbox.register("evaluate", evalKnapsack)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register( "mate", tools.cxTwoPoint )
toolbox.register( "mutate", tools.mutFlipBit, indpb=0.05 )


# シミュレーション中に表示する情報の設定

# In[15]:


hof = tools.ParetoFront()
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)


# シミュレーションの実行

# In[16]:


pop = toolbox.population(n=50)#個体数50
algorithms.eaSimple( pop, toolbox, cxpb = 0.8, mutpb=0.5, ngen=100, stats=stats, halloffame=hof, verbose=True)#世代数100


# 最もよい個体の表示

# In[17]:


best_ind = tools.selBest(pop, 1)[0]
print(best_ind)
print(evalKnapsack(best_ind))

