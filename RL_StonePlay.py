#!/usr/bin/env python
# coding: utf-8

# # 2人の意思決定（石取りゲームで人間と対戦）
# 
# ・強化学習の1つであるQラーニングの学習結果を用いる<br>
# ・人間とエージェントで対戦する<br>
# ・大戦前にRL_Stone.ipynbを実行してQV1.txtを作成しておく必要がある
# 

# In[1]:


import numpy as np


# 石の数の設定（学習と同じ石の数にする必要がある）

# In[2]:


BOTTLE_N = 11


# 状態遷移：状態を変化させるための関数

# In[3]:


def step(action, state, turn):
    state = state + action + 1
    rewards = [0,0]
    done = False
    if (state>=BOTTLE_N):
        state = BOTTLE_N
        rewards[turn] = -1
        rewards[(turn+1)%2] = 1
        done = True
    return state, rewards, done


# 行動選択：Q値から次の行動を選択するための関数

# In[4]:


def getAction(state, epsilon, qv):
    if epsilon > np.random.uniform(0, 1):#徐々に最適行動のみをとる、ε-greedy法
        next_action = np.random.choice([0, 1, 2])
    else:
        a = np.where(qv[state]==qv[state].max())[0]
        next_action = np.random.choice(a)
    return next_action


# 対戦の実行

# In[5]:


state = 0
QV = np.loadtxt('QV1.txt')
print('New Game pin:{}'.format(BOTTLE_N))
while(1):
    action = int(input('[1-3]'))
    state, rewards, done = step(state, action-1, 1)
    print('act:{0}, pin:{1}'.format(action, BOTTLE_N - state))
    if (done==True):
        print('You Lose!')
        break
    action = getAction(state, 0, QV)
    state, rewards, done = step(state, action, 0)
    print('act:{0}, pin:{1}'.format(action+1, BOTTLE_N - state))
    if (done==True):
        print('You Win!')
        break


# In[ ]:




