#!/usr/bin/env python
# coding: utf-8

# # 2人の意思決定（石取りゲームの学習）
# 
# ・強化学習の1つであるQラーニングを用いる<br>
# ・エージェントは2つ用いる
# 

# In[1]:


import numpy as np


# 変数の設定

# In[2]:


BOTTLE_N = 11
QV0=np.zeros((BOTTLE_N+1,3), dtype=np.float32)
QV1=np.zeros((BOTTLE_N+1,3), dtype=np.float32)
QVs = [QV0, QV1]


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


# Q値の更新：状態，行動，報酬，次の状態を用いてQ値を更新するための関数

# In[5]:


def updateQValue(action, reward, state, state_old, qv):
    alpha = 0.5
    gamma = 0.9
    maxQ = np.max(qv[state])
    qv[state_old][action] = (1-alpha)*qv[state_old][action]+alpha*(reward + gamma*maxQ);


# 変数の設定（学習の繰り返しの回数）

# In[6]:


num_episodes = 100


# 強化学習の実行

# In[10]:


for episode in range(num_episodes):  #試行数分繰り返す
    state = 0
    state_old = [0,0]
    rewards = [0,0]
    actions = [0,0]
    epsilon = 0.5 * (1 / (episode + 1))
    while(1):
        actions[0] = getAction(state, epsilon, QVs[0])
        state_old[0] = state
        state, rewards, done = step(state, actions[0], 0)
        updateQValue(actions[1], rewards[1], state, state_old[1], QVs[1])
        if (done==True):
            updateQValue(actions[0], rewards[0], state, state_old[0], QVs[0])
            print('{} : 0 Lose, 1 Win!!'.format(episode))
            break
        actions[1] = getAction(state, epsilon, QVs[1])
        state_old[1] = state
        state, rewards, done = step(state, actions[1], 1)
        updateQValue(actions[0], rewards[0], state, state_old[0], QVs[0])
        if (done==True):
            updateQValue(actions[1], rewards[1], state, state_old[1], QVs[1])
            print('{} : 0 Win!!, 1 Lose'.format(episode))
            break


# In[8]:


print("Agent 0")
print(QVs[0])
print("Agent 1")
print(QVs[1])
np.savetxt('QV0.txt', QVs[0])
np.savetxt('QV1.txt', QVs[1])


# 必勝法と同じ取り方を学習しているかの確認

# In[9]:


for j in range(2):
    print("Agent", j)
    for i in range(BOTTLE_N):
        a = np.where(QVs[j][i]==QVs[j][i].max())[0]
        print('残り本数',BOTTLE_N-i,'取る数',a+1,'必勝法',(BOTTLE_N-i-1)%4,'なんでもよい' if (BOTTLE_N-i-1)%4 == 0 else \
              ('不定' if a.size >1 else ('正解' if (BOTTLE_N-i-1)%4 == a+1 else '不正解')))

