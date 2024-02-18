#!/usr/bin/env python
# coding: utf-8

# # 1人の意思決定（ミントタブレット問題）
# 
# ・強化学習の1つであるQラーニングを用いる<br>
# ・エージェントは1つだけ用いる
# 

# In[1]:


import numpy as np


# 状態遷移：状態を変化させるための関数

# In[2]:


def step(state, action):
    reward = 0
    if state==0:#閉じている
        if action==0:#開ける
            state = 1
    elif state==1:#開いていて，ミント菓子がある
        if action==1:#閉じる
            state = 0
        elif action==2:#傾ける
            state = 2
            reward = 1
    else:#開いていて，ミント菓子がない
        if action==1:
            state = 0
    return state, reward


# 行動選択：Q値から次の行動を選択するための関数

# In[3]:


def getAction(state, epsilon, qv):
    if epsilon > np.random.uniform(0, 1):#徐々に最適行動のみをとる、ε-greedy法
        next_action = np.random.choice([0, 1])
    else:
        a = np.where(qv[state]==qv[state].max())[0]
        next_action = np.random.choice(a)
    return next_action


# Q値の更新：状態，行動，報酬，次の状態を用いてQ値を更新するための関数

# In[4]:


def updateQValue(qv, state, action, reward, next_state):
    gamma = 0.9
    alpha = 0.5
    next_maxQ=max(qv[next_state])
    qv[state, action] = (1 - alpha) * qv[state, action] + alpha * (reward + gamma * next_maxQ)
    return qv


# 変数の設定

# In[5]:


num_episodes = 5  #総試行回数
num_steps = 10  #1試行の中の行動数


# 強化学習の実行

# In[6]:


QV = np.zeros((3, 3))
for episode in range(num_episodes):  #試行数分繰り返す
    state = 0#初期状態に戻す
    sum_reward = 0#累積報酬
    epsilon = 0.5 * (1 / (episode + 1))
    for t in range(num_steps):  #1試行のループ
        action = getAction(state, epsilon, QV)    # a_{t+1} 
        next_state, reward = step(state, action)
        print(state, action, reward)
        sum_reward += reward  #報酬を追加
        QV = updateQValue(QV, state, action, reward, next_state)
        state = next_state
    print('episode : %d total reward %d' %(episode+1, sum_reward))
    print(QV)

