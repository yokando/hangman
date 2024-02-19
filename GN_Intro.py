#!/usr/bin/env python
# coding: utf-8

# # グラフ理論の基礎
# 
# ・グラフ理論を理解するうえで重要な言葉の説明<br>
# ・NetworkXの基本的な使い方の説明
# 
# NetworkX
# https://networkx.org/documentation/stable/index.html

# In[1]:


import networkx as nx
import matplotlib.pyplot as plt


# #### グラフの基本的な書き方
# ネットワークの作成：この例では2点のつながりを設定することでネットワークを作成<br>
# レイアウトの設定：この例では円形に配置するレイアウトを設定<br>
# グラフの描画：ネットワークを設定したレイアウトで描画

# In[2]:


G = nx.Graph()
G.add_edges_from([(0, 1), (0, 3), (0, 5), (2, 3), (2, 4), (2, 5), (3, 5), (4, 5)])
plt.subplots(figsize = (6, 6))
pos = nx.circular_layout(G) 
nx.draw_networkx(G, pos,node_color="lightgray", edgecolors='k')


# #### 隣接行列
# つながっている場合は1，つながっていない場合は0として表現する行列
# 例えば，0と1はつながっているため1となっており，1と2はつながっていないため0となっている。

# In[3]:


adj = nx.adjacency_matrix(G)#隣接行列
print(adj.todense())


# #### ノード間の距離
# 0から4に至るには「0，5，4」を順にたどるため距離は2
# 1から2に至るには「１，0，3，2」を順にたどるため距離は3

# In[4]:


length = dict(nx.shortest_path_length(G))#距離
print(length[0][4])
print(length[1][2])


# #### ノード間の経路
# 0から4に至る経路は「0，5，4」
# 1から2に至る経路は「１，0，3，2」

# In[5]:


path = dict(nx.shortest_path(G))#経路
print(path[0][4])
print(path[1][2])


# #### 全ノード間の平均距離
# すべのノードの組合わせの距離の平均

# In[6]:


avr_length = nx.average_shortest_path_length(G)#平均距離
print(avr_length)


# #### 次元と次数中心性
# 次数は各ノードの持つエッジの数<br>
# 次数中心性はそれをエッジの最大数（全ノード数-1）で割った数

# In[8]:


dg = nx.degree(G)#次数
dg_cent = nx.degree_centrality(G)#次数中心性
print(dg)
print(dg_cent)


# #### ヒストグラムと平均次数
# ヒストグラムは各ノードの持つエッジを並べた結果<br>
# 平均次数は次数の平均

# In[9]:


dg_hist = nx.degree_histogram(G)#ヒストグラム
dg_dens = nx.density(G)#平均次数
print(dg_hist)
print(dg_dens)


# #### クラスター係数と平均クラスター係数
# クラスター係数はノード同士の密接さを表す係数br>
# 平均クラスター係数はその平均値

# In[10]:


cl = nx.clustering(G)#クラスター係数
cl_avr = nx.average_clustering(G)#平均クラスター係数
print(cl)
print(cl_avr)


# #### グラフ例：全結合グラフを円形レイアウトで表示

# In[16]:


G = nx.complete_graph(10)
pos = nx.circular_layout(G)
plt.subplots(figsize = (6, 6))
nx.draw_networkx(G, pos,node_color="lightgray", edgecolors='k')


# #### グラフ例：グリッドグラフをスプリングレイアウトで表示

# In[30]:


G = nx.grid_2d_graph(4,6)
pos = nx.spring_layout(G)
plt.subplots(figsize = (6, 6))
nx.draw_networkx(G, pos,node_color="lightgray", edgecolors='k')


# #### グラフ例：グリッドグラフをスプリングレイアウトで表示

# In[26]:


G = nx.gnp_random_graph(10, 0.3)
pos = nx.circular_layout(G)
plt.subplots(figsize = (6, 6))
nx.draw_networkx(G, pos,node_color="lightgray", edgecolors='k')


# #### グラフ例：レギュラーグラフ（次数が一定のグラフ）を円形レイアウトで表示

# In[28]:


G = nx.random_regular_graph(4,10)
pos = nx.circular_layout(G)
plt.subplots(figsize = (6, 6))
nx.draw_networkx(G, pos,node_color="lightgray", edgecolors='k')


# #### グラフ例：星形グラフを円形レイアウトで表示

# In[29]:


G = nx.star_graph(20)
pos = nx.spring_layout(G)
plt.subplots(figsize = (6, 6))
nx.draw_networkx(G, pos,node_color="lightgray", edgecolors='k')

