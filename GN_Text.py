#!/usr/bin/env python
# coding: utf-8

# # 文章の相関の可視化（共起ネットワーク）
# ・文章を単語へ分かち書き（分解）<br>
# ・単語の接続<br>
# ・共起ネットワークの作成<br>
# 
# NetworkX
# https://networkx.org/documentation/stable/index.html

# In[1]:


import MeCab#分かち書き用ライブラリ


# 分かち書きの基礎

# In[2]:


mecab = MeCab.Tagger()
node = mecab.parseToNode("メロスは激怒した。")
while node:
    w = node.surface
    w_type = node.feature.split(',')[0]
    print(w, w_type)
    node = node.next


# #### 走れメロスの最初の文の共起ネットワーク（文単位で解析）

# 走れメロスの最初の文だけ書かれたファイルの読み込み

# In[3]:


with open("hashire_merosu_first.txt", mode="r", encoding="utf-8") as f:
    text_all = f.read() 


# 不必要な部分（ヘッダや説明）を削除

# In[4]:


import re

text = re.split(r'\-{5,}', text_all)[2]
text = re.split(r'底本：', text)[0]
text = re.sub(r'《.+?》', '', text)
text = re.sub(r'［＃.+?］', '', text)
text = re.sub(r'\n', '', text)
text = re.sub(r'「', '', text)
text = re.sub(r'」', '', text)
text = re.sub(r'\u3000', '', text)
text = text.strip()
text = text.split("。")
#text = text.split("\n")
text


# 名詞だけ抜き出す

# In[5]:


mecab = MeCab.Tagger()
for line in text:
    node = mecab.parseToNode(line)
    while node:
        w = node.surface
        w_type = node.feature.split(',')[0]
        if w_type in ["名詞"]:
            print(w, w_type)
        node = node.next


# 文章内の単語のつながりを解析

# In[6]:


mecab = MeCab.Tagger()
list_2 = []
for line in text:
    node = mecab.parseToNode(line)
    list_1 = []
    while node:
        w = node.surface
        w_type = node.feature.split(',')[0]
        if w_type in ["名詞"]:
            list_1.append(w)
        node = node.next
    if list_1:
        list_2.append(list_1)
print(sorted(list_2)) 


# In[7]:


from collections import defaultdict, Counter, OrderedDict
from copy import copy, deepcopy
from itertools import combinations, dropwhile


# 単語のペアを作成

# In[8]:


pair_all = []
for list_1 in list_2:
    pair_1 = list(combinations(set(list_1), 2))
    for i,pair in enumerate(pair_1):
        pair_1[i] = tuple(sorted(pair))
    pair_all += pair_1
pair_count = Counter(pair_all)
min_cnt=0
for key, count in dropwhile(lambda key_count: key_count[1] >= min_cnt, pair_count.most_common()):
    del pair_count[key]
print(pair_count)


# 出現頻度のカウント

# In[9]:


word_count = Counter()
for list_1 in list_2:
    word_count += Counter(set(list_1))
print(word_count)


# jaccard係数の計算

# In[10]:


jaccard_coef = []
for pair, cnt in pair_count.items():
    jaccard_coef.append(cnt / (word_count[pair[0]] + word_count[pair[1]] - cnt))
print(jaccard_coef)


# jaccard係数がedge_th未満の単語ペアを除外

# In[11]:


edge_th=0.0
jaccard_dict = OrderedDict()
for (pair, cnt), coef in zip(pair_count.items(), jaccard_coef):
    if coef >= edge_th:
        jaccard_dict[pair] = coef
        print(pair, cnt, coef, word_count[pair[0]], word_count[pair[1]], sep='\t')


# In[12]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'MS Gothic'
import japanize_matplotlib

import networkx as nx


# グラフで表示

# In[13]:


G = nx.Graph()
nodes = sorted(set([j for pair in jaccard_dict.keys() for j in pair]))
G.add_nodes_from(nodes)
print('Number of nodes =', G.number_of_nodes())

#  線（edge）の追加
for pair, coef in jaccard_dict.items():
    G.add_edge(pair[0], pair[1], weight=coef)

print('Number of edges =', G.number_of_edges())

plt.figure(figsize=(6, 6))

# nodeの配置方法の指定
seed = 0
np.random.seed(seed)
pos = nx.spring_layout(G, k=0.3, seed=seed)
pr = nx.pagerank(G)
nx.draw_networkx_nodes(
    G,
    pos,
    node_color='w',#list(pr.values()),
    edgecolors='k', 
#    cmap=plt.cm.rainbow,
    alpha=0.7,
    node_size=1000)#[10000*v for v in pr.values()])

# 日本語ラベルの設定
nx.draw_networkx_labels(G, pos, font_size=15, font_family='MS Gothic', font_weight='bold')

# エッジの太さをJaccard係数により変える
edge_width = [d['weight'] * 2 for (u, v, d) in G.edges(data=True)]
nx.draw_networkx_edges(G, pos, alpha=0.7, edge_color='darkgrey', width=edge_width)
plt.axis('off')
plt.tight_layout()
plt.savefig("mrs_first1.png")
plt.savefig("mrs_first1.svg")
plt.show()


# #### 走れメロスの最初の文の共起ネットワーク（段落単位で解析）
# ここまでのスクリプトは句点（。）で区切って解析の塊を作っていたが，以下では段落で区切って解析の塊を作る。<br>
# 以下のスクリプトは段落で繰りるように修正している。

# In[14]:


text = re.split(r'\-{5,}', text_all)[2]
text = re.split(r'底本：', text)[0]
text = re.sub(r'《.+?》', '', text)
text = re.sub(r'［＃.+?］', '', text)
#text = re.sub(r'\n', '', text)
text = re.sub(r'「', '', text)
text = re.sub(r'」', '', text)
text = re.sub(r'\u3000', '', text)
text = text.strip()
#text = text.split("。")
text = text.split("\n")
text


# 名詞だけ抜き出す

# In[15]:


mecab = MeCab.Tagger()
list_2 = []
for line in text:
    node = mecab.parseToNode(line)
    list_1 = []
    while node:
        w = node.surface
        w_type = node.feature.split(',')[0]
        if w_type in ["名詞"]:
            list_1.append(w)
        node = node.next
    if list_1:
        list_2.append(list_1)
print(sorted(list_2)) 


# 単語のペアを作成，出現頻度のカウント，jaccard係数の計算，jaccard係数がedge_th未満の単語ペアを除外

# In[16]:


#単語のペアを作成
pair_all = []
for list_1 in list_2:
    pair_1 = list(combinations(set(list_1), 2))
    for i,pair in enumerate(pair_1):
        pair_1[i] = tuple(sorted(pair))
    pair_all += pair_1
pair_count = Counter(pair_all)
min_cnt=0
for key, count in dropwhile(lambda key_count: key_count[1] >= min_cnt, pair_count.most_common()):
    del pair_count[key]
print(pair_count)
#出現頻度のカウント
word_count = Counter()
for list_1 in list_2:
    word_count += Counter(set(list_1))
print(word_count)
#jaccard係数の計算
jaccard_coef = []
for pair, cnt in pair_count.items():
    jaccard_coef.append(cnt / (word_count[pair[0]] + word_count[pair[1]] - cnt))
print(jaccard_coef)
# jaccard係数がedge_th未満の単語ペアを除外
edge_th=0.0
jaccard_dict = OrderedDict()
for (pair, cnt), coef in zip(pair_count.items(), jaccard_coef):
    if coef >= edge_th:
        jaccard_dict[pair] = coef
        print(pair, cnt, coef, word_count[pair[0]], word_count[pair[1]], sep='\t')


# In[17]:


G = nx.Graph()
nodes = sorted(set([j for pair in jaccard_dict.keys() for j in pair]))
G.add_nodes_from(nodes)
print('Number of nodes =', G.number_of_nodes())

#  線（edge）の追加
for pair, coef in jaccard_dict.items():
    G.add_edge(pair[0], pair[1], weight=coef)

print('Number of edges =', G.number_of_edges())

plt.figure(figsize=(6, 6))

# nodeの配置方法の指定
seed = 0
np.random.seed(seed)
pos = nx.spring_layout(G, k=0.3, seed=seed)
pr = nx.pagerank(G)
nx.draw_networkx_nodes(
    G,
    pos,
    node_color='w',#list(pr.values()),
    edgecolors='k', 
#    cmap=plt.cm.rainbow,
    alpha=0.7,
    node_size=1000)#[10000*v for v in pr.values()])

# 日本語ラベルの設定
nx.draw_networkx_labels(G, pos, font_size=15, font_family='MS Gothic', font_weight='bold')

# エッジ太さをJaccard係数により変える
edge_width = [d['weight'] * 2 for (u, v, d) in G.edges(data=True)]
nx.draw_networkx_edges(G, pos, alpha=0.7, edge_color='darkgrey', width=edge_width)
plt.axis('off')
plt.tight_layout()
plt.savefig("mrs_first2.png")
plt.savefig("mrs_first2.svg")
plt.show()


# #### 走れメロスの最初の文の共起ネットワーク（文全体を解析）

# 【手順１（１）】走れメロスの全文を読み込む

# In[18]:


with open("hashire_merosu.txt", mode="r", encoding="utf-8") as f:
    text_all = f.read() 


# 【手順１（１）】段落に分ける

# In[19]:


text = re.split(r'\-{5,}', text_all)[2]
text = re.split(r'底本：', text)[0]
text = re.sub(r'《.+?》', '', text)
text = re.sub(r'［＃.+?］', '', text)
#text = re.sub(r'\n', '', text)
text = re.sub(r'「', '', text)
text = re.sub(r'」', '', text)
text = re.sub(r'\u3000', '', text)
text = text.strip()
#text = text.split("。")
text = text.split("\n")
text


# 【手順１（３）】抽出する単語（登場人物）のリストを読み込む
# この単語だけを対象とする。

# In[20]:


with open("hashire_merosu_name.txt", mode="r", encoding="utf-8") as f:
    selectwords = f.read() 
selectwords = selectwords.split("\n")
selectwords


# 【手順１（４）】単語リストの作成

# In[21]:


mecab = MeCab.Tagger()
list_2 = []
for line in text:
    #line = text[0]
    list_1 = []
    node = mecab.parseToNode(line)
    while node:
        w = node.surface
        w_type = node.feature.split(',')[0]
        
        replace_words = {
            '私': 'メロス', 
            '友': 'セリヌンティウス', 
            '国王':'ディオニス',
            '王':'ディオニス', 
            '妹':'花嫁',
        }

        for key, value in replace_words.items():
            w = w.replace(key, value)
            
        if(w_type in ["代名詞", "名詞"]) & (w in selectwords):
#        if(w_type in ["名詞"]):
            list_1.append(w)
        node = node.next
    if list_1:
        list_2.append(list_1)
#    print(list_1)
#print(sorted(list_2))


# 【手順２】その単語リストのすべての組み合わせのリストを作成

# In[22]:


pair_all = []
for list_1 in list_2:
    pair_1 = list(combinations(set(list_1), 2))
#    print(pair_1)
    for i,pair in enumerate(pair_1):
        pair_1[i] = tuple(sorted(pair))
    pair_all += pair_1
print(pair_all)


# 【手順３】単語の【登場回数が設定した値】より小さい場合，その単語を含む組み合わせを削除min_cnt=2を変更する。

# In[23]:


pair_count = Counter(pair_all)
min_cnt=2
for key, count in dropwhile(lambda key_count: key_count[1] >= min_cnt, pair_count.most_common()):
    del pair_count[key]
#print(pair_count)


# 【手順４】Jaccard係数を計算し，【Jaccard係数が設定した値】より小さい組み合わせを削除edge_th=0.1の値を変更する。

# In[24]:


word_count = Counter()
for list_1 in list_2:
    word_count += Counter(set(list_1))
#print(word_count)
jaccard_coef = []
for pair, cnt in pair_count.items():
    jaccard_coef.append(cnt / (word_count[pair[0]] + word_count[pair[1]] - cnt))
print(jaccard_coef)
# jaccard係数がedge_th未満の単語ペアを除外
edge_th=0.1
jaccard_dict = OrderedDict()
for (pair, cnt), coef in zip(pair_count.items(), jaccard_coef):
    if coef >= edge_th:
        jaccard_dict[pair] = coef
        print(pair, cnt, coef, word_count[pair[0]], word_count[pair[1]], sep='\t')


# In[25]:


word_count = Counter()
for list_1 in list_2:
    word_count += Counter(set(list_1))
    print(Counter(set(list_1)))
print(word_count)
list_2


# 【手順５】ネットワークの作成

# In[26]:


G = nx.Graph()
nodes = sorted(set([j for pair in jaccard_dict.keys() for j in pair]))
G.add_nodes_from(nodes)
print('Number of nodes =', G.number_of_nodes())

#  線（edge）の追加
for pair, coef in jaccard_dict.items():
    G.add_edge(pair[0], pair[1], weight=coef)

print('Number of edges =', G.number_of_edges())

plt.figure(figsize=(6, 6))

# nodeの配置方法の指定
seed = 0
np.random.seed(seed)
pos = nx.spring_layout(G, k=0.3, seed=seed)
pr = nx.pagerank(G)
nx.draw_networkx_nodes(
    G,
    pos,
    node_color='w',#list(pr.values()),
    edgecolors='k', 
#    cmap=plt.cm.rainbow,
    alpha=0.7,
    node_size=1000)#[10000*v for v in pr.values()])

# 日本語ラベルの設定
nx.draw_networkx_labels(G, pos, font_size=15, font_family='MS Gothic', font_weight='bold')

# エッジ太さをJaccard係数により変える
edge_width = [d['weight'] * 2 for (u, v, d) in G.edges(data=True)]
nx.draw_networkx_edges(G, pos, alpha=0.7, edge_color='darkgrey', width=edge_width)
plt.axis('off')
plt.tight_layout()
plt.savefig("mrs_all.png")
plt.savefig("mrs_all.svg")
plt.show()


# #### レビューの共起ネットワーク

# In[27]:


with open("camera_review.txt", mode="r", encoding="utf-8") as f:
    text_all = f.read() 


# In[28]:


text = text_all
#text = re.split(r'\-{5,}', text)[2]
#text = re.split(r'底本：', text)[0]
text = re.sub(r'【.+?】', '', text)
#text = re.sub(r'［＃.+?］', '', text)
text = re.sub(r'\n', '', text)
text = re.sub(r'「', '', text)
text = re.sub(r'」', '', text)
text = re.sub(r'★', '', text)
#text = re.sub(r'\u3000', '', text)
#text = text.strip()
#text = text.split("。")
text = text.split("@@@@@")


# In[29]:


mecab = MeCab.Tagger()
list_2 = []
for line in text:
    #line = text[0]
    list_1 = []
    node = mecab.parseToNode(line)
    while node:
        w = node.surface
        w_type = node.feature.split(',')[0]
        for key, value in replace_words.items():
            w = w.replace(key, value)
            
        if(w_type in ["名詞"]):
            list_1.append(w)
        node = node.next
    if list_1:
        list_2.append(list_1)
#    print(list_1)
#print(sorted(list_2))

pair_all = []
for list_1 in list_2:
    pair_1 = list(combinations(set(list_1), 2))
#    print(pair_1)
    for i,pair in enumerate(pair_1):
        pair_1[i] = tuple(sorted(pair))
    pair_all += pair_1
#print(pair_all)
pair_count = Counter(pair_all)
min_cnt=8
for key, count in dropwhile(lambda key_count: key_count[1] >= min_cnt, pair_count.most_common()):
    del pair_count[key]
#print(pair_count)

word_count = Counter()
for list_1 in list_2:
    word_count += Counter(set(list_1))
#print(word_count)
jaccard_coef = []
for pair, cnt in pair_count.items():
    jaccard_coef.append(cnt / (word_count[pair[0]] + word_count[pair[1]] - cnt))
#print(jaccard_coef)
# jaccard係数がedge_th未満の単語ペアを除外
edge_th=0.6#0.6
jaccard_dict = OrderedDict()
for (pair, cnt), coef in zip(pair_count.items(), jaccard_coef):
    if coef >= edge_th:
        jaccard_dict[pair] = coef
#        print(pair, cnt, coef, word_count[pair[0]], word_count[pair[1]], sep='\t')


# In[30]:


G = nx.Graph()
nodes = sorted(set([j for pair in jaccard_dict.keys() for j in pair]))
G.add_nodes_from(nodes)
print('Number of nodes =', G.number_of_nodes())

#  線（edge）の追加
for pair, coef in jaccard_dict.items():
    G.add_edge(pair[0], pair[1], weight=coef)

print('Number of edges =', G.number_of_edges())

plt.figure(figsize=(6, 6))

# nodeの配置方法の指定
seed = 0
np.random.seed(seed)
pos = nx.spring_layout(G, k=0.3, seed=seed)
pr = nx.pagerank(G)
nx.draw_networkx_nodes(
    G,
    pos,
    node_color='w',#list(pr.values()),
    edgecolors='k', 
#    cmap=plt.cm.rainbow,
    alpha=0.7,
    node_size=1000)#[10000*v for v in pr.values()])

# 日本語ラベルの設定
nx.draw_networkx_labels(G, pos, font_size=15, font_family='MS Gothic', font_weight='bold')

# エッジ太さをJaccard係数により変える
edge_width = [d['weight'] * 2 for (u, v, d) in G.edges(data=True)]
nx.draw_networkx_edges(G, pos, alpha=0.7, edge_color='darkgrey', width=edge_width)
plt.axis('off')
plt.tight_layout()
plt.savefig("review.png")
plt.savefig("review.svg")
plt.show()

