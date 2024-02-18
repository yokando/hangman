#!/usr/bin/env python
# coding: utf-8

# # ARMA(2,1) パラメータ推定
# ARMA(2,1)のデータを生成して，パラメータ推定の様子を見る
# 
# パラメータの記述の仕方：ARパラメータは，本文とは符号が逆転する<br>
# statsmodels.tsa.arima_process.ArmaProcess  
# http://www.statsmodels.org/dev/generated/statsmodels.tsa.arima_process.ArmaProcess.html<br>
# 
# ARMAの出力サンプルの生成する関数<br>
# statsmodels.tsa.arima_process.arma_generate_sample<br>
# https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_process.arma_generate_sample.html<br>
# 
# ARMAの出力サンプル生成の例<br>
# https://www.statsmodels.org/dev/examples/notebooks/generated/tsa_arma_1.html

# #### 備考
# statsmodels ver 0.12から，statsmodels.tsa.arima_model.ARMAは非推奨となり，代わりにstatsmodels.tsa.arima.model.ARIMAを用いている。ここに,
# モデルのIを1とおけばARMAモデルとして扱うことができる。

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as stats
  
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima.model import ARIMA

import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')


# データ生成 arma_generate_sample<br>
# 引数の説明：  
# distrvs: ARMAへの入力でディフォルトは標準正規分布(0, 1) as np.random.randn
# burnin: 初期期間を何点かで定め，この期間後にデータを出力する。過渡現象を観測しないため。

# In[ ]:


ar = [1, -1.5, 0.7] # pole = 0.75 +/- 0.37 j < unit circle
ma = [1.0, 0.6]
pole = np.roots(ar)
print(pole, np.abs(pole))


# In[ ]:


np.random.seed(123)
nobs = 10000 # 観測データ数
dist = stats.norm(loc=0, scale=1.0).rvs
y = arma_generate_sample(ar, ma, nsample=nobs, distrvs=dist, burnin=1000)
print(type(y))


# これ以降のセルで用いている関数は次を参照<br>
# 
# statsmodels.tsa.arima.model.ARIMA<br>
# https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html<br>
# .fit()の作用 statsmodels.tsa.arima.model.ARIMA.fit<br>
# https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.fit.html<br>
# .fit()の結果 statsmodels.tsa.arima.model.ARIMAResults<br>
# https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMAResults.html<br>

# 真の次数は未知として，幾つかの推定次数を用い，比較する<br>

# In[ ]:


arma20 = ARIMA(endog=y, order=(2,0,0), trend='n').fit( ) # order=(p,d,q),  d = 0 in use of ARMA model
arma21 = ARIMA(endog=y, order=(2,0,1), trend='n').fit( ) 
arma32 = ARIMA(endog=y, order=(3,0,2), trend='n').fit( )
arma43 = ARIMA(endog=y, order=(4,0,3), trend='n').fit( )


# ## 注意！
# ARのパラメータの符号は反転している。これは，モデルの立て方でAR部の符号反転を行っているためである。  
# http://www.statsmodels.org/dev/generated/statsmodels.tsa.arima_process.ArmaProcess.html  
# 
# 
# このことは，次の例題でもARの符号を反転して与えていることからもわかる<br>
# http://www.statsmodels.org/dev/examples/notebooks/generated/tsa_arma_1.html  
# 
# また，極は$z$でなく$z^{-1}$で見ており，単位円外ならば安定極となっている。

# ## 注意！
# パッケージのバージョン更新に伴い，テキストや書籍の計算結果と少しずつ変わることがある。

# この例において，summaryで見る項目は，推定パラメータ（ar.Lx, ma.Lx）と推定誤差分散sigma2である。<br>
# このことは次のsigma2 = ssr / nobsの計算式より言える。<br>
# https://www.statsmodels.org/stable/_modules/statsmodels/tsa/arima_model.html<br>

# In[ ]:


print('arma20-----------summary--------------------')
print(arma20.summary())
print('arma21-----------summary--------------------')
print(arma21.summary())
print('arma32-----------summary--------------------')
print(arma32.summary())
print('arma43-----------summary--------------------')
print(arma43.summary())


# In[ ]:


fig = plt.subplots(figsize=(10,3))
plt.plot(y, label='y')
resid = arma21.resid  # short for residual
#print(len(resid))
plt.plot(resid, label='resid')
plt.xlabel('k')
plt.legend()

#plt.savefig('fig_SM_ARMA_Resid.png', bbox_inches='tight')
plt.show()


# #### 白色性の検定
# 自己相関をとり，白色であれば，$r(0)$のみに値があり，$r(\tau), \tau \ge 1$は0になる。
# 実際には0にはならないが，ある程度小さければ0とみなす，という考え方。

# In[ ]:


auto_corr = np.correlate(resid, resid, mode='full')
center = int(len(auto_corr)/2)
AutoR = auto_corr[center:]/np.max(auto_corr)
plt.xlabel('k')
plt.plot(AutoR)

#plt.savefig('fig_SM_ARMA_AutoCorr.png')
plt.show()
    
count = 0
for k in np.arange(1,len(AutoR)-1):
    if np.abs(AutoR[k]) > 2.17/np.sqrt(len(AutoR)):
        count += 1
    if np.abs(AutoR[k]) < -2.17/np.sqrt(len(AutoR)):
        count += 1
#        print("Warning", i, AutoR[i])
print('count = ', count, ' len(AutoR) = ', len(AutoR), '  rate =', count/len(AutoR))
print('k >= 1, max(AutoR[k] =', np.max(AutoR[1:]),  '    min(AutorR[k] =', np.min(AutoR[1:]))


# In[ ]:




