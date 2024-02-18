#!/usr/bin/env python
# coding: utf-8

# # SARIMAX model for AirPassenger
# Seasonal AutoRegressive Integrated Moving Average with eXogenous model  
# 
# SARIMAのモデル構造，引数（パラメータ）は次を参照：statsmodels.tsa.statespace.sarimax.SARIMAX  
# http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
# 
# AirPassenger:国際線の航空旅客数，月ごと，1949年1月1日から1960年12月1日まで  
# https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/AirPassengers.html  

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


fname = 'AirPassengers.csv'
df = pd.read_csv(fname, index_col='Date', parse_dates=True)
df.head()


# In[ ]:


df.Passengers


# In[ ]:


df.plot()

#plt.savefig('fig_SM_SARIMAX_PassengerData.png')
plt.show()


# In[ ]:


acf = sm.tsa.stattools.acf(df, nlags=40)
#fig, ax = plt.subplots(figsize=(4,4))
plt.plot(acf, marker='o')
plt.xlabel('lag')
plt.ylabel('acf')

#plt.savefig('fig_SM_SARIMAX_PassengerData_acf.png')
plt.show()


# 次の引数  
# order = (p,d,q): ARMA(p,q), 差分の次数d  
# seasonal_order = (P, D, Q, s), 季節性用のモデルの次数で，(P,D,Q)は(p,d,q)に類似したもの。sは季節調整に適用する周期を指定する。  
# 上記のデータの場合，12点ごとに周期性（単位時間で見ている）があるので，s=12とする。

# もし，ValueWarning: No frequency information was provided, so inferred frequency MS will be used.というWarningが出ても無視して構わない。<br>
# この警告は時間間隔(frequency)が指定されていなから，MS（month start frequency）を仮定しますよ，と言っている。<br>
# ちなみに，時間間隔（statsmodelsはOffset Aliasesと称している）は次を参照：http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases <br>
# しかし，dfにはその情報を与えているので，気にしないこととする。<br>
# それに，statsmodelsが示す次の例題でもこのWarningを出しながら計算を進めている。<br>
# https://www.statsmodels.org/dev/examples/notebooks/generated/tsa_arma_0.html<br>
# https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_varmax.html<br>
# 

# In[ ]:


#SARIMA_model = sm.tsa.SARIMAX(df, order=(3,1,2), seasonal_order=(1,1,1,12)).fit(method='bfgs', maxiter=500)
SARIMAX_model = sm.tsa.SARIMAX(df, order=(3,1,2), seasonal_order=(1,1,1,12)).fit(maxiter=200)
print(SARIMAX_model.summary())


# In[ ]:


# 残差のチェック
resid = SARIMAX_model.resid
fig, ax = plt.subplots(figsize=(10,4))
fig2 = sm.graphics.tsa.plot_acf(resid, lags=40,  alpha=0.05, ax=ax)

#plt.savefig('fig_SM_SARIMAX_Resid_acf.png')
plt.show()


# In[ ]:


#予測
pred = SARIMAX_model.predict(start='1960-01-01', end='1962-12-01')


# In[ ]:


plt.plot(df)
plt.plot(pred, 'r')

plt.xlabel('Date')
plt.ylabel('Passengers')

#plt.savefig('fig_SM_SARIMAX_Predict.png')
plt.show()


# In[ ]:




