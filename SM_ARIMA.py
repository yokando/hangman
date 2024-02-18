#!/usr/bin/env python
# coding: utf-8

# # ARIMAモデル，1次式型トレンドを含むデータ
# 1. ARMAシステム（真）のデータに1次式データ（非定常）が重畳する。<br>
# 2. ARIMAモデルを用いて，予測を行う。<br>
# Ref:<br>
# Autoregressive Moving Average (ARMA): Artificial data https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima_model.ARIMA.html<br>
# statsmodels.tsa.arima.model.ARIMA: https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
# statsmodels.tsa.arima.model.ARIMAResults: https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMAResults.html<br>

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')

import statsmodels.api as sm
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima.model import ARIMA

from scipy import stats

np.random.seed(123)


# 観測データ（トレニーングデータ）              y  
# 予測精度を見るための実データ（テストデータ）  y_test

# In[ ]:


ar = [1, -1.5, 0.7]
ma = [1.0, 0.6]

nobs = 1000
nobs_test = 100
nobs_all = nobs + nobs_test

dist = stats.norm(loc=0, scale=1.0).rvs

# 知りたい信号成分
sig_all = arma_generate_sample(ar, ma, nsample=nobs_all, distrvs=dist, burnin=500)

# トレンドの信号（1次式型）
coef_a, coef_b = 0.05, 4
trend_all = coef_a*np.arange(len(sig_all)) + coef_b

# 出力信号（トレーニングデータ＋テストデータ）
y_all = sig_all + trend_all

# インデックスを与える
index = pd.date_range('1/1/2000', periods=nobs_all, freq='D')
y_all = pd.Series(y_all, index=index)

#信号データにindexを付加
sig_all = pd.Series(sig_all, index=index)

y = y_all[:nobs]      #観測データはｙ
y_test = y_all[nobs:] #予測精度を見るためのテストデータはy_test

y.tail(5)

y.plot(color='b')
y_test.plot(color='c')

#plt.savefig('fig_SM_ARIMA_signal_trend.pdf')
plt.show()


# yの1階差分系列（青），元の信号（灰色）とは振幅，位相が異なることがわかる

# In[ ]:


diff = (y - y.shift()).dropna(axis=0) #先頭のデータは NaNとなるため
diff.plot(color='b')
sig_all[:nobs-1].plot(color='gray')

#plt.savefig('fig_SM_ARIMA_ident_y_diff.png')
plt.show()


# ARIMAモデル,  トレンドが重畳した観測値y に対して適用

# In[ ]:


arima_result = ARIMA(y, order=(2,1,1), trend='n').fit() # 引数trendは'n'（トレンド無し）を指定
print(arima_result.summary())


# #### 残差（誤差）系列とその自己相関関数のプロット
# 自己相関のプロットでは，sig_valの領域内にあるかを確認する。

# In[ ]:


resid = arima_result.resid # residual sequence
sig_val = 0.05 # 有意水準

#resid.plot(figsize=(12,4))
#print(stats.normaltest(resid))

fig = plt.figure(figsize=(12,3))
ax1 = fig.add_subplot(111)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=20, alpha=sig_val, ax=ax1)

#plt.savefig('fig_SM_ARIMA_resid_acf.png')
plt.show()


# #### statsmodels.tsa.arima.model.ARIMAResults.predict
# https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMAResults.predict.html

# In[ ]:


fig = plt.figure(figsize=(8,3))

start, end, pred_start = '2002-08-20', '2002-10-15', '2002-09-27'
pred = arima_result.predict(start=start, end=end)
pred.plot(label='predict', color='k')
y[start:].plot(color='m', label='y')
y_test[pred_start:end].plot(color='b', label='y_test')
plt.legend(loc='upper left')

#plt.savefig('fig_SM_ARIMA_y_predict.png')
plt.show()


# #### yとy_testのインデックスの先頭と最終の値を知る
# 上記のグラフの範囲を定める目安にしたいため。

# In[ ]:


print(y.index[0], y.index[-1])


# In[ ]:


print(y_test.index[0], y_test.index[-1])


# #### 見る区間を拡げる

# In[ ]:


plt.plot(y)
plt.plot(pred, color='r')


# In[ ]:




