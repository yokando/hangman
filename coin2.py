#
# コイン投げ問題のベイズ統計による計算
##################################################################

import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pandas as pd
import pymc3 as pm
plt.style.use('seaborn-darkgrid')

np.random.seed(123)
n_experiments = 4
theta_real = 0.35
data = stats.bernoulli.rvs(p=theta_real, size=n_experiments)
print("data= ", data)

if __name__ == '__main__':
  with pm.Model() as our_first_model:
# 事前確率（ベータ分布）を設定
    theta = pm.Beta('theta', alpha=1, beta=1)
# 尤度（二項分布）を設定
    y = pm.Bernoulli('y', p=theta, observed=data)
# 事後分布（NUTS法）を求める（最初の100個のデータは不使用）
    trace = pm.sample(1000)
    burnin = 100
    chain = trace[burnin:]
# 事後分布およびサンプル軌跡の作画   
    pm.traceplot(trace, lines={'theta':theta_real});
    plt.savefig('coin4.png', dpi=300, figsize=(5.5, 5.5))
# 事後分布の統計値を表示
    print(pm.summary(chain))
    plt.figure()
# 自己相関グラフの作画
    pm.autocorrplot(chain)
    plt.savefig('coin8.png', dpi=300, figsize=(5.5, 5.5))
    plt.figure()
# 事後分布と95%HPIを作画
    pm.plot_posterior(chain)
    plt.savefig('coin9.png', dpi=300, figsize=(5.5, 5.5))
    plt.figure()
# 事後分布とROPE（実質同値域）を作画
    pm.plot_posterior(chain, rope=[0.45,.55])
    plt.savefig('coin10.png', dpi=300, figsize=(5.5, 5.5))
    plt.figure()
# 事後分布と参照点を作画
    pm.plot_posterior(chain, ref_val=0.5)
    plt.savefig('coin11.png', dpi=300, figsize=(5.5, 5.5))
    plt.figure()