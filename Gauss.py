#
# Pythonによるベイズ統計モデリング
# 《ガウス過程》
########################################
#
# モジュールの取り込み
import pymc3 as pm
import numpy as np
import pandas as pd
import theano.tensor as tt
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
#
# 分析に使う人工的なデータの作成、表示
plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])
np.set_printoptions(precision=2)
np.random.seed(1)
N = 20
x = np.random.uniform(0, 10, size=N)
y = np.random.normal(np.sin(x), 0.2)
plt.plot(x, y, 'o')
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$f(x)$', fontsize=16, rotation=0)
plt.savefig('GP1.png', dpi=300, figsize=[5.5, 5.5])
plt.figure()
#
# 平方距離
squared_distance = lambda x, y: np.array([[(x[i] - y[j])**2 for i in range(len(x))] for j in range(len(y))])
#
# ガウス過程におけるハイパーパラメータの推論
with pm.Model() as GP:
  mu = np.zeros(N)
  eta = pm.HalfCauchy('eta', 5)
  rho = pm.HalfCauchy('rho', 5)
  sigma = pm.HalfCauchy('sigma', 5)
  D = squared_distance(x, x)
  K = tt.fill_diagonal(eta * pm.math.exp(-rho * D), eta + sigma)
  obs = pm.MvNormal('obs', mu, cov=K, observed=y)
  test_points = np.linspace(0, 10, 100)
  D_pred = squared_distance(test_points, test_points)
  D_off_diag = squared_distance(x, test_points)
  K_oo = eta * pm.math.exp(-rho * D_pred)
  K_o = eta * pm.math.exp(-rho * D_off_diag)
  mu_post = pm.Deterministic('mu_post', pm.math.dot(pm.math.dot(K_o, tt.nlinalg.matrix_inverse(K)), y))
  SIGMA_post = pm.Deterministic('SIGMA_post', K_oo - pm.math.dot(pm.math.dot(K_o, tt.nlinalg.matrix_inverse(K)), K_o.T))
  trace = pm.sample(1000, chains=1)
varnames = ['eta', 'rho', 'sigma']
chain = trace[100:]
pm.traceplot(chain, varnames)
plt.savefig('GP2.png', dpi=300, figsize=[5.5, 5.5])
plt.figure()
#
# 要約統計量の出力
print("tableGP")
print(pm.summary(chain, varnames).round(4))
#
# 事後予測チェック
y_pred = [np.random.multivariate_normal(m, S) for m,S in zip(chain['mu_post'][::5], chain['SIGMA_post'][::5])]
for yp in y_pred:
  plt.plot(test_points, yp, 'r-', alpha=0.1)
plt.plot(x, y, 'bo')
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$f(x)$', fontsize=16, rotation=0)
plt.savefig('GP3.png', dpi=300, figsize=[5.5, 5.5])
plt.figure()