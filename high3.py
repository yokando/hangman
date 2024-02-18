import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo']

x = ['国語', '数学', '英吾', '化学u', '日本史']
y = [78, 65, 82, 73, 90]
plt.pie(y, labels=x)
plt.show()
