import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

# plt.style.use('_mpl-gallery')

data=pd.read_excel("data\data25.3.3画收敛.xlsx")


d1=data["dqn1"]
d2=data["dqn2"]
d3=data["dqn3"]
d4=data["dqn4"]
d5=data["dqn5"]
d=np.array([d1,d2,d3,d4,d5])
d_mean=np.mean(d,axis=0)
d_std = np.std(d, axis=0)
d_max=d_mean+d_std*0.95
d_min=d_mean-d_std*0.95
x = np.linspace(0,d_mean.shape[0]*4.85,d_mean.shape[0]) # 生成 个数据点的横坐标
# fig = plt.figure(1,(5,3))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, d_mean, label='D3QN', color='#e75840',linewidth=0.9)
ax.fill_between(x, d_max, d_min, alpha=0.1, facecolor='#e75840')


d1=data["ppo1"]
d2=data["ppo2"]
d3=data["ppo3"]
d=np.array([d1,d2,d3])
d_mean=np.mean(d,axis=0)
d_std = np.std(d, axis=0)
d_max=d_mean+d_std*0.95
d_min=d_mean-d_std*0.95
# x = np.linspace(0,d_mean.shape[0],d_mean.shape[0]) # 生成 个数据点的横坐标
x = np.linspace(0,d_mean.shape[0]*4.85,d_mean.shape[0]) # 生成 个数据点的横坐标

ax.plot(x, d_mean, label='PPO', color='#628cee',linewidth=0.9)
ax.fill_between(x, d_max, d_min, alpha=0.1, facecolor='#628cee')


ax.set_ylabel("Reward",fontsize = 16)
ax.set_xlabel("Time(s)",fontsize = 16)

labels = ax.get_xticklabels()
# 设置新的刻度标签，调整字体大小
ax.set_xticklabels(labels, fontsize=16)
labels = ax.get_yticklabels()
# 设置新的刻度标签，调整字体大小
ax.set_yticklabels(labels, fontsize=16)

ax.legend(loc="upper right",prop = {'size':16})
# plt.grid(True)
plt.savefig("Convergence_reward.pdf")
plt.show()
