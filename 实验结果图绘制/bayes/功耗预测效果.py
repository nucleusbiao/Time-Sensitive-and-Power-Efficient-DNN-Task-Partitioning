import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#数据来源：data15.1.4功耗预测效果，树莓派k数据
data=pd.read_excel("data/25.1.9for画图功耗预测效果.xlsx")
#一个任务 1 
r1=data["powerreal1"]
p1=data["powerpredicted1"]
# # #3个任务 
# r1=data["powerreal3"]
# p1=data["powerpredicted3"]
#4个任务 
# r1=data["powerreal4"]
# p1=data["powerpredicted4"]


x=range(0,len(r1))
# plt.figure(figsize=(8, 3), dpi=80)
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(x,r1,marker='.', markersize=8,linewidth=3,label='Real Power')
ax.plot(x,p1,marker='.',markersize=8,linewidth=3,label='Predicted Power',linestyle='--')
ax.set_ylabel('Power(w)',fontsize = 20)
ax.set_xlabel('Iterations',fontsize = 20)
labels = ax.get_xticklabels()
# 设置新的刻度标签，调整字体大小
ax.set_xticklabels(labels, fontsize=20)

labels = ax.get_yticklabels()
# 设置新的刻度标签，调整字体大小
ax.set_yticklabels(labels, fontsize=20)


# plt.plot(x,test_power,marker='.',markersize=5,label='real power',alpha=0.5)

plt.legend(loc="lower right",prop = {'size':20})
plt.show()


