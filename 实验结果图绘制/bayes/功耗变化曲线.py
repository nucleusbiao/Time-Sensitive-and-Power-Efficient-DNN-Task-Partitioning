import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#数据来源：树莓派k数据
data=pd.read_excel("data/data25.1.21收敛曲线.xlsx")

cmap=[plt.cm.Paired(1),plt.cm.Paired(6),plt.cm.Paired(2),plt.cm.Paired(3),plt.cm.Paired(4),plt.cm.Paired(5),plt.cm.Paired(0),plt.cm.Paired(7),plt.cm.Paired(8),plt.cm.Paired(9),plt.cm.Paired(10)]

bayes2=data["bayes_yolo"]
ga2=data["ga_yolo"]
parse=data["parse_yolo"] #
fig, ax = plt.subplots(figsize=(8, 5))
x=[0+4.85*i for i in range(157)]

ax.plot(x,bayes2,label='Bayes',color='#f6b57b')
ax.plot(x,parse,label='SparseB',color='#2b9fc9',linestyle='--')
ax.plot(x,ga2,label='GA',color='#e54d4c',linestyle='dotted')

ax.set_ylabel("Power(w)",fontsize = 16)
ax.set_xlabel("Time(s)",fontsize = 16)

labels = ax.get_xticklabels()
# 设置新的刻度标签，调整字体大小
ax.set_xticklabels(labels, fontsize=16)
labels = ax.get_yticklabels()
# 设置新的刻度标签，调整字体大小
ax.set_yticklabels(labels, fontsize=16)

ax.legend(loc="upper right",prop = {'size':16})
plt.savefig("Convergence_power.pdf")
plt.show()