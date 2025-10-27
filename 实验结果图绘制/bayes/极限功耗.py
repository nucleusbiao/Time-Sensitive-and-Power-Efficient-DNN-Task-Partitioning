import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

# plt.style.use('_mpl-gallery')
#4_6_5
data=pd.read_excel("data/data25.3.5极限画图.xlsx")

netname="resnet"
x_name=netname+'time'
y_name=netname+"cputotal"
# z_name=netname+"partitionpoint"

x=data[x_name]#180-
y=data[y_name]
# z=data[z_name]

# fig, ax = plt.subplots(2,1,figsize=(8, 5))
fig,ax=plt.subplots(figsize=(8, 5))
ax.plot(x, y,linewidth=0.9)

ax.axvline(x=180,ls="--",color='#dd201c')
ax.axvline(x=960,ls="--",color='#dd201c')

ax.set_ylabel("Latency(s)",fontsize = 16)
# ax[0].set_xlabel("Time(s)",fontsize = 16)

labels = ax.get_xticklabels()
# 设置新的刻度标签，调整字体大小
ax.set_xticklabels(labels, fontsize=14)
labels = ax.get_yticklabels()
# 设置新的刻度标签，调整字体大小
ax.set_yticklabels(labels, fontsize=14)

if netname =="resnet":
    ax.text(10, 70, '55Mbps', fontsize=14)
    ax.text(240, 70, '1Mbps',fontsize=14)
    ax.text(1010, 70, '55Mbps',fontsize=14)

if netname =="yolo":
    ax.text(10, 150, '55Mbps', fontsize=14)
    ax.text(240, 150, '1Mbps',fontsize=14)
    ax.text(1010, 150, '55Mbps',fontsize=14)

if netname=="alexnet":
    ax.text(10, 15, '55Mbps', fontsize=14)
    ax.text(240, 15, '1Mbps',fontsize=14)
    ax.text(1010, 15, '55Mbps',fontsize=14)

# ax[1].scatter(x, z,linewidth=0.9)
# ax[1].axvline(x=180,ls="--",color='#dd201c')
# ax[1].axvline(x=960,ls="--",color='#dd201c')
# ax[1].set_ylabel("Partition point",fontsize = 16)
# ax[1].set_xlabel("Time(s)",fontsize = 16)

# if netname =="yolo":
#     ax[1].set_ylim(-1,33)
# if netname =="alexnet":
#     ax[1].set_ylim(-1,12)
# if netname =="resnet":
#     ax[1].set_ylim(-1,22)

# labels = ax[1].get_xticklabels()
# ax[1].set_xticklabels(labels, fontsize=14)
# labels = ax[1].get_yticklabels()
# ax[1].set_yticklabels(labels, fontsize=14)

savepath="limit_power"+netname+".eps"
# plt.savefig("latnecy_resnet.pdf")
plt.savefig(savepath)
plt.show()



