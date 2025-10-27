
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

# # # 设置字体  
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  
matplotlib.rcParams['axes.unicode_minus'] = False 

filepath="E:\深度deepproject\data24.8.15-\\data24.11.26_pp.xlsx"  #单个静态数据。
# dnnname="Yolo"
# dnnname="ResNet"
dnnname="Vgg"
typename="kurtosis"  #峰度
# typename="skewness"  #偏度
sheetname=dnnname+"_"+typename
data = pd.read_excel(filepath, sheet_name = sheetname)
num_columns=len(data.columns)

species = [str(i) for i in range(len(data["client"]))]

penguin_means = {
    '设备计算': data['client'],
    '传输': data['trans'],
    '边缘计算': data['server'],
}

x = np.arange(len(species))  # the label locations
width = 0.28  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
fig.set_size_inches(5.5, 3.3)  #resnet vgg

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    # rects = ax.bar(x + offset, measurement, width, label=attribute)
    rects = ax.bar(x + offset, measurement, width,label=attribute)
    # ax.bar_label(rects, padding=3)
    multiplier += 1

x_lable=species 
if dnnname=="Vgg":
    x_lable=("input","conv1","conv2","maxpool","conv3","conv4","maxpool","conv5",
    "conv6","conv7","maxpool","conv8","conv9","conv10","maxpool",
    "conv11","conv12","conv13","maxpool","avepool","fc","fc","fc")
    # x_lable=("input","conv3-64","conv3-64","maxpool","conv3-128","conv3-128","maxpool","conv3-256",
    # "conv3-256","conv3-256","maxpool","conv3-512","conv3-512","conv3-512","maxpool",
    # "conv3-512","conv3-512","conv3-512","maxpool","avepool","fc","fc","fc")
elif dnnname=="ResNet":
    x_lable=("input","conv1","maxpool","conv2","conv3","conv4","conv5","conv6","conv7","conv8",
    "conv9","conv10","conv11","conv12","conv13","conv14","conv15","conv16","conv17","avepool","fc")
    # x_lable=("input","conv1","maxpool","conv2","conv2","conv2","conv3","conv3","conv3","conv3",
    # "conv4","conv4","conv4","conv4","conv4","conv4","conv5","conv5","conv5","avepool","fc")
    # x_lable=("input","conv1","maxpool","conv2_1","conv2_2","conv2_3","conv3_1","conv3_2","conv3_3","conv3_4",
    # "conv4_1","conv4_2","conv4_3","conv4_4","conv4_5","conv4_6","conv5_1","conv5_2","conv5_3","avepool","fc")
# elif dnnname=="Yolo":
#     x_lable=('input','conv1','bn1','leaky1','pool1','conv2','bn2','leaky2','pool2','conv3','bn3','leaky3','pool3',   #13
#     'conv4','bn4','leaky4','pool4','conv5','bn5','leaky5','pool5','conv6','bn6','leaky6','pool6',     #12
#     'conv7','bn7','leaky7','conv8','bn8','leaky8','conv9')   #7
elif dnnname=="Yolo":
    x_lable=('       input','       conv1','       bn1','leaky1','pool1','conv2','bn2','leaky2','pool2','conv3','bn3','leaky3','pool3',   #13
    'conv4','bn4','leaky4','pool4','conv5','bn5','leaky5','pool5','conv6','bn6','leaky6','pool6',     #12
    'conv7','bn7','leaky7','conv8','bn8','leaky8','conv9')   #7

# Add some text for labels, title and custom x-axis tick labels, etc.
if typename=="skewness":
    ax.set_ylabel("偏度",fontsize = 16)
else:
    ax.set_ylabel("峰度",fontsize = 16)
title="Latency " +typename+ " of "+dnnname
# ax.set_title(title,fontsize = 14)


if dnnname!="Yolo":
    ax.set_xticks(x + width, x_lable,rotation=90,fontsize=15)  #resnet vgg
else:
    ax.set_xticks(x + width, x_lable,rotation=90,fontsize=12)  #yolo

# if dnnname=="ResNet" :
#     locs='upper right'
# else:
#     locs='upper left'
locs='upper left'
ax.legend(loc=locs, ncol=1, prop = {'size':13},framealpha=0.2)
# ax.set_ylim(0, 50)
if typename=="skewness":
    plt.plot([0, len(data["client"])], [3, 3], c='y', linestyle='--')
else:
    plt.plot([0, len(data["client"])], [10, 10], c='y', linestyle='--')


plt.savefig(title)
plt.show()




