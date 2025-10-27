import numpy as np
from scipy import stats
# 创建100个服从正太分布的数据集
# np.random.seed(0)
# data = np.random.normal(0,1, 1000)
# import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import math

# 创建 Q-Q 图，并增加 45度线
# fig = sm.qqplot(data, line='45')

sheetname=""
def readdata():
    global sheetname
    filepath="E:\深度deepproject\data24.8.15-\\data24.11.26_pp.xlsx"
    # dnnname="vgg"
    # dnnname="resnet"
    dnnname="yolo"
    # dnnname="alexnet"

    # partname="trans" 
    partname="client"
    # partname="server"

    # devicename="_tx2_"
    devicename="_k_"

    sheetname=dnnname+devicename+partname
    data = pd.read_excel(filepath, sheet_name = sheetname)#指定读取第一个sheet
    return data

# def sknewness(data):
#     n = len(data) #样本个数
#     average=np.mean(data) #计算平均值
#     m1=0
#     m2=0
#     k=math.sqrt(n*(n-1))/(n-2)
#     for i in data:
#         m1+=(i-average)**3
#         m2+=(i-average)**2
#     m1/=n
#     m2/=n
#     m2=math.sqrt(m2**3)
#     skewness=0
#     if m2!=0 :
#         skewness=k*m1/m2
#     else:
#         skewness=0
#     return skewness

# plt.ion()
# plt.show()
data=readdata() 
# print(data.head())

num_columns=len(data.columns)
for i in range(num_columns):
    stats.probplot(data,dist='norm',plot=plt)
    # skewness_D=sknewness(data[i])
    # skewness_D=stats.skew(data[i])  #计算偏度
    kurtosis=stats.kurtosis(data[i])  #计算峰度
    # plt.draw()
    # plt.title(str(i))
    # plt.pause(0.5)
    # savepath=sheetname+"\\"+sheetname+str(i)+".png"
    # plt.savefig(savepath)
    # plt.close()
    print(kurtosis)

   

    



