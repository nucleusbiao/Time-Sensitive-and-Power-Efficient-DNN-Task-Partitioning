# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ##预测误差###  变带宽，4.1.1
# species = ("VGG","AlexNet","Yolo","ResNet","MobileNet")
# # penguin_means = {
# #     'AdaNS': (0.0438,0.0564,0.0243,0.0407,0.0666),
# #     'AutoNS': (0.0484, 0.0533, 0.0414,0.0689,0.0711),
# #     }

# penguin_means = {
#     'Gauss': (0.061,0.0015,0.0399,0.0542,0.0523),
#     'AutoNS': (0.2111,0.0668,0.5446,0.1191,0.1738,),
#     }


# x = np.arange(len(species))  # the label locations
# width = 0.45  # the width of the bars3
# multiplier = 0

# fig, ax = plt.subplots(layout='constrained',figsize=(5, 3))

# for attribute, measurement in penguin_means.items():
#     offset = width * multiplier
#     if attribute=='AutoNS':
#         rects = ax.bar(1.2*x + offset, measurement, width, label=attribute, hatch="//",linewidth=0.01,facecolor='tab:blue',edgecolor='0.4')
#     else:
#         rects = ax.bar(1.2*x + offset, measurement, width, label=attribute,color='tab:orange')
                       
#     ax.bar_label(rects, padding=1,fontsize = 7)
#     multiplier += 1

# # Add some text for labels, title and custom x-axis tick labels, etc.
# #ax.set_xlabel(gront)
# ax.set_ylabel('error (s)',fontsize = 12)
# ax.set_title('Prediction Error')
# ax.set_xticks(1.2*x + 0.5*width, species,fontsize = 10)
# ax.legend(loc='upper left', ncols=1)
# ax.set_ylim(0, 0.58)

# plt.show()



# ##预测误差2   ### 多设备运行网络数量1-6变化  4.1.2
# species = ("Tiny-Yolo","ResNet","AlexNet","MobileNet","AlexNet","VGG")
# penguin_means = {
#     'Gauss': (0.03,0.106,0.002,0.074,0.031,0.079),
#     'AutoNS': (0.271,0.473,0.156,0.695,3.414,0.215),
#     }
# x = np.arange(len(species))  # the label locations
# width = 0.45  # the width of the bars3
# multiplier = 0

# fig, ax = plt.subplots(layout='constrained',figsize=(5, 3))

# for attribute, measurement in penguin_means.items():
#     offset = width * multiplier
#     if attribute=='AutoNS':
#         rects = ax.bar(1.2*x + offset, measurement, width, label=attribute, hatch="//",linewidth=0.01,facecolor='tab:blue',edgecolor='0.4')
#     else:
#         rects = ax.bar(1.2*x + offset, measurement, width, label=attribute,color='tab:orange')
                       
#     ax.bar_label(rects, padding=1,fontsize = 7)
#     multiplier += 1

# # Add some text for labels, title and custom x-axis tick labels, etc.
# #ax.set_xlabel(gront)
# ax.set_ylabel('error (s)',fontsize = 12)
# ax.set_title('Prediction Error')
# ax.set_xticks(1.2*x + 0.5*width, species,fontsize = 10)
# ax.legend(loc='upper left', ncols=1)
# ax.set_ylim(0, 3.6)

# plt.show()











# ##average time per frame  bandrand 1.1.3
# species = ("AlexNet","ResNet","Yolo","MobileNet","VGG",)
# penguin_means = {
#     'Gauss': (0.105,0.384,0.474,0.472,0.391),
#     'AutoNS': (0.230,1.412,0.789,0.479,0.726,),
#     'SOP': (0.390,0.425,0.723,0.442,0.400,),
#     'Local': (0.232,1.612,0.482,5.930,5.983,),
#     'Server': (0.814,0.754,1.839,0.778,0.831)}
# x = np.arange(len(species))  # the label locations
# width = 0.15  # the width of the bars3
# multiplier = 0

# fig, ax = plt.subplots(layout='constrained',figsize=(5, 3))

# for attribute, measurement in penguin_means.items():
#     offset = width * multiplier
#     colors = ['#3682be','#45a776','#f05326','#eed777','#334f65','#b3974e','#38cb7d','#ddae33','#844bb3','#93c555','#5f6694','#df3881']
#     if attribute=='Gauss':
#         rects = ax.bar(x + offset, measurement, width, label=attribute,color=colors[0],hatch="//",linewidth=0.01,edgecolor='0.3')
#     if attribute=='AutoNS':
#         rects = ax.bar(x + offset, measurement, width, label=attribute,facecolor=colors[4],hatch="+++",linewidth=0.01,edgecolor='0.3')
#     if attribute=='SOP':
#          rects = ax.bar(x + offset, measurement, width, label=attribute,color=colors[2],hatch="--",linewidth=0.01,edgecolor='0.4') #yellowgreen
#     if attribute=='Local':
#         rects = ax.bar(x + offset, measurement, width, label=attribute,color=colors[1],hatch="\\\\",linewidth=0.01,edgecolor='0.3')
#     if attribute=='Server':
#          rects = ax.bar(x + offset, measurement, width, label=attribute,color=colors[9],hatch="xx",linewidth=0.01,edgecolor='0.4')    

#     # if attribute=='AdaNS':
#     #     rects = ax.bar(x + offset, measurement, width, label=attribute,color=(.99,.67,.40))
#     # if attribute=='AutoNS':
#     #     rects = ax.bar(x + offset, measurement, width, label=attribute,facecolor=(.00,.00,.78))
#     # if attribute=='SOP':
#     #      rects = ax.bar(x + offset, measurement, width, label=attribute,color=(.33,.53,.53)) #yellowgreen
#     # if attribute=='Local':
#     #     rects = ax.bar(x + offset, measurement, width, label=attribute,color=(.28,.20,.21))
#     # if attribute=='Server':
#     #      rects = ax.bar(x + offset, measurement, width, label=attribute,color=(.00,.67,.20))
#     #rects = ax.bar(x + offset, measurement, width, label=attribute)          
#     #ax.bar_label(rects, padding=5,fontsize = 7)
#     multiplier += 1

# # Add some text for labels, title and custom x-axis tick labels, etc.
# #ax.set_xlabel(gront)
# ax.set_ylabel('time (s)',fontsize = 12)
# ax.set_title('Average Latency Per Frame')
# ax.set_xticks(x +2*width, species,fontsize =10)
# ax.legend(loc='upper left', ncols=2)
# ax.set_ylim(0, 6.1)

# plt.show()






# ##average time per frame k  bandrand 1.1.5
# species = ("AlexNet","ResNet","Yolo","MobileNet","VGG",)
# penguin_means = {
#     'Gauss': (0.344,0.913,0.377,0.385,0.354),
#     'AutoNS': (0.522,1.588,0.815,0.821,1.006),
#     'SOP':   (0.432,1.132,0.480,0.457,0.481),
#     'Local': (1.537,2.873,4.918,10.946,13.902),
#     'Server': (0.555,1.343,0.668,0.638,0.644)}
# x = np.arange(len(species))  # the label locations
# width = 0.15  # the width of the bars3
# multiplier = 0

# fig, ax = plt.subplots(layout='constrained',figsize=(5, 3))

# for attribute, measurement in penguin_means.items():
#     offset = width * multiplier
#     colors = ['#3682be','#45a776','#f05326','#eed777','#334f65','#b3974e','#38cb7d','#ddae33','#844bb3','#93c555','#5f6694','#df3881']
#     if attribute=='Gauss':
#         rects = ax.bar(x + offset, measurement, width, label=attribute,color=colors[0],hatch="//",linewidth=0.01,edgecolor='0.3')
#     if attribute=='AutoNS':
#         rects = ax.bar(x + offset, measurement, width, label=attribute,facecolor=colors[4],hatch="+++",linewidth=0.01,edgecolor='0.3')
#     if attribute=='SOP':
#          rects = ax.bar(x + offset, measurement, width, label=attribute,color=colors[2],hatch="--",linewidth=0.01,edgecolor='0.4') #yellowgreen
#     if attribute=='Local':
#         rects = ax.bar(x + offset, measurement, width, label=attribute,color=colors[1],hatch="\\\\",linewidth=0.01,edgecolor='0.3')
#     if attribute=='Server':
#          rects = ax.bar(x + offset, measurement, width, label=attribute,color=colors[9],hatch="xx",linewidth=0.01,edgecolor='0.4')    
#     multiplier += 1

# # Add some text for labels, title and custom x-axis tick labels, etc.
# #ax.set_xlabel(gront)
# ax.set_ylabel('time (s)',fontsize = 12)
# ax.set_title('Average Latency Per Frame')
# ax.set_xticks(x +2*width, species,fontsize =10)
# ax.legend(loc='upper left', ncols=2)
# ax.set_ylim(0, 14.1)

# plt.show()








# ##average time per frame k  task add 2.2
# species = ("AlexNet","ResNet","Yolo","MobileNet","VGG","AlexNet_2")

# penguin_means = {
#     'Gauss': (0.127,0.264,0.614,0.485,0.553,0.164),
#     'AutoNS': (0.172,0.261,1.095,0.381,0.986,0.317),
#     'SOP':   (0.192,0.350,1.042,0.499,0.369,0.426),
#     'Local': (0.309,3.683,1.236,5.050,2.001,0.506),
#     'Server': (0.248,0.364,0.752,0.567,0.513,0.411)}
# x = np.arange(len(species))  # the label locations
# width = 0.15  # the width of the bars3
# multiplier = 0

# fig, ax = plt.subplots(layout='constrained',figsize=(5, 3))

# for attribute, measurement in penguin_means.items():
#     offset = width * multiplier
#     colors = ['#3682be','#45a776','#f05326','#eed777','#334f65','#b3974e','#38cb7d','#ddae33','#844bb3','#93c555','#5f6694','#df3881']
#     if attribute=='Gauss':
#         rects = ax.bar(x + offset, measurement, width, label=attribute,color=colors[0],hatch="//",linewidth=0.01,edgecolor='0.3')
#     if attribute=='AutoNS':
#         rects = ax.bar(x + offset, measurement, width, label=attribute,facecolor=colors[4],hatch="+++",linewidth=0.01,edgecolor='0.3')
#     if attribute=='SOP':
#          rects = ax.bar(x + offset, measurement, width, label=attribute,color=colors[2],hatch="--",linewidth=0.01,edgecolor='0.4') #yellowgreen
#     if attribute=='Local':
#         rects = ax.bar(x + offset, measurement, width, label=attribute,color=colors[1],hatch="\\\\",linewidth=0.01,edgecolor='0.3')
#     if attribute=='Server':
#          rects = ax.bar(x + offset, measurement, width, label=attribute,color=colors[9],hatch="xx",linewidth=0.01,edgecolor='0.4')    
#     multiplier += 1

# # Add some text for labels, title and custom x-axis tick labels, etc.
# #ax.set_xlabel(gront)
# ax.set_ylabel('time (s)',fontsize = 12)
# ax.set_title('Average Latency Per Frame')
# ax.set_xticks(x +2*width, species,fontsize =10)
# ax.legend(loc='upper left', ncols=1)
# ax.set_ylim(0, 5.2)

# plt.show()







##average time per frame bandrand and task add 3.1
# species = ("AlexNet","MobileNet","Yolo","ResNet","VGG","Yolo_2","Yolo_3","ResNet_2","AlexNet_2","MobileNet_2",)

# penguin_means = {
#     'Gauss': (0.222,0.258,1.012,0.367,0.674,0.519,0.450,0.677,0.259,0.384,),
#     'AutoNS': (0.249,0.310,1.190,0.998,1.189,0.921,1.077,1.069,0.267,0.514,),
#     'SOP':   (0.239,0.308,0.744,0.239,0.487,0.894,0.819,0.490,0.349,0.425,),
#     'Local': (0.447,2.443,1.292,5.335,2.074,0.465,0.455,1.191,0.258,1.272,),
#     'Server': (0.417,0.507,1.121,0.667,0.646,1.819,1.692,0.796,0.545,0.631)}
# x = np.arange(len(species))  # the label locations
# width = 0.15  # the width of the bars3
# multiplier = 0

# fig, ax = plt.subplots(layout='constrained',figsize=(5, 3))

# for attribute, measurement in penguin_means.items():
#     offset = width * multiplier
#     colors = ['#3682be','#45a776','#f05326','#eed777','#334f65','#b3974e','#38cb7d','#ddae33','#844bb3','#93c555','#5f6694','#df3881']
#     if attribute=='Gauss':
#         rects = ax.bar(x + offset, measurement, width, label=attribute,color=colors[0],hatch="//",linewidth=0.01,edgecolor='0.3')
#     if attribute=='AutoNS':
#         rects = ax.bar(x + offset, measurement, width, label=attribute,facecolor=colors[4],hatch="+++",linewidth=0.01,edgecolor='0.3')
#     if attribute=='SOP':
#          rects = ax.bar(x + offset, measurement, width, label=attribute,color=colors[2],hatch="--",linewidth=0.01,edgecolor='0.4') #yellowgreen
#     if attribute=='Local':
#         rects = ax.bar(x + offset, measurement, width, label=attribute,color=colors[1],hatch="\\\\",linewidth=0.01,edgecolor='0.3')
#     if attribute=='Server':
#          rects = ax.bar(x + offset, measurement, width, label=attribute,color=colors[9],hatch="xx",linewidth=0.01,edgecolor='0.4')    
#     multiplier += 1

# # Add some text for labels, title and custom x-axis tick labels, etc.
# #ax.set_xlabel(gront)
# ax.set_ylabel('time (s)',fontsize = 12)
# ax.set_title('Average Latency Per Frame')
# ax.set_xticks(x +2*width, species,fontsize =10)
# ax.legend(loc='upper left', ncols=1)
# plt.xticks(rotation=45)
# ax.set_ylim(0, 5.7)

# plt.show()







# #波动散点图 yolo

# data = pd.read_csv('E:\\深度deepproject\\data24.8.15-\\7.csv')
# x1=data.values[:,0]
# y1=data.values[:,1]
# x2=data.values[:,2]
# y2=data.values[:,3]

# # 绘制第一组数据  
# plt.scatter(x1, y1, color='red',label='55Mbps',marker=",",alpha=0.5)  # 使用红色点表示第一组数据  

# # 绘制第二组数据  
# plt.scatter(x2, y2, color='blue',label='7Mbps',alpha=0.5)  # 使用蓝色点表示第二组数据  

# # 添加标题和坐标轴标签  
# # plt.title('Latency of VGG under different wireless bandwidth',size=14)  
# plt.xlabel('Partition Point',size=14)  
# plt.ylabel('Latency/s',size=14)  
# plt.legend(loc='upper right', ncols=1)
# # 显示图形  
# plt.show()  




# ##波动箱线图 yolo
# import pandas as pd
# data = pd.read_csv('E:\\深度deepproject\\data24.8.15-\\7.csv')

# x1=data.values[:,0]
# y1=data.values[:,1]
# x2=data.values[:,2]
# y2=data.values[:,3]

# #数据处理。
# data1=[[] for i in range(32)]
# for i in range(len(x1)):
#     data1[int(x1[i])].append(y1[i])

# data2=[[] for i in range(32)]
# for i in range(64):
#         data2[int(x2[i])].append(y2[i])

# fig, ax = plt.subplots()
# VP1 = ax.boxplot(data1,widths=0.7, patch_artist=True,showmeans=False,
#                  showfliers=False,notch=True,
#                 medianprops={"color": "midnightblue", "linewidth": 0.5},
#                 boxprops={"facecolor": "blue", "edgecolor": "blue",
#                           "linewidth": 0.5,"linestyle":'--'},
#                 whiskerprops={"color": "blue", "linewidth": 1.5},
#                 capprops={"color": "blue", "linewidth": 1.5},
#                 )

# VP2 = ax.boxplot(data2,patch_artist=True,widths=0.7,
#                 showmeans=False, showfliers=False,
#                 medianprops={"color": "white", "linewidth": 0.5},
#                 boxprops={"facecolor": 'red', "edgecolor": "red",
#                           "linewidth": 0.5},
#                 whiskerprops={"color": 'red', "linewidth": 1.5},
#                 capprops={"color": 'red', "linewidth": 1.5},
#                 )



# p=[i for i in range(1,33)]
# a55=[0.4175994,
# 1.372241139,
# 1.33751628,
# 1.379963061,
# 0.516482512,
# 0.924847086,
# 0.918415387,
# 0.896297375,
# 0.469501177,
# 0.698641883,
# 0.668327265,
# 0.644392067,
# 0.456717955,
# 0.605229378,
# 0.571326574,
# 0.558379345,
# 0.421020627,
# 0.48797365,
# 0.495254,
# 0.525628037,
# 0.435361067,
# 0.538834903,
# 0.520842301,
# 0.543309583,
# 0.532634444,
# 0.906859676,
# 0.937403785,
# 0.962741216,
# 1.646419009,
# 1.671168619,
# 1.665138258,
# 1.430148429,]
# ax.plot(p,a55)

# ax.set(xticks=np.arange(1, 33),xticklabels=np.arange(0, 32))
# # ax.set_ylim(0, 13)
# ax.grid()
# ax.legend() #lables=['55Mbps','7Mbps'],loc='upper right', ncols=2
# ax.set_xlabel('Partition Point')
# ax.set_ylabel('Latency/s')
# plt.show()








##average time per frame  addDNN
# species = ("VGG","MobileNet","AlexNet","ResNet","Yolo","AlexNet_2")
# penguin_means = {
#     'AdaNS': (0.465,0.763,0.478,0.460,1.531,0.747),
#     'AutoNS': (1.522,1.538,1.319,1.055,1.935,1.184),
#     'SOP': (0.610,0.589,0.554,0.609,2.857,0.845),
#     'Local': (6.563,1.674,0.777,4.118,2.158,0.938),
#     'Server': (0.694,0.733,0.866,0.652,3.571,1.071)
# }
# x = np.arange(len(species))  # the label locations
# width = 0.18  # the width of the bars3
# multiplier = 0

# fig, ax = plt.subplots(layout='constrained',figsize=(5, 3))

# for attribute, measurement in penguin_means.items():
#     offset = width * multiplier
    
#     if attribute=='AdaNS':
#         rects = ax.bar(x + offset, measurement, width, label=attribute,color=(.94,.39,.29),hatch="//",linewidth=0.01,edgecolor='0.3')
#     if attribute=='AutoNS':
#         rects = ax.bar(x + offset, measurement, width, label=attribute,facecolor=(.00,.67,.20),hatch="\\\\",linewidth=0.01,edgecolor='0.3')
#     if attribute=='SOP':
#          rects = ax.bar(x + offset, measurement, width, label=attribute,color=(.02,.31,.36),hatch="+++",linewidth=0.01,edgecolor='0.4') #yellowgreen
#     if attribute=='Local':
#         rects = ax.bar(x + offset, measurement, width, label=attribute,color=(.19,.59,.64),hatch="--",linewidth=0.01,edgecolor='0.3')
#     if attribute=='Server':
#          rects = ax.bar(x + offset, measurement, width, label=attribute,color=(.24,.17,.43),hatch="xx",linewidth=0.01,edgecolor='0.4')    

#     # if attribute=='AdaNS':
#     #     rects = ax.bar(x + offset, measurement, width, label=attribute,color=(.99,.67,.40))
#     # if attribute=='AutoNS':
#     #     rects = ax.bar(x + offset, measurement, width, label=attribute,facecolor=(.00,.00,.78))
#     # if attribute=='SOP':
#     #      rects = ax.bar(x + offset, measurement, width, label=attribute,color=(.33,.53,.53)) #yellowgreen
#     # if attribute=='Local':
#     #     rects = ax.bar(x + offset, measurement, width, label=attribute,color=(.28,.20,.21))
#     # if attribute=='Server':
#     #      rects = ax.bar(x + offset, measurement, width, label=attribute,color=(.00,.67,.20))
#     #rects = ax.bar(x + offset, measurement, width, label=attribute)          
#     #ax.bar_label(rects, padding=5,fontsize = 7)
#     multiplier += 1

# # Add some text for labels, title and custom x-axis tick labels, etc.
# #ax.set_xlabel(gront)
# ax.set_ylabel('time (s)',fontsize = 12)
# ax.set_title('Average Latency Per Frame')
# ax.set_xticks(x +2*width, species,fontsize =10)
# ax.legend(loc='upper right', ncols=2)
# ax.set_ylim(0.0, 6.7)

# plt.show()








#3.1 task 甘特图    ####https://blog.csdn.net/sgzqc/article/details/121893158
# colors = ['#3682be','#45a776','#f05326','#eed777','#334f65','#b3974e','#38cb7d','#ddae33','#844bb3','#93c555','#5f6694','#df3881']
# def color(row):
#     c_dict = {'AlexNet':'#E64646', 'MobileNet':'#E69646', 'Yolo':'#34D05C', 'ResNet':'#34D0C3', 'VGG':'#3475D0','Yolo_2':'#3682be',
#               'Yolo_3':'#45a776','ResNet_2':'#f05326','AlexNet_2':'#eed777','MobileNet_2':'#334f65'
# }
#     return c_dict[row['Task']]
# df = pd.read_excel('E:\\深度deepproject\\data24.8.15-\\gante.xlsx')
# df['color'] = df.apply(color, axis=1)
# # fig, ax = plt.subplots(1, figsize=(5,1))
# fig, ax = plt.subplots(1,)
# ax.barh(df.Task,df.length, left=df.Start,color=df.color)

# ax.grid(which='major',axis='x')
# ax.set_xlabel('time/min',size=14)
# # ax.set_yticks([0,1,2,3,4,5,6,7,8,9],lablels=['MobileNet_2','AlexNet_2','ResNet_2','Yolo_3','Yolo_2','VGG','ResNet','Yolo','MobileNet','AlexNet'])
# # ax.set_ylabel(size=14)
# plt.show()



#3.1 bandwidth 折线图
# bandwidthlist=[55,7,7,50,5,15,5,5]
# x=[i for i in range(8)]
# plt.step(x,bandwidthlist,where='post')
# plt.xlabel('time/min',size=14)
# plt.ylabel('Mbps',size=14)
# plt.show()















#5.1 bandwidth 折线图
bandwidthlist=[20, 15, 5, 15, 7, 10, 10, 20, 30, 7, 20, 15, 7, 5, 5, 30, 7, 15, 10, 7, 7, 7, 5, 25, 20, 7, 15, 30, 10, 20, 25, 5]
x=[i for i in range(len(bandwidthlist))]
plt.step(x,bandwidthlist,where='post')
plt.xlabel('time/min',size=14)
plt.ylabel('Mbps',size=14)
plt.show()



#5.1 task gantt
# colors = ['#3682be','#45a776','#f05326','#eed777','#334f65','#b3974e','#38cb7d','#ddae33','#844bb3','#93c555','#5f6694','#df3881']
# def color(row):
#     c_dict = {'AlexNet':'#E64646', 'MobileNet':'#E69646', 'Yolo':'#34D05C', 'ResNet':'#34D0C3', 'VGG':'#3475D0','Yolo_2':'#3682be',
#               'Yolo_r1':'#45a776','ResNet_r1':'#f05326','AlexNet_r1':'#eed777','MobileNet_r1':'#334f65','Yolo2_r1':'#45a776','VGG_r1':'#3475D0',
#               'Yolo_r2':'#45a776','ResNet_r2':'#f05326','AlexNet_r2':'#eed777','MobileNet_r2':'#334f65','Yolo2_r2':'#45a776','VGG_r2':'#3475D0',
#               'Yolo_tx':'#45a776','ResNet_tx':'#f05326','AlexNet_tx':'#eed777','MobileNet_tx':'#334f65','VGG_tx':'#3475D0',
#               'Yolo_hs':'#45a776','ResNet_hs':'#f05326','AlexNet_hs':'#eed777','MobileNet_hs':'#334f65','VGG_hs':'#3475D0',
#     }
#     return c_dict[row['Task']]
# df = pd.read_excel('E:\\深度deepproject\\data24.8.15-\\gant5.xlsx')
# df['color'] = df.apply(color, axis=1)
# # fig, ax = plt.subplots(1, figsize=(5,1))
# fig, ax = plt.subplots(1,)
# ax.barh(df.Task,df.length, left=df.Start,color=df.color)

# ax.grid(which='major',axis='x')
# ax.set_xlabel('time/min',size=14)
# # ax.set_yticks([0,1,2,3,4,5,6,7,8,9],lablels=['MobileNet_2','AlexNet_2','ResNet_2','Yolo_3','Yolo_2','VGG','ResNet','Yolo','MobileNet','AlexNet'])
# # ax.set_ylabel(size=14)
# plt.show()
































# ###*********ax.grid()**********######
# ax.grid() 是 Matplotlib 库中的一个方法，用于在绘图中添加网格线。该方法接受几个参数，例如：
# which: 指定网格线类型，可以是 "major"、"minor"、"both" 或 "none"，默认值为 "major"。
# axis: 指定网格线方向，可以是 "x"、"y" 或 "both"，默认值为 "both"。
# color: 指定网格线颜色。
# linestyle: 指定网格线样式，例如 "-"、"--"、":" 等。
# linewidth: 指定网格线宽度。
# 例如，以下代码将创建一个带有横向和纵向网格线的绘图：
# import matplotlib.pyplot as plt  
# # 创建绘图  
# plt.plot([1, 2, 3], [4, 5, 6])  
# # 添加网格线  
# plt.grid(True)  
# # 显示绘图  
# plt.show()  


















