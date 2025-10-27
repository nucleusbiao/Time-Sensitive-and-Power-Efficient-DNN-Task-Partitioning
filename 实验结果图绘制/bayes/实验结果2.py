# data from https://allisonhorst.github.io/palmerpenguins/
#25.3.4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image



##1_3 average time per frame
species = ("Yolo_s1","VGG_s1","ResNet_t","AlexNet_t",)
penguin_means = {
    'Bayes': (0.774 ,1.326 ,2.106 ,0.721 ),
    'SparseB':(3.251 ,3.783 ,3.911 ,0.982 ),
    'D3QN':(2.488 ,5.078 ,5.289 ,0.447 ),
    'PPO':(3.832 ,6.072 ,8.166 ,0.511 ),
    'AutoNS':(1.16,2.43,1.59,0.23),
    'GA': (1.530 ,2.980 ,5.030 ,1.150 ),
    'Random':(2.75,5.87,4.40,0.65),
    'Local': (1.76,6.05,8.06,0.94),
    'Server': (7.81,2.21,2.85,2.85),}
x = np.arange(len(species))  # the label locations
width = 0.1  # the width of the bars3
multiplier = 0

fig, ax = plt.subplots(layout='constrained',figsize=(5, 3))

colormap = plt.get_cmap('Blues')
colors = plt.cm.viridis(np.linspace(0, 1, 14))
for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    if attribute=='Bayes':#"#E3716E","#F9D580","#CCA7AF"
        rects = ax.bar(x + offset, measurement, width, label=attribute,color="#F29500",hatch="/",linewidth=0.007,edgecolor='0.3')
    if attribute=='GA': #"#9FD4AE"
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#18B6B9",hatch="/",linewidth=0.001,edgecolor='0.3') #yellowgreen
    if attribute=='Local':
        rects = ax.bar(x + offset, measurement, width, label=attribute,color="#0D8B43",hatch="x",linewidth=0.01,edgecolor='0.3')
    if attribute=='Server':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#68BD48",hatch="\\",linewidth=0.01,edgecolor='0.3')  
    if attribute=='Random':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#9FD4AE",hatch="-",linewidth=0.01,edgecolor='0.3')       
    if attribute=='AutoNS':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#5560AC",hatch="//",linewidth=0.01,edgecolor='0.3')           
    if attribute=='SparseB':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#FAC074",linewidth=0.01,edgecolor='0.3')           
    if attribute=='D3QN':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color='#F28147',hatch="-",linewidth=0.01,edgecolor='0.3')           
    if attribute=='PPO':#""#567db0 #ec505d
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#428DBF",hatch="\\\\",linewidth=0.01,edgecolor='0.3')              
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
#ax.set_xlabel(gront)
ax.set_ylabel('Latency(s)',fontsize = 12)
#ax.set_title('Average Latency Per Frame')
ax.set_xticks(x +2*width, species,fontsize =10)
ax.legend(loc='upper right', ncols=3,framealpha=0.6)
ax.set_ylim(0.0, 10)
plt.savefig("latency_bandchange.pdf")
plt.show()



#1_3 average power per frame

species = ("s1","tx2")
penguin_means = {
    'Bayes': (4.193,7.539),
    'SparseB':(3.887,6.288),
    'D3QN':(4.467,10.108),
    'PPO':(4.806,7.734),
    'AutoNS':(5.080,10.330),
    'GA': (4.340,6.200),
    'Random':(4.533,7.629),
    'Local': (6.670,9.800),
    'Server': (3.280,3.830),
    }
x = np.arange(len(species))  # the label locations
width = 0.1  # the width of the bars3
multiplier = 0

fig, ax = plt.subplots(layout='constrained',figsize=(5, 3))

colormap = plt.get_cmap('Blues')
colors = plt.cm.viridis(np.linspace(0, 1, 14))
for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    if attribute=='Bayes':#"#E3716E","#F9D580","#CCA7AF"
        rects = ax.bar(x + offset, measurement, width, label=attribute,color="#F29500",hatch="/",linewidth=0.01,edgecolor='0.3')
    if attribute=='GA': #"#9FD4AE"
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#18B6B9",hatch="/",linewidth=0.01,edgecolor='0.3') #yellowgreen
    if attribute=='Local':
        rects = ax.bar(x + offset, measurement, width, label=attribute,color="#0D8B43",hatch="x",linewidth=0.01,edgecolor='0.3')
    if attribute=='Server':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#68BD48",hatch="\\",linewidth=0.01,edgecolor='0.3')  
    if attribute=='Random':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#9FD4AE",hatch="-",linewidth=0.01,edgecolor='0.3')       
    if attribute=='AutoNS':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#5560AC",hatch="//",linewidth=0.01,edgecolor='0.3')           
    if attribute=='SparseB':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#FAC074",linewidth=0.01,edgecolor='0.3')           
    if attribute=='D3QN':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color='#F28147',hatch="-",linewidth=0.01,edgecolor='0.3')           
    if attribute=='PPO':#""#567db0 #ec505d
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#428DBF",hatch="\\\\",linewidth=0.01,edgecolor='0.3')              
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
#ax.set_xlabel(gront)
ax.set_ylabel('Power(w)',fontsize = 12)
#ax.set_title('Average Power Per Frame')
ax.set_xticks(x +2*width, species,fontsize =10)
ax.legend(loc='upper left', ncols=2)
ax.set_ylim(0.0,11)
# ax.grid(True,'major','y')
# ax.set_rasterized(True)
plt.savefig("power_bandchange.pdf")
plt.show()





#1_3 manzulv  ratio
species = ("Yolo_s1","VGG_s1","ResNet_t","AlexNet_t",)
penguin_means = {
    'Bayes': (0.936 ,0.691 ,0.522 ,0.894 ),
    'SparseB':(0.507 ,0.373 ,0.237 ,0.829),
    'D3QN':(0.500 ,0.304 ,0.026 ,0.977 ),
    'PPO':(0.388 ,0.000 ,0.074 ,0.939 ),
    'AutoNS':(0.826 ,0.322 ,0.695 ,0.996 ),
    'GA': (0.721 ,0.437 ,0.119 ,0.740 ),
    'Random':(0.470 ,0.130 ,0.190 ,0.920 ),
    'Local': (0.101 ,0.000 ,0.000 ,1.000 ),
    'Server': (0.000 ,0.311 ,0.219 ,0.267 ),
    }
x = np.arange(len(species))  # the label locations
width = 0.1  # the width of the bars3
multiplier = 0

fig, ax = plt.subplots(layout='constrained',figsize=(5, 3))

colormap = plt.get_cmap('Blues')
colors = plt.cm.viridis(np.linspace(0, 1, 14))
for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    if attribute=='Bayes':#"#E3716E","#F9D580","#CCA7AF"
        rects = ax.bar(x + offset, measurement, width, label=attribute,color="#F29500",hatch="/",linewidth=0.007,edgecolor='0.3')
    if attribute=='GA': #"#9FD4AE"
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#18B6B9",hatch="/",linewidth=0.001,edgecolor='0.3') #yellowgreen
    if attribute=='Local':
        rects = ax.bar(x + offset, measurement, width, label=attribute,color="#0D8B43",hatch="x",linewidth=0.01,edgecolor='0.3')
    if attribute=='Server':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#68BD48",hatch="\\",linewidth=0.01,edgecolor='0.3')  
    if attribute=='Random':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#9FD4AE",hatch="-",linewidth=0.01,edgecolor='0.3')       
    if attribute=='AutoNS':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#5560AC",hatch="//",linewidth=0.01,edgecolor='0.3')           
    if attribute=='SparseB':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#FAC074",linewidth=0.01,edgecolor='0.3')           
    if attribute=='D3QN':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color='#F28147',hatch="-",linewidth=0.01,edgecolor='0.3')           
    if attribute=='PPO':#""#567db0 #ec505d
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#428DBF",hatch="\\\\",linewidth=0.01,edgecolor='0.3')              
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
#ax.set_xlabel(gront)
ax.set_ylabel('Ratio of latency within constraints',fontsize = 11)
#ax.set_title('The probability that the latency satisfies the condition')
ax.set_xticks(x +2*width, species,fontsize =10)
legend=ax.legend(loc='upper center', ncols=2,framealpha=0.6)
# for handle in legend.legendHandles:
    # handle.set_sizes([30])
    # handle.set_alpha(0.5)
ax.set_ylim(0.0,1.1)
# ax.grid(True,'major','y')
plt.savefig("ratio_bandchange.pdf")
plt.show()




##3_1_1 average power per frame

species = ("s1","tx2","hs110","s2"
)
penguin_means = {
    'Bayes': (4.756, 3.871 ,10.585 ,4.634),
    'SparseB':(4.425 ,3.813 ,10.345 ,4.629),
    'D3QN':(4.843 ,4.285 ,10.271 ,5.179),
    'PPO':(4.713,5.739 ,10.469 ,5.249),
    'AutoNS':(5.593 ,8.478 ,10.287 ,5.646),
    'GA': (4.942 ,4.103 ,10.271 ,4.599),
    'Random':(4.845 ,5.251 ,10.268 ,4.776 ),
    'Local': (6.326 ,10.566 ,10.815 ,6.048),
    'Server': (3.529 ,3.785 ,9.938 ,4.512),
    }
x = np.arange(len(species))  # the label locations
width = 0.1  # the width of the bars3
multiplier = 0

fig, ax = plt.subplots(layout='constrained',figsize=(5, 3))

colormap = plt.get_cmap('Blues')
colors = plt.cm.viridis(np.linspace(0, 1, 14))
for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    if attribute=='Bayes':#"#E3716E","#F9D580","#CCA7AF"
        rects = ax.bar(x + offset, measurement, width, label=attribute,color="#F29500",hatch="/",linewidth=0.01,edgecolor='0.3')
    if attribute=='GA': #"#9FD4AE"
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#18B6B9",hatch="/",linewidth=0.01,edgecolor='0.3') #yellowgreen
    if attribute=='Local':
        rects = ax.bar(x + offset, measurement, width, label=attribute,color="#0D8B43",hatch="x",linewidth=0.01,edgecolor='0.3')
    if attribute=='Server':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#68BD48",hatch="\\",linewidth=0.01,edgecolor='0.3')  
    if attribute=='Random':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#9FD4AE",hatch="-",linewidth=0.01,edgecolor='0.3')       
    if attribute=='AutoNS':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#5560AC",hatch="//",linewidth=0.01,edgecolor='0.3')           
    if attribute=='SparseB':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#FAC074",linewidth=0.01,edgecolor='0.3')           
    if attribute=='D3QN':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color='#F28147',hatch="-",linewidth=0.01,edgecolor='0.3')           
    if attribute=='PPO':#""#567db0 #ec505d
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#428DBF",hatch="\\\\",linewidth=0.01,edgecolor='0.3')              
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
#ax.set_xlabel(gront)
ax.set_ylabel('Power(w)',fontsize = 12)
#ax.set_title('Average Power Per Frame')
ax.set_xticks(x +2*width, species,fontsize =12)
ax.legend(loc='upper left', ncols=3,framealpha=0.6)
ax.set_ylim(0.0,15)
# fig.colorbar()
# ax.grid(True,'major','y')
plt.savefig("power_taskbandchange.pdf")
plt.show()




# ##3_1_1 average time per frame

species = ("Yolo_s1","ResNet_s1","AlexNet_s1","MobileNet_s1","VGG_t","AlexNet_h","ResNet_h","VGG_s2","AlexNet_s2")
penguin_means = {
    'Bayes': (1.908 ,2.420 ,0.653 ,1.320 ,2.267 ,0.448 ,1.840 ,2.581 ,0.342 ),
    'SparseB':(3.434 ,2.540 ,0.604 ,2.209 ,5.957 ,0.516 ,2.489 ,4.581 ,0.395 ),
    'D3QN':(4.410 ,4.054 ,0.530 ,1.765 ,11.935 ,0.395 ,4.249 ,8.145 ,0.136 ),
    'PPO':(3.009 ,3.548 ,0.544 ,1.644 ,4.873 ,0.320 ,2.896 ,7.806 ,0.142 ),
    'AutoNS':(0.704 ,1.506 ,0.189 ,1.287 ,1.267 ,0.233 ,1.178 ,1.645 ,0.163  ),
    'GA': (2.467 ,2.930 ,0.603 ,1.853 ,4.932 ,0.359 ,3.827 ,7.445 ,0.400 ),
    'Random':(6.518 ,3.249 ,0.579 ,2.113 ,6.543 ,0.478 ,3.881 ,8.121 ,0.306 ),
    'Local': (0.943 ,2.681 ,0.695 ,4.103 ,1.324 ,0.263 ,1.644 ,1.932 ,0.080 ),
    'Server': (8.886 ,2.941 ,3.170 ,2.919 ,2.145 ,2.247 ,2.190 ,2.570 ,2.445 ),
    }
x = np.arange(len(species))  # the label locations
width = 0.1  # the width of the bars3
multiplier = 0

fig, ax = plt.subplots(layout='constrained',figsize=(5, 3))

colormap = plt.get_cmap('Blues')
colors = plt.cm.viridis(np.linspace(0, 1, 14))
for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    if attribute=='Bayes':#"#E3716E","#F9D580","#CCA7AF"
        rects = ax.bar(x + offset, measurement, width, label=attribute,color="#F29500",hatch="/",linewidth=0.01,edgecolor='0.3')
    if attribute=='GA': #"#9FD4AE"
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#18B6B9",hatch="/",linewidth=0.01,edgecolor='0.3') #yellowgreen
    if attribute=='Local':
        rects = ax.bar(x + offset, measurement, width, label=attribute,color="#0D8B43",hatch="x",linewidth=0.01,edgecolor='0.3')
    if attribute=='Server':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#68BD48",hatch="\\",linewidth=0.01,edgecolor='0.3')  
    if attribute=='Random':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#9FD4AE",hatch="-",linewidth=0.01,edgecolor='0.3')       
    if attribute=='AutoNS':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#5560AC",hatch="//",linewidth=0.01,edgecolor='0.3')           
    if attribute=='SparseB':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#FAC074",linewidth=0.01,edgecolor='0.3')           
    if attribute=='D3QN':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color='#F28147',hatch="-",linewidth=0.01,edgecolor='0.3')           
    if attribute=='PPO':#""#567db0 #ec505d
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#428DBF",hatch="\\\\",linewidth=0.01,edgecolor='0.3')              
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
#ax.set_xlabel(gront)
ax.set_ylabel('Latency(s)',fontsize = 12)
#ax.set_title('Average Latency Per Frame')
ax.set_xticks(x +2*width, species,fontsize =10,rotation=45)
ax.legend(loc='upper left', ncols=3,framealpha=0.6)
ax.set_ylim(0.0, 13)
plt.savefig("latency_taskbandchange.pdf")
plt.show()




# ###3_1_1 yanshimanzulv

species = ("Yolo_s1","ResNet_s1","AlexNet_s1","MobileNet_s1","VGG_t","AlexNet_h","ResNet_h","VGG_s2","AlexNet_s2")
penguin_means = {
    'Bayes': (0.593 ,0.509 ,0.941 ,0.728 ,0.753 ,0.976 ,0.002 ,0.176 ,0.979  ),
    'SparseB':(0.479 ,0.528 ,0.950 ,0.539 ,0.349 ,0.952 ,0.058 ,0.184 ,0.965  ),
    'D3QN':(0.134 ,0.137 ,0.967 ,0.600 ,0.318 ,0.975 ,0.072 ,0.192 ,0.987 ),
    'PPO':(0.424 ,0.210 ,0.933 ,0.606 ,0.487 ,0.983 ,0.000 ,0.000 ,0.998 ),
    'AutoNS':(0.953 ,0.664 ,1.000 ,0.730 ,0.833 ,0.997 ,0.750 ,0.454 ,1.000  ),
    'GA': (0.463 ,0.303 ,0.937 ,0.532 ,0.367 ,0.988 ,0.126 ,0.090 ,0.976 ),
    'Random':(0.411 ,0.339 ,0.932 ,0.487 ,0.411 ,0.953 ,0.045 ,0.080 ,0.973   ),
    'Local': (0.695 ,0.516 ,1.000 ,0.000 ,0.999 ,1.000 ,0.000 ,0.000 ,1.000  ),
    'Server': (0.000 ,0.192 ,0.114 ,0.347 ,0.485 ,0.378 ,0.431 ,0.229 ,0.301  ),  
    }
x = np.arange(len(species))  # the label locations
width = 0.1  # the width of the bars3
multiplier = 0

fig, ax = plt.subplots(layout='constrained',figsize=(5, 3))

colormap = plt.get_cmap('Blues')
colors = plt.cm.viridis(np.linspace(0, 1, 14))
for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    if attribute=='Bayes':#"#E3716E","#F9D580","#CCA7AF"
        rects = ax.bar(x + offset, measurement, width, label=attribute,color="#F29500",hatch="/",linewidth=0.01,edgecolor='0.3')
    if attribute=='GA': #"#9FD4AE"
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#18B6B9",hatch="/",linewidth=0.01,edgecolor='0.3') #yellowgreen
    if attribute=='Local':
        rects = ax.bar(x + offset, measurement, width, label=attribute,color="#0D8B43",hatch="x",linewidth=0.01,edgecolor='0.3')
    if attribute=='Server':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#68BD48",hatch="\\",linewidth=0.01,edgecolor='0.3')  
    if attribute=='Random':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#9FD4AE",hatch="-",linewidth=0.01,edgecolor='0.3')       
    if attribute=='AutoNS':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#5560AC",hatch="//",linewidth=0.01,edgecolor='0.3')           
    if attribute=='SparseB':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#FAC074",linewidth=0.01,edgecolor='0.3')           
    if attribute=='D3QN':
         rects = ax.bar(x + offset, measurement, width, label=attribute,color='#F28147',hatch="-",linewidth=0.01,edgecolor='0.3')           
    if attribute=='PPO':#""#567db0 #ec505d
         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#428DBF",hatch="\\\\",linewidth=0.01,edgecolor='0.3')              
    multiplier += 1

# Add some text for labels, title and custom x-axiss tick labels, etc.
#ax.set_xlabel("Tasks",fontsize = 10)
ax.set_ylabel('Ratio of latency within constraints',fontsize = 9)
#ax.set_title('The probability of the latency satisfies the condition')
ax.set_xticks(x +2*width, species,fontsize =10,rotation=45)
ax.legend(loc='upper center', ncols=3,framealpha=0.5)
ax.set_ylim(0.0, 1.35)
plt.savefig("ratio_taskbandchange.pdf")
plt.show()



# #bayes 3_1_1 task 甘特图    ####https://blog.csdn.net/sgzqc/article/details/121893158
# colors = ['#3682be','#45a776','#f05326','#eed777','#334f65','#b3974e','#38cb7d','#ddae33','#844bb3','#93c555','#5f6694','#df3881']
# cmap=plt.get_cmap("RdBu")
# # colorss=
# def color(row):
#     c_dict = {'AlexNet':'#b3974e', 'MobileNet':'#5f6694', 'Yolo':'#34D05C', 'ResNet':'#34D0C3', 'VGG':'#3475D0','VGG_2':'#3682be',
#               'Yolo_3':'#45a776','ResNet_2':'#45a776','AlexNet_3':'#eed777','AlexNet_2':'#334f65'
# }
    
#     #'#f05326','#E64646'
#     return c_dict[row['Task']]
# df = pd.read_excel('E:\\深度deepproject\\data24.8.15-\\bayes3_1_1gant.xlsx')
# # Task	Start	length
# # AlexNet	0	3
# # MobileNet	0	7
# # Yolo	1	2
# # ResNet	1	2
# # VGG	3	2
# # Yolo_2	5	2
# # Yolo_3	3	2
# # ResNet_2	5	2
# # AlexNet_2	3	2
# # MobileNet_2	5	2
# df['color'] = df.apply(color, axis=1)
# # fig, ax = plt.subplots(1, figsize=(5,1))
# fig, ax = plt.subplots(1,)
# ax.barh(df.Task,df.length, left=df.Start,color=df.color)

# ax.grid(which='major',axis='x')
# ax.set_xlabel('time/min',size=14)
# # ax.set_yticks([0,1,2,3,4,5,6,7,8,9],lablels=['MobileNet_2','AlexNet_2','ResNet_2','Yolo_3','Yolo_2','VGG','ResNet','Yolo','MobileNet','AlexNet'])
# # ax.set_ylabel(size=14)
# plt.show()




# bayes 3_1_1 bandwidth 折线图
# bandwidthlist=[55,55,8,8,15,15,5,5,8,8,15,15,55,55,15,15,5,5,8,8,8,8,5,5,55,55,55]
# x=[i for i in range(len(bandwidthlist))]
# plt.step(x,bandwidthlist,where='post')
# plt.xlabel('time/min',size=14)
# plt.ylabel('Mbps',size=14)
# plt.show()




###*********ax.grid()**********######
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





#一种配色
# offset = width * multiplier
#     if attribute=='Bayes':#"#E3716E","#F9D580","#CCA7AF"#F29500
#         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#F29500",hatch="/",linewidth=0.007,edgecolor='0.3')
#     if attribute=='GA': #"#9FD4AE"
#          rects = ax.bar(x + offset, measurement, width, label=attribute,color="#8CD0C3",hatch="/",linewidth=0.001,edgecolor='0.3') #yellowgreen
#     if attribute=='Local':
#         rects = ax.bar(x + offset, measurement, width, label=attribute,color="#0D8B43",hatch="x",linewidth=0.01,edgecolor='0.3')
#     if attribute=='Server':
#          rects = ax.bar(x + offset, measurement, width, label=attribute,color="#68BD48",hatch="\\",linewidth=0.01,edgecolor='0.3')  
#     if attribute=='Random':
#          rects = ax.bar(x + offset, measurement, width, label=attribute,color="#9FD4AE",hatch="-",linewidth=0.01,edgecolor='0.3')       
#     if attribute=='AutoNS':
#          rects = ax.bar(x + offset, measurement, width, label=attribute,color="#5560AC",hatch="//",linewidth=0.01,edgecolor='0.3')           
#     if attribute=='SparseB':
#          rects = ax.bar(x + offset, measurement, width, label=attribute,color="#FAC074",linewidth=0.01,edgecolor='0.3')           
#     if attribute=='D3QN':##F28147
#          rects = ax.bar(x + offset, measurement, width, label=attribute,color='#83a4b6',hatch="\\\\",linewidth=0.01,edgecolor='0.3')           
#     if attribute=='PPO':#"#428DBF"#567db0#253A69
#          rects = ax.bar(x + offset, measurement, width, label=attribute,color="#ec505d",hatch="-",linewidth=0.01,edgecolor='0.3')              
#     multiplier += 1











