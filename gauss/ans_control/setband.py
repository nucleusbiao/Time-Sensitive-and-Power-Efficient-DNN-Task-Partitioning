import os

#设置带宽
# setargs="trickled -d 50 -u 10"
# os.system(setargs)

# input()

# #取消设置
# os.system('ps -ef | grep trickled | grep -v grep > trickled.txt')
# print("o\n")
# pid=''
# with open('trickled.txt','r') as tf:
#     line=tf.readline().split()
#     print(line)
#     pid=line[1]
# killarg='echo 6 | sudo -S kill '+pid
# os.system(killarg)

# def deltrickled():
#     global passward
#     pid=''
#     killarg=''
#     os.system('ps -ef | grep trickled | grep -v grep > trickled.txt')
#     with open('trickled.txt','r') as tf:
#         line=tf.readline().split()
#         # print(line)
#         if(len(line)>1):
#             pid=line[1]
#             killarg='echo '+passward+' | sudo -S kill '+pid
#     os.system(killarg)



# timecounter=0
# passward='6'
# while True:
#     bandwidthlist=[5,8,10,10,7,15,7,50,50]
#     bandwidth=bandwidthlist[timecounter]
#     deltrickled()
#     input('killed. Press any key to continue')

#     strwidths="trickled"+ " -d "+ str(bandwidth)+" -u "+ str(bandwidth)
#     res_ch=os.system(strwidths)
#     print("res_ch",res_ch, timecounter,' min  bandwidth is set to ',bandwidth)
#     input('press any key to continue')

#     timecounter+=1
#     if timecounter>=len(bandwidthlist):
#         deltrickled()
#         print("killed")
#         break


import random
# bandlist=[5,7,3,15,50]
# bandrandlist=[]
# for i in range(14) :
#     i=random.randint(0,4)
#     bandrandlist.append(bandlist[i])
#     # print(i)
# print(bandrandlist)
timelen=32
bandlist=[30,25,20,15,15,10,10,7,7,5,5]
list=[0 for i in range(timelen)]
len=len(bandlist)
for i in range(timelen):
    # if i%2!=0:
    #     list[i]=list[i-1]
    #     continue
    list[i]=bandlist[random.randint(0,len-1)]
    #if flag:
    #    list[i]=1
print(list)

# for i in range(10):
#     print(random.randint(0,5))
