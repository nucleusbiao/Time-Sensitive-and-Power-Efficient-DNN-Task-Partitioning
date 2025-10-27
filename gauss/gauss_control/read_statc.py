#读取静态延时数据示例。

import pickle
import numpy as np

username='k'
netnamelist=['alexnet','mobilenetv2','yolo','resnet','vgg']
#for i in range(len(netnamelist)):
for i in range(len(netnamelist)):
    netname=netnamelist[i]
    netname=='yolo'
    #静态测量的数据  client,trans,edge
    avergedatadic_net=[]
    pdata=""
    user=username   #!!!!!!!!!1
    if(pdata)!='':
        #file='models/pkl/'+pdata+'/'+netname +user+"s2"+'.pkl'
        file='models/pkl/'+pdata+'/'+netname +user+'.pkl'
    else:
        file='models/pkl/'+ netname +user+'.pkl'
        #file='models/pkl/'+ netname +user+"s2"+'.pkl'
    print('file: ',file)

    with open (file, 'rb') as temp:
        pret= pickle.load(temp)
        avergedatadic_net.append(pret[0])  #clientaver
        avergedatadic_net.append(pret[2])  #transaver
        avergedatadic_net.append(pret[4])  #edgeaver
        if file=="models/pkl/yolok.pkl":
            print(avergedatadic_net)
    avergedatadic_net=np.array(avergedatadic_net)
    sum=np.sum(avergedatadic_net,axis=0)
    minindex=np.argmin(sum)

    print(netname,len(sum),'minindex:',minindex)
    # print(sum)
    # print(sum[minindex])