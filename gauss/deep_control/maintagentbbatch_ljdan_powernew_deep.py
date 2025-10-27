#深度神经网络预测
#-----------------------------------#
import os
import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
print(parent_dir)
sys.path.append(parent_dir)
parent_parent_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(parent_parent_dir)

import random
import torch
import argparse
import time
import threading
import threading
import collections

import json
from models.vgg16 import vgg16
from models.tiny_yolo import tinyYolo
from models.mobilenetv2 import mobilenetv2
from models.resnet import resnet50
from models.alexnet import alexnet
from models.mobileformer import MobileFormer
from utils.config import config_294
from communicationdeep import serverCommunication
from prodnntwonetnewnorbbatch_ljdan_lessupdate_new import Cnet

def parse_args():
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnns', dest='dnns',
                        help='vgg, yolo,resnet,alexnet,mobilenetv2',
                        default='alexnet', type=str)
    parser.add_argument('--epoch', dest='epoch',
                        help='epoch',
                        default=30000, type=int) 
    parser.add_argument('--hosts', dest='hosts',
                        help='hosts communicate with client',
                        default="0.0.0.0", type=str)
    parser.add_argument('--ports', dest='ports',
                        help='port communicate with client for infer',
                        default="9096", type=str)  #"8081,8082,8083"  8084,8085,8086
    parser.add_argument('--totalnumber', dest='totalnumber',
                        help='totalnumber of all tasks all devices',
                        default=0, type=int)    
    parser.add_argument('--username', dest='username',
                        help='username',
                        default='k', type=str)  
    
    args = parser.parse_args()
    return args

def edge(dnn,netnumer):
    global partition_pointlist
    global partition_pointlistofthelatency
    global datalist
    global infersteplist
    global msg
    global inferend_step
    global actuallatencylist
    global actuallatencystep
    global taskifon 
    global totalnumber
    # global mainnumber
    step=-1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device="cpu"
    print(f'totalnumber {totalnumber}, using {device} device.')
    time1=time.time()

    if dnn == 'vgg':
        model = vgg16()
        model.eval()
        acttotal=22
    elif dnn == 'yolo':
        model = tinyYolo()
        model.eval()
        acttotal=31
    elif dnn == 'alexnet':
        model=alexnet()
        model.eval()
        acttotal=11
    elif dnn == 'mobilenetv2':
        model=mobilenetv2()
        model.eval()
        acttotal=21
    elif dnn == 'resnet':
        model = resnet50(num_classes=5)
        weights_path = "models/resNet50_e3_1_0.pth"
        if os.path.exists(weights_path) == False:
            weights_path = "../models/resNet50_e3_1_0.pth"
        # load model weights
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        # read class_indict
        json_path = 'models/class_indices.json'
        if os.path.exists(json_path)== False:
            json_path = '../models/class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
        with open(json_path, "r") as f:
            class_indict = json.load(f)
        acttotal=20
    elif dnn=='mobileformer':
        model=MobileFormer(config_294)
        weights_path='models/mobileformer.pth.tar'
        if os.path.exists(weights_path) == False:
            weights_path = "../models/mobileformer.pth.tar"
        # load model weights
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device)['state_dict'])
        model.eval()
        acttotal=18
    else:
        print('please input right net name')
        assert False
    #w
    #model.cuda()
    model.to(device)
    print(f'{totalnumber} 模型加载时间：{time.time()-time1}',)
    filepath=str(parent_dir)+"/latency/latency_server"+"_"+str(dnn)+str(totalnumber)+".txt"
    myfile=open(filepath,'w')
    myfile.write(f'starttime: {str(time.time())}')
    myfile.write("receive_time              infer_endtime     partition_point      sendendtime")

    lock=threading.Lock()

    while taskifon[netnumer]:
        try:
            if step < infersteplist[netnumer]:  #说明已经收到了数据
                lock.acquire()
                step=infersteplist[netnumer]
                partition_point = partition_pointlist[netnumer]
                data=datalist[netnumer]
                lock.release()
                receive_time=time.time()

                if partition_point!=acttotal:
                    #data = torch.autograd.Variable(data)
                    if dnn=='mobileformer':
                        if type(data[1])!=int:
                            zd=data[1].to(device)
                        prediction = model(x=data[0].to(device),z=zd,b=data[2],server=True, partition=partition_point)
                    else:
                        prediction = model(data.to(device), server=True, partition=partition_point)
                    res = prediction.data
                    res=res.to('cpu' )
                    # torch.cuda.synchronize()
                    infer_endtime=time.time()
                    res=[res,infer_endtime-receive_time]
                    #lock ~~~~~~
                    lock.acquire()
                    msg[netnumer]=res
                    inferend_step[netnumer]=step
                    lock.release()
                else:
                    lock.acquire()
                    # msg[netnumer]=
                    inferend_step[netnumer]=step
                    lock.release()

                    # 或者可以用唤醒
                send_endtime=time.time()
                myfile.write("\n")
                if(partition_point!=acttotal):
                    myfile.write(f'{receive_time:<22}{infer_endtime:<22}{partition_point:<8}{send_endtime}')
                else:
                    myfile.write(f'{receive_time:<22}{receive_time:<22}{partition_point:<8}{send_endtime}')
            time.sleep(0.001)

        except KeyboardInterrupt or TypeError or OSError:
            myfile.close()
            del model
            torch.cuda.empty_cache()
            time.sleep(2)
            print(dnn," edge closed now.")
    myfile.close()
    del model
    torch.cuda.empty_cache()
    time.sleep(2)
    print(dnn," edge closed now.")

def communicationthwithclient(host,port,netnumber,actotal):
    global ddpartitionlist
    global partition_pointlist
    global partition_pointlistofthelatency
    global datalist
    global infersteplist
    global msg
    global inferend_step
    global actuallatencylist
    global actuallatencystep
    global actuallatencyclient
    global actuallatencytrans
    global actuallatencyedge
    global taskifon

    print(f'ready to connect{port}')
    communication=serverCommunication(host,port)
    conn, addr = communication.accept_conn()
    lock=threading.Lock()

    while True:
        try:
            #发送分割点数据
            lock.acquire()
            ddpartition=ddpartitionlist[netnumber]
            lock.release()
            communication.send_msg(conn, ddpartition)

            #接收，分割点，中间数据
            recevieim = communication.receive_msg(conn)
            #判断是否结束
            if(recevieim=="close"):
                # communication.send_msg("ok")
                break
            partition_point=recevieim[0]
            imdata=recevieim[1]
            lock.acquire()
            partition_pointlist[netnumber]=partition_point
            datalist[netnumber]=imdata
            infersteplist[netnumber]+=1  #表示接收到了中间数据  中间数据要都保存下来啊！！！！然后，用所得数据多个batch一起训练～～～～～
            lock.release()

            #发送推理结果
            lock.acquire()
            inferend_s=inferend_step[netnumber]
            inferstepl=infersteplist[netnumber]
            lock.release()
            while inferend_s!= inferstepl:
                time.sleep(0.001)
                lock.acquire()
                inferend_s=inferend_step[netnumber]
                inferstepl=infersteplist[netnumber]
                lock.release()
            if partition_point != actotal:
                communication.send_msg(conn, msg[netnumber])
            #接受总延时数据
            latencydata=communication.receive_msg(conn)  #[actuallatency,partitonpoint,actual_client,actual_trans,actual_edge]
            #判断是否结束
            if(latencydata=="close"):
                # communication.send_msg("ok")
                break
            lock.acquire()
            partition_pointlistofthelatency[netnumber]=latencydata[1] #partition_pointlistofthelatency更新须与actuallatencylist同步
            actuallatencystep[netnumber]=inferend_s
            actuallatencylist[netnumber]=latencydata[0]
            actuallatencyclient[netnumber]=latencydata[2]
            actuallatencytrans[netnumber]=latencydata[3]
            actuallatencyedge[netnumber]=latencydata[4]
            lock.release()
        except  KeyboardInterrupt or TypeError or OSError:
            taskifon[netnumber]=False
            # print("taskifon is false ",dnnslist[netnumber] )
    taskifon[netnumber]=False
    communication.close_channel()
    # print("taskifon is false ",dnnslist[netnumber] )
    

def deepthread(dnnnumber,username,dnn):
    global ddpartitionlist   
    global partition_pointlist
    global partition_pointlistofthelatency
    global datalist
    global infersteplist
    global msg
    global inferend_step
    global actuallatencylist
    global actuallatencystep
    global args
    global dnnsnum
    global dnnslist
    # global mainnumber
    global portlist
    global actuallatencyclient
    global actuallatencytrans
    global actuallatencyedge
    global taskifon
    global totalnumber

    #超参
    lr = 1e-2
    #lr = 2e-3  # 学习率
    gamma = 0.9  # 折扣因子
    historylen=3 #历史数据个数
    alpha=0.2   #static 的权重
    #lossrate=0.005  #模型训练完成的判断 之前调好的。
    lossrate=0.2  #没有预训练模型的话，还是要尽快收敛，给调大了。


    print("alpha ",alpha)
    print("lossrate ",lossrate)

    trainmode=False
    iftrainready=False

    actuallatencystepold=-1
    dnnname=dnnslist[dnnnumber]
    
   
    #vgg, yolo,resnet,alexnet,mobilenetv2 的总延时范围 用于数据归一化处理
    latencyrangedic_high={"vgg":2.3733,  "yolo":1.907, "resnet":2.144,"alexnet": 0.1988, "mobilenetv2":0.399,"mobileformer":1.7}
    latencyrangedic_low={"vgg":0.3171,  "yolo":0.322, "resnet":0.107, "alexnet":0.036, "mobilenetv2":0.178,"mobileformer":0.2}
    #vgg, yolo,resnet,alexnet,mobilenetv2 的分割点数量
    partitioncountdict={"vgg":22,"yolo": 31,"resnet":20,"alexnet":11,"mobilenetv2":21,'mobileformer':18}
    partitionmax=partitioncountdict[dnnname]   #!!!!'mobileformer':0是随便写的！！！！！
    
    #变量
    totaloss=0
    lossnum=0
    averloss=0
    loss=0
    lossc=collections.deque(maxlen=10)   #将20改成10
    losslist=[]
    #step=args.step
    step=0

    totallossall=0
    totallossallold=0
    lossdowncount=0
    totalallcount=0

    #实例化Cnet
    #Gausspro=Gaussmodel(netname=dnnname,username=username)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device ",device)
    CNet=Cnet(learning_rate=lr,gamma=gamma,historylength=historylen,arg=args,device=device,dnnname=dnnname,dnnsnum=1,partitioncountdic=partitioncountdict,latencyrangedic_low=latencyrangedic_low,latencyrangedic_high=latencyrangedic_high,mainnumber=dnnnumber)
    print(CNet)

    pprf=str(parent_dir)+"/latency/deeplatency"+str(dnn)+str(totalnumber)+".txt"
    with open(pprf,'w') as prf:
        prf.write(f'real_latency    predict_latency      ')  
        prf.write("averloss    step ")
        # for pi in range(dnnsnum):
        prf.write(f'    partitionpoint      ')
        prf.write("       totaleverytime        predicting_pp_time    updatetime")
        # for pi in range(dnnsnum):
        prf.write(f'    ifupdate       ')
        prf.write(" totalnum         loss_mbatch   ifupdatenet   predicttime    time")

    lock=threading.Lock()

    #数据收集阶段，随机选择分割点进行测试
    start_time=time.time() # # for ba # # for batch in range(batchsize):
    stepoldlist=-1
    ifreadyfortrain=-1
    lock=threading.Lock()
    actuallatencystepold=-1
    ifupdate=-1
    real_latency=-1

    partitionmax=partitioncountdict[dnnname]  #各个网络的分割点上限
    partitp=0 #
    # totalh=0  #cnet historypool collections总个数
    #for i in range(dnnsnum):
    totalh=partitionmax+1
    ifreadyfortrain=0   #判断是否收集够历史数据
    historyready=0

    averloss=0
    loss_mbatch=0
    pppaver=[]
    pppavercount=[]
    # for dnn in range(dnnsnum):
    for pp in range(partitionmax+1):
        pppaver.append(0)
        pppavercount.append(0)
    pppaverflag=False
    #print("非train 不做训练")
    lossready=False   #loss 是否下降判断

    # testf=0  #for test
    partitionpoint_use=0   #实际使用的分割点

    # losslist=[]
    step=0
    starttimeall=time.time()
    for epoch in range(5000):

        #判断是否client要结束
        if(taskifon[dnnnumber]==False):
            break
        partition_lmin=0

        #选取分割点
        #是否收集够历史数据
        if ifreadyfortrain!=totalh:
            ifreadyfortrain=0
            # for dnnc in range(dnnsnum):
            for hd in CNet.historypool:
                if len(hd)==historylen+1:
                    ifreadyfortrain+=1
        if historyready!=1:
            #historyready=0
            # for i in range(dnnsnum):
            if len(CNet.historylatency_datalist)==historylen:
                historyready=1
        p_starttime=time.time()

        #预测分割点
        # if args.train==False or epoch>3000:  
        prdicttime=0
        totalnum=0
        if trainmode==False: #决策模式
            if(ifreadyfortrain!=totalh):     #采集够数据
                # for i in range(dnnsnum):
                for p in range(partitionmax+1):
                    if len(CNet.historypool[p])<historylen+1:
                        partitp=p
                        break
                lock.acquire()
                # for i in range(dnnsnum):
                ddpartitionlist[dnnnumber]=partitp
                lock.release()

            ###loss不再下降后改为自主决策。
            #if lossdowncount<5:  #之前调好的，为5，现为了速度，尝试改为3
            if lossdowncount<5:
                # lossready=judgeloss()
                if totallossallold!=0:
                    if (totallossallold-totallossall)/totallossallold< lossrate:
                        lossdowncount+=1
                        # print(lossdowncount)
                        # print("totallossall ",totallossall)
                    else:
                        lossdowncount=0
                else:
                    lossdowncount=0
                if lossdowncount==5:
                    # print("转为决策模式 step{} dnnnumber{}",step,dnnnumber)
                    print(f'转为决策模式 step{str(step)} dnnnumber{dnnnumber}"')
            
            if iftrainready==True and lossdowncount<5:
                print("直接为决策模式")
                lossdowncount=5
            
            if(ifreadyfortrain==totalh and historyready==1 and lossdowncount>=5):  #采集够数据后,预测最小延时分割点
                CNet.net.eval()
                with torch.no_grad():
                    predictstarttime=time.time()
                    pplist=0      #初始化
                    pplistold=-1      #初始化
                    min_pp=0
                    minlatencylist=CNet.predict_latency(pplist)  #
                    minlatency=0
                    minlatency=minlatencylist[0][0].item() 
                    for p_this in range(1,partitionmax+1):
                        latencylistof_pp=CNet.predict_latency(p_this)[0][0].item() 
                        if minlatency>latencylistof_pp:
                            min_pp=p_this
                            minlatency=latencylistof_pp
                    predictendtime=time.time()
                    prdicttime=predictendtime-predictstarttime
                    lock.acquire()
                    ddpartitionlist[dnnnumber]=min_pp
                    lock.release()
            else:   ##随机生成分割点
                lock.acquire()
                # for i in range(dnnsnum):
                ddpartitionlist[dnnnumber]=random.randint(0,partitionmax)
                lock.release()

        else:  # #训练模式
            if(ifreadyfortrain!=totalh):     #采集够数据后，随机选点
                # for i in range(dnnsnum):
                for p in range(partitionmax+1):
                    if len(CNet.historypool[p])<historylen+1:
                        partitp=p
                        break
                lock.acquire()
                # for i in range(dnnsnum):
                ddpartitionlist[dnnnumber]=partitp
                lock.release()
            else:
                ##随机生成分割点
                lock.acquire()
                # for i in range(dnnsnum):
                ddpartitionlist[dnnnumber]=random.randint(0,partitionmax)
                lock.release()
        pre_endtime=time.time()

        ##等待网络完成一帧推理
        update=0
        while update==0 :
            update=0
            # for i in range(dnnsnum):    
            if actuallatencystepold<actuallatencystep[dnnnumber]:
                update+=1  
            time.sleep(0.001)

        #获取时间延时和分割点
        lock.acquire()
        real_latency=actuallatencylist[dnnnumber]
        partitionpoint_use=partition_pointlistofthelatency[dnnnumber] #如果发送信息快了一步，就不对应了！所以，partition_pointlistofthelatency 更新须与actuallatencylist同步
        real_latencyclient=actuallatencyclient[dnnnumber]
        real_latencytrans=actuallatencytrans[dnnnumber]
        real_latencyedge=actuallatencyedge[dnnnumber]
        if partitionpoint_use>partitionmax:
                print("wrong",partitionpoint_use) 
        ifupdate=True
        actuallatencystepold=actuallatencystep[dnnnumber]
        lock.release()

        #训练预测模型
        if(ifreadyfortrain==totalh and historyready==1):
            update_starttime=time.time()
            # if args.train==False:   #这里，非train 不做训练
            #     outputs=CNet.predict_cy(partitionpoint_u             # else:
            ifmbatch= (lossdowncount<5)
            loss,outputs,loss_mbatch,ifupdatenet=CNet.update(real_latency=real_latency,partitionpint=partitionpoint_use,ifmbatch=ifmbatch,averloss=averloss)
            lossc.append(loss)
            totaloss=0
            for loss in lossc:
                totaloss+=loss 
            lossnum=len(lossc)
            averloss=totaloss/lossnum
            losslist.append(averloss)

            totallossallold=totallossall
            totalallcount+=1
            totallossall=totallossallold+(loss-totallossallold)/(totalallcount)

            #每次预测+运行的总时间
            start_time_new=time.time()
            totaleverytime=start_time_new-start_time
            start_time=start_time_new
            #记录
            with open(pprf,'a') as prf:
                prf.write("\n")
                # for i in range(dnnsnum):
                prf.write(f'{real_latency:<23}{str(outputs[0][0].item()):<23}')
                prf.write(f'{str(averloss):<23}{str(step):<7}')
                # for i in range(dnnsnum):
                prf.write(f'{str(partitionpoint_use):<23}')
                prf.write(f'{str(totaleverytime):<23}{str(pre_endtime-p_starttime):<23}{str(start_time_new-update_starttime):<23}')
                # for i in range(dnnsnum): 
                prf.write(f'{ifupdate:<13}')
                prf.write(f'{totalnum} ')
                prf.write(f'{str(loss_mbatch):<23}   {int(ifupdatenet)}   ')
                prf.write(f'{prdicttime}     {str(time.time()-starttimeall)}')
        else:
             #记录
            with open(pprf,'a') as prf:
                prf.write("\n")
                # for i in range(dnnsnum):
                prf.write(f'{real_latency:<23}{str(-1):<23}')
                prf.write(f'{str(-1):<23}{str(step):<7}')
                # for i in range(dnnsnum):
                prf.write(f'{str(partitionpoint_use):<23}')
                prf.write(f'{str(-1):<23}{str(pre_endtime-p_starttime):<23}{str(-1):<23}')
                # for i in range(dnnsnum): 
                prf.write(f'{ifupdate:<13}')
                prf.write(f'{totalnum} ')
                prf.write(f'{str(-1):<23}   {int(-1)}   ')
                prf.write(f'{prdicttime} ')
                prf.write(f'{prdicttime}   {str(time.time()-starttimeall)}')


        #更新历史延时
        if ifupdate:
            if len(CNet.historylatency_datalist)==historylen:
                if ifreadyfortrain!=totalh :
                    for i in range(historylen):
                        CNet.historypool[partitionpoint_use].append(CNet.historylatency_datalist[i])
                    CNet.historypool[partitionpoint_use].append(real_latency)
                else:
                    for i in range(historylen):
                        if partitionpoint_use>partitionmax:
                            print("wrong",partitionpoint_use)
                        CNet.historypool[partitionpoint_use][i]=CNet.historylatency_datalist[i]
                    CNet.historypool[partitionpoint_use][-1]=real_latency
            CNet.historylatency_datalist.append(real_latency)    

        ####***增加更新分割点输入数据   所有数据加权平均
        if(ifreadyfortrain!=totalh or historyready!=1):
                if ifupdate:
                    pppaver[partitionpoint_use]+=real_latency
                    pppavercount[partitionpoint_use]+=1
        elif(ifreadyfortrain==totalh and historyready==1 and pppaverflag==False):
                pppaverflag=True
                print("pppaverflag is true now ",dnnnumber)
                print("ifreadyfortrain",ifreadyfortrain)
                for pp in range(partitionmax+1):    
                    if pppavercount[pp]!=0:  ##
                        CNet.staticlatencyofpplist[pp]=pppaver[pp]/pppavercount[pp] ##
                print(CNet.staticlatencyofpplist)
        else:
            if ifupdate:
                CNet.staticlatencyofpplist[partitionpoint_use]=CNet.staticlatencyofpplist[partitionpoint_use]+alpha*(real_latency-CNet.staticlatencyofpplist[partitionpoint_use])
        
        step+=1

    #print(username," ",dnnname," ",totalnumber," average loss : ",losssum/lossnum)
    print(username," ",dnnname," ",totalnumber," average loss : ")

#全局变量
args=parse_args()
# print('Called with args:')
print(args)
#dnn 任务列表
dnnslist=args.dnns.split(",")
dnnsnum=len(dnnslist)
totalnumber=args.totalnumber

portlist=args.ports.split(",")
for i in range(len(portlist)):
    portlist[i]=int(portlist[i])
# print("portlist ",portlist)
HOST = '0.0.0.0'
#print("dnnslist ",dnnslist)

#延时信息
ddpartitionlist=[0 for _ in range(dnnsnum)]    #预测给出的分割点
partition_pointlist=[0 for _ in range(dnnsnum)]  #client使用的分割点
partition_pointlistofthelatency=[0 for _ in range(dnnsnum)] #与actuallatencylist同步
datalist=[0.0 for _ in range(dnnsnum)]          #推理中间数据
infersteplist=[-1 for _ in range(dnnsnum)]      #接受中间数据次数
msg=[0 for _ in range(dnnsnum)]                 # 推理结果
inferend_step=[-1 for _ in range(dnnsnum)]       #edge推理完成次数
actuallatencylist=[0.0 for _ in range(dnnsnum)] #总延时
actuallatencystep=[-1 for _ in range(dnnsnum)]  #更新actuallatency次数
actuallatencyclient=[0.0 for _ in range(dnnsnum)]
actuallatencytrans=[0.0 for _ in range(dnnsnum)]
actuallatencyedge=[0.0 for _ in range(dnnsnum)]

#任务是否关闭标志
taskifon=[True for _ in range(dnnsnum)]

# mainnumber=""    ###区分文件路径
if __name__=="__main__":
    partitioncountdict={"vgg":22,"yolo": 31,"resnet":20,"alexnet":11,"mobilenetv2":21,'mobileformer':18}

    #启动延时预测模型线程
    # usernamelist=["k"]
    for i in range(dnnsnum):
    #    dnnnumber=i
       t=threading.Thread(target=deepthread,args=(i,args.username,dnnslist[i]))
       t.setDaemon(True)
       t.start()

    #创建edge推理线程
    for i in range(dnnsnum): 
        t=threading.Thread(target=edge,args=(dnnslist[i],i))
        t.setDaemon(True)
        t.start()
        #创建与client通信子线程,与DNN client 通讯 子线程， 负责接收延时，发送分割点
    
    #启动与client通信线程
    for i in range(dnnsnum): 
        th=threading.Thread(target=communicationthwithclient,args=(HOST,portlist[i],i,partitioncountdict[dnnslist[i]]))    
        th.setDaemon(True)
        th.start()

    while True:
        time.sleep(2)
        if (taskifon[0])==False:
            time.sleep(2)
            # print(dnnslist[0], "main break.")
            break


    
    
    
    
    




