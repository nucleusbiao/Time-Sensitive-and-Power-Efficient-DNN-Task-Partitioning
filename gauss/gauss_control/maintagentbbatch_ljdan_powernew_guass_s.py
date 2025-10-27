
# guass process 预测延时。
#为单个任务，动态启停功能。
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
from gaussmodel2 import Gaussmodel
import math

import json
from models.vgg16 import vgg16
from models.tiny_yolo import tinyYolo
from models.mobilenetv2 import mobilenetv2
from models.resnet import resnet50
from models.alexnet import alexnet
from models.mobileformer import MobileFormer
from utils.config import config_294
from communicationdeep import serverCommunication


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
                    if dnn=='mobileformer':
                        if type(data[1])!=int:
                            zd=data[1].to(device)
                        prediction = model(x=data[0].to(device),z=zd,b=data[2],server=True, partition=partition_point)
                    else:
                        prediction = model(data.to(device), server=True, partition=partition_point)
                    res = prediction.data
                    res=res.to('cpu')
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
    taskifon[netnumber]=False
    communication.close_channel()
    
def guassthread(dnnnumber,username,dnn):
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

    partitioncountdict={"vgg":22,"yolo": 31,"resnet":20,"alexnet":11,"mobilenetv2":21,'mobileformer':18}
    actuallatencystepold=-1
    dnnname=dnnslist[dnnnumber]
    partitionmax=partitioncountdict[dnnname]   #!!!!'mobileformer':0是随便写的！！！！！
    optimaldic={'k':{"alexnet":2,"mobilenetv2":0,"yolo":16,"resnet":0,"vgg":0,'mobileformer':0},
            'u':{"alexnet":8,"mobilenetv2":5,"yolo":20,"resnet":0,"vgg":0,'mobileformer':4},
            'tx2':{"alexnet":2,"mobilenetv2":0,"yolo":16,"resnet":0,"vgg":0,'mobileformer':18},
            'ly':{"alexnet":2,"mobilenetv2":0,"yolo":0,"resnet":0,"vgg":0,'mobileformer':0},
            }
    optimalp=optimaldic[username][dnn]

    #实例化lsmmodel
    Gausspro=Gaussmodel(netname=dnnname,username=username)

    pprf=str(parent_dir)+"/latency/Gausslatency"+str(dnn)+str(totalnumber)+".txt"
    with open(pprf,'w') as prf:
        prf.write(f'real_latency       predict_latency            step       partitionpoint                 ')  
        prf.write(f'real_latencyclient     predict_client       real_latencytrans      predict_trans       real_latencyedge    predict_edge    ')  
        prf.write("predicttime    time\n")
        prf.write("\n")

    lock=threading.Lock()
    mindatanum=5
    readyflag=False
    losssum=0
    lossnum=0
    # losslist=[]
    step=0
    starttime_all=time.time()
    for epoch in range(args.epoch):
        #判断是否client要结束
        if(taskifon[dnnnumber]==False):
            break
        partition_lmin=0
        predict_starttime=time.time()
        if readyflag ==False:  #判断数据是否满足
            if len(Gausspro.latencypool_pp)>=mindatanum:
                readyflag=True 
                print("readyflag is true now. step ",step)
            if epoch<=1:
                partition_lmin=0
            elif epoch<=3:
                partition_lmin=optimalp
            else:
                partition_lmin=random.randint(0,partitionmax)       # 
        else :  #预测分割点
            partition_lmin,mu_sum,mu=Gausspro.predict()
        predict_endtime=time.time()
        
        lock.acquire()
        ddpartitionlist[dnnnumber]=partition_lmin
        lock.release()
        ##等待网络完成一帧推理
        update=0
        while update==0 :
            update=0 
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
        actuallatencystepold=actuallatencystep[dnnnumber]
        lock.release()
        
        if readyflag :
            #loss
            predictpp,mu_sum,mu=Gausspro.predict()
            loss=mu_sum[partitionpoint_use][dnnnumber]-real_latency

            #记录
            if epoch>=20:
                losssum+=math.fabs(loss)
                lossnum+=1
            with open(pprf,'a') as prf:
                prf.write(f'{real_latency:<23}{str(mu_sum[partitionpoint_use][0]):<23}{str(step):<23}{str(partitionpoint_use):<23}')
                prf.write(f"{str(real_latencyclient):<23}{str(mu[0][partitionpoint_use][0]):<23}{str(real_latencytrans):<23}{str(mu[1][partitionpoint_use][0]):<23}{str(real_latencyedge):<23}{str(mu[2][partitionpoint_use][0]):<23}")
                prf.write(f'{(predict_endtime-predict_starttime):<23}{str(time.time()-starttime_all):<23}')
                prf.write(f'\n')
        
        if step>0:
            Gausspro.update_data(actual_latency=real_latency,a_client=real_latencyclient,a_edge=real_latencyedge,a_trans=real_latencytrans,pp=partitionpoint_use)

        step+=1

    print(username," ",dnnname," ",totalnumber," average loss : ",losssum/lossnum)
       
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
HOST = '0.0.0.0'

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
       t=threading.Thread(target=guassthread,args=(i,args.username,dnnslist[i]))
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


    
    
    
    
    




