#edge 控制加载任务和关闭任务
# import maintagentbbatch_ljdan_powernew_guass_s
#24.10.28 改为多进程队列通讯 multiprocessing
#24.11.18 增加贝叶斯决策



#**************edge 主体部分**************#
#-----------------------------------#
import os
import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
print(parent_dir)
sys.path.append(parent_dir)
parent_parent_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(parent_parent_dir)

import torch
import argparse
import time
import threading
from gaussmodel2 import Gaussmodel
import math
import json
import numpy as np
from models.vgg16 import vgg16
from models.tiny_yolo import tinyYolo
from models.mobilenetv2 import mobilenetv2
from models.resnet import resnet50
from models.alexnet import alexnet
from models.Googlenet import googlenet

from communicationdeep import serverCommunication
import random


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
    global partitioncountdict
    step=-1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device="cpu"
    print(f'totalnumber {netnumer}, using {device} device.')
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
    elif dnn == 'googlenet':
        model=googlenet()
        model.eval()
        acttotal=18
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
    else:
        print('please input right net name')
        assert False
    #w
    #model.cuda()
    model.to(device)
    print(f'{netnumer} 模型加载时间：{time.time()-time1}',)
    filepath=str(parent_dir)+"/latency/latency_server"+"_"+str(dnn)+str(netnumer)+".txt"
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
                    data = torch.autograd.Variable(data)
                    prediction = model(data.to(device), server=True, partition=partition_point)
                    res = prediction.data
                    res=res.to('cpu')
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
    global cputotalsum
    # global powerlist
    global cputotallist
    global memaverlist
    # global cpu_timelenth
    global powersumlist #用于存放一段时间内的总
    global powercountlist
    global powerperiodtime
    global powerperiodcount
    global cputotalperiod
    global cputotalperiodcount

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
                #                              0                1             2         3            4            5                6          7        8      9        10 
            #接受总延时数据   ladata=[real_latency_offrame,partitionPoint,real_client,real_trans,real_edgeinfer,cpuutotalsum,cpu_timelenth,cputotal,memaver,powersum,powercount]
            latencydata=communication.receive_msg(conn)  #[actuallatency,partitonpoint,actual_client,actual_trans,actual_edge,cpuu,cpu_timelenth,cputotal,memaver]
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
            cputotalsum[netnumber]=+latencydata[5]
            #cpu_timelenth[netnumber]=latencydata[6]
            cputotallist[netnumber]=latencydata[7]
            memaverlist[netnumber]=latencydata[8]
            powersumlist[netnumber]=latencydata[9]
            powercountlist[netnumber]=latencydata[10]
            powerperiodtime[netnumber].append(latencydata[9])
            powerperiodcount[netnumber].append(latencydata[10])
            cputotalperiod[netnumber].append(latencydata[7])
            cputotalperiodcount[netnumber].append(latencydata[6])
            lock.release()
        except  KeyboardInterrupt or TypeError or OSError:
            taskifon[netnumber]=False
            # print("taskifon is false ",dnnslist[netnumber] )
    taskifon[netnumber]=False
    communication.close_channel()
    # print("taskifon is false ",dnnslist[netnumber] )

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
    global dnnsnum
    global actuallatencyclient
    global actuallatencytrans
    global actuallatencyedge
    global taskifon
    global cputotalsum
    # global powerlist
    global cputotallist
    global memaverlist
    global optimaldic
    global partitioncountdict
    global addflag
    global predict_plist
    global powersumlist
    global powercountlist
    
    actuallatencystepold=-1
    dnnname=dnn
    partitionmax=partitioncountdict[dnnname]
    # optimalp=optimaldic[username][dnn]

    #实例化lsmmodel
    Gausspro=Gaussmodel(netname=dnnname,username=username)

    pprf=str(parent_dir)+"/latency/Gausslatency"+str(dnn)+str(dnnnumber)+".txt"
    with open(pprf,'w') as prf:
        prf.write(f'real_latency       predict_latency            step       partitionpoint                 ')  
        prf.write(f'real_latencyclient     predict_client       real_latencytrans      predict_trans       real_latencyedge    predict_edge    ')  
        prf.write(f'      cputotal        poweraver  memaver_ ')
        prf.write("\n")


    lock=threading.Lock()
    mindatanum=3
    readyflag=False
    losssum=0
    lossnum=0
    # losslist=[]
    step=0

    for epoch in range(30000):
        #判断是否client要结束
        if(taskifon[dnnnumber]==False):
            break
        if readyflag ==False:  #判断数据是否满足
            if len(Gausspro.latencypool_pp)>=mindatanum:
                readyflag=True 

        ##等待网络完成一帧推理
        update=0
        while update==0 :
            update=0  
            if actuallatencystepold<actuallatencystep[dnnnumber]:
                update+=1  
            time.sleep(0.001)

        #获取时间延时和分割点
        powerarver=0
        lock.acquire()
        real_latency=actuallatencylist[dnnnumber]
        partitionpoint_use=partition_pointlistofthelatency[dnnnumber] #如果发送信息快了一步，就不对应了！所以，partition_pointlistofthelatency 更新须与actuallatencylist同步
        real_latencyclient=actuallatencyclient[dnnnumber]
        real_latencytrans=actuallatencytrans[dnnnumber]
        real_latencyedge=actuallatencyedge[dnnnumber]
        #获取cpuu,和powersum ~~~~
        cputotal_=cputotallist[dnnnumber]
        memaver_=memaverlist[dnnnumber]
        if powercountlist[dnnnumber]!=0:
            powerarver=powersumlist[dnnnumber]/powercountlist[dnnnumber]
        if partitionpoint_use>partitionmax:
                print("wrong",partitionpoint_use) 
        actuallatencystepold=actuallatencystep[dnnnumber]
        lock.release()
        
        if readyflag :
            #预测
            predictpp,mu_sum,cov_sum,mu=Gausspro.predict()
            loss=mu_sum[partitionpoint_use]-real_latency

            #若收到更新falg，更新延时全局变量 待~~~
            mu_sum_s=np.array(mu_sum).squeeze()
            lock.acquire()
            predict_plist[dnnnumber]=mu_sum_s.tolist()
            lock.release()

            #记录
            if epoch>5:
                losssum+=math.fabs(loss)
                lossnum+=1
            # averloss=losssum/lossnum
            with open(pprf,'a') as prf:
                prf.write(f'{real_latency:<23}{str(mu_sum[partitionpoint_use]):<23}{str(step):<23}{str(partitionpoint_use):<23}')
                prf.write(f"{str(real_latencyclient):<23}{str(mu[0][partitionpoint_use][0]):<23}{str(real_latencytrans):<23}{str(mu[1][partitionpoint_use][0]):<23}{str(real_latencyedge):<23}{str(mu[2][partitionpoint_use][0]):<23}")
                # prf.write(f'{cpu_:<23}{str(powers_):<23}{str(cputotal_):<23}{str(memaver_):<23}')
                prf.write(f'{str(cputotal_):<23}{str(powerarver):<23}{str(memaver_):<23}')
                prf.write(f'\n')
          
        if step>0:
            Gausspro.update_data(actual_latency=real_latency,a_client=real_latencyclient,a_edge=real_latencyedge,a_trans=real_latencytrans,pp=partitionpoint_use)
        step+=1

    print(username," ",dnnname," ",dnnnumber," average loss : ",losssum/lossnum)
 
#全局变量

HOST = '0.0.0.0'

#延时信息
dnnsnum=7
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

cputotalsum=[0.0 for _ in range(dnnsnum)]  
cputotallist=[0.0 for _ in range(dnnsnum)]  #一个分割点一帧推理的cpu利用率和
# cpu_timelenth=[0.0 for _ in range(dnnsnum)] #一个分割点一帧推理的cpu利用率计数
memaverlist=[0.0 for _ in range(dnnsnum)]
powersumlist=[0.0 for _ in range(dnnsnum)]  #一个分割点一帧推理的功耗和
powercountlist=[0 for _ in range(dnnsnum)]  #一个分割点一帧推理的功耗计数
predict_plist=[[] for _ in range(dnnsnum)]  #用于存放最新的能耗预测值

powerperiodtime=[[] for _ in range(dnnsnum)]  #用于blackbox统计power，一段时间的功耗和列表，
powerperiodcount=[[] for _ in range(dnnsnum)] #用于blackbox统计power，一段时间的功耗计数列表，
cputotalperiod=[[] for _ in range(dnnsnum)]   #用于blackbox统计cpu利用率，一段时间列表
cputotalperiodcount=[[] for _ in range(dnnsnum)] #用于blackbox统计cpu利用率计数，一段时间列表



#任务是否关闭标志
taskifon=[True for _ in range(dnnsnum)]
#设备任务是否改变标志
taskchangeflag={"k":False,"tx2":False,"ly":False,"u":False}
devicednnnumber={"k":{},"tx2":{},"ly":{},"u":{}}  #"k":[dnnnumber:dnnname,]  #存放各个设备上的任务编号及其任务类型
taskchangecount={"k":0,"tx2":0,"ly":0,"u":0}

#最优分割点
optimaldic={'k':{"alexnet":2,"mobilenetv2":0,"yolo":16,"resnet":0,"vgg":0,"googlenet":0},  #注："googlenet" 未核实！！！！！ 待~~~~
            'u':{"alexnet":8,"mobilenetv2":5,"yolo":20,"resnet":0,"vgg":0,"googlenet":0},
            'tx2':{"alexnet":2,"mobilenetv2":0,"yolo":16,"resnet":0,"vgg":0,"googlenet":0},
            'ly':{"alexnet":2,"mobilenetv2":0,"yolo":0,"resnet":0,"vgg":0,"googlenet":0},
            }
#最大分割点
partitioncountdict={"vgg":22,"yolo": 31,"resnet":20,"alexnet":11,"mobilenetv2":21,"googlenet":18}

#add 测试数据标志
addflag=False

##待补充，扩充全部变量大小
def globaldataexpend():
    ddpartitionlist.append[0]

# mainnumber=""    ###区分文件路径

def edgetotalthread(port,dnn,totalnumber,username):  #username=device , 
    global partitioncountdict
    #启动延时预测模型线程
    t=threading.Thread(target=guassthread,args=(totalnumber,username,dnn))
    t.setDaemon(True)
    t.start()

    #创建edge推理线程
    t=threading.Thread(target=edge,args=(dnn,totalnumber))
    t.setDaemon(True)
    t.start()
        #创建与client通信子线程,与DNN client 通讯 子线程， 负责接收延时，发送分割点
    
    #启动与client通信线程
    th=threading.Thread(target=communicationthwithclient,args=(HOST,port,totalnumber,partitioncountdict[dnn]))    
    th.setDaemon(True)
    th.start()

firstblakboxflag=True

###       (**next_point_to_probe) 例如：{0:4,1:7}  netnumber:pp
def black_box_function(next_point_to_probe,numberlist):#该设备上任务根据分割点做出决策，等待运行一段时间后,返回能耗
    global ddpartitionlist
    global cputotalsum
    global cputotallist
    # global cpu_timelenth
    global firstblakboxflag
    global powerperiodtime
    global powerperiodcount
    global cputotalperiod
    global cputotalperiodcount

    lockc=threading.Lock()

    #1. 方法2：将决策分割点放入全局变量:ddpartitionlist 中，等待直至实际分割点都执行后，cpu清零。
    for netnumber,pp in next_point_to_probe.items():
        lockc.acquire()
        ddpartitionlist[netnumber]=round(pp)
        lockc.release()
    readyflag=False
    while(readyflag==False):
        readyflag=True
        for netnumber,pp in next_point_to_probe.items():
            lockc.acquire()
            if partition_pointlistofthelatency[netnumber]!=round(pp):
                readyflag=False
            lockc.release()
        time.sleep(0.005)
    for netnumber,pp in next_point_to_probe.items():  #cpu清零,功耗period清零
        lockc.acquire()
        powerperiodtime[netnumber]=[]
        powerperiodcount[netnumber]=[]
        cputotalperiod[netnumber]=[]
        cputotalperiodcount[netnumber]=[]
        lockc.release()
    
    #清空延时统计
   
    #2.等待启动，等待执行一段时间
    if firstblakboxflag==True:
        time.sleep(10)
        firstblakboxflag=False
    else:
        time.sleep(5)
    #3.返回这段时间的平均能耗,从全局变量拿出来，并清空。 等等，这里统计的应该是cpu利用率和时间,用power模型计算功耗！！
    lockc.acquire()
    powerlist=[]
    for netnumber in numberlist:
        power=0.0
        #if cpu_timelenth[netnumber]!=0:
        powersum=0    
        powersum=sum(powerperiodtime[netnumber])
        countsum=sum(powerperiodcount[netnumber])
        #print("powerperiodtime ",powerperiodtime[netnumber],"powerperiodcount ",powerperiodcount[netnumber])
        if countsum!=0:
            power=powersum/countsum
        powerlist.append(power)
    
    #这段时间的cpu利用率。
    cpulist=[]
    for netnumber in numberlist:
        cpu=0.0
        #if cpu_timelenth[netnumber]!=0:
        cpusum=0    
        for i in range(len(cputotalperiod[netnumber])):
            cpusum+=(cputotalperiod[netnumber][i]*cputotalperiodcount[netnumber][i])
            cpu=cpusum/sum(cputotalperiodcount[netnumber])
        cpulist.append(cpu)

    lockc.release()
    # powerresult = [-1.0* power for power in powerlist]
    print('balck box pp is :',next_point_to_probe,"powerlist: ",powerlist," cpulist: ",cpulist)

    return powerlist,cpulist

#目前只针对单个任务实现
def constraint_function(**next_point_to_probe):
# def constraint_function(**next_point_to_probe):
    global predict_plist
    lock=threading.Lock()
    #1.获取分割点延时的预测值
    lock.acquire()
    result=np.array(predict_plist[0])
    lock.release()
    #2.返回延时
    return result

def overallcontrol(username): #每个设备上有一个 username，设备名称
    global taskchangeflag
    global devicednnnumber
    global partitioncountdict
    global taskchangeflag
    global taskchangecount
    global firstblakboxflag

    print("overallcontrol username: ",username)
    lockc=threading.Lock()
    cstep=0
    while True:
        #****基础部分***#
        # ppcount=7
        while taskchangeflag[username]==False:
            time.sleep(0.001)
        lockc.acquire()
        taskchangeflag[username]=False
        lockc.release()
        firstblakboxflag=True
        #获取任务编号和名字
        # tasknums=len(devicednnnumber[username])  #devicednnnumber[username]  字典，存放改设备上任务的现阶段任务编号，及其任务名称,如 {1:"yolo"}
        pbounds_d={}
        numberlist=[]
        print("devicednnnumber: ",devicednnnumber)
        for number,dnn in devicednnnumber[username].items():   
            varible=number
            pbounds_d[varible]=(0,partitioncountdict[dnn])    ## pbounds   如  {0:(0,20),3:(0,31)}
            numberlist.append(number)
        print("pbounds_d: ",pbounds_d)
        numberlist.sort()
        print("numberlist: ",numberlist)

        bayesf=str(parent_dir)+"/latency/bayeslog"+username+str(taskchangecount[username])+".txt"
        with open(bayesf,"w") as f:
            for i in range(len(numberlist)):
                f.write(f'partition_point{str(numberlist[i])}         ')
            for i in range(len(numberlist)):
                f.write(f'cpu{str(numberlist[i])}         power{str(numberlist[i])}   ')    
            f.write("\n")
        
        next_point_to_probe={}
        #****循环部分***#
        while taskchangeflag[username]==False:
            cstep+=1
            # 1.随机给出决策
            for number in numberlist:
                next_point_to_probe[number]=random.randint(0,pbounds_d[number][1])
            print("Next point to probe is:", next_point_to_probe)

            #2.等待一段时间，获取设备能耗。
            powerlist,cpulist = black_box_function(next_point_to_probe,numberlist)

            with open(bayesf,"a") as f:
                for number in numberlist:
                    f.write(str(round(next_point_to_probe[number])))
                    f.write("    ")
                for i in range(len(powerlist)):
                    f.write(str(cpulist[i]))
                    f.write("    ")
                    f.write(str(powerlist[i]))
                    f.write("    ")
                f.write("\n")

#*****************************************#

def parse_argse():
    parser=argparse.ArgumentParser()
    parser.add_argument('--devicelist',dest='devicelist',help="device",default='k',type=str)
    parser.add_argument('--portlist',dest='portlist',help="portlist",default='8080',type=str)
    parser.add_argument("--bandrand",dest="bandrand",help="bandrand",action='store_true')
    args=parser.parse_args()
    return args

#与设备总控连接，共设备个数个线程，负责启动任务（接收到任务类型和端口号后）
def commuicatethread(device,port):
    global totalnumber
    global devicednnnumber
    global taskchangeflag
    global taskchangecount
    communication=serverCommunication("0.0.0.0",port)
    conn, addr = communication.accept_conn()
    print("已连接设备：",addr)
    lockc=threading.Lock()
    #这里，驱动贝叶斯决定线程，但是，没有动态调整功能  待~~~~
    oct=threading.Thread(target=overallcontrol,args=(device,))
    oct.daemon=True
    oct.start()
    while True:
        lockc.acquire()
        dnn,dnnport = communication.receive_msg(conn)  #dnn , port,
        # print("reveive: ",receive)
        #通知贝叶斯进程，有任务变动
        devicednnnumber[device][totalnumber]=dnn   #devicednnnumber   "k":[dnnnumber:dnnname]  #存放各个设备上的任务编号及其任务类型
        taskchangeflag[device]=True
        taskchangecount[device]+=1
        print("comu devicednnnumber: ",devicednnnumber)
        #若收到请求信息，启动线程
        totalnumberc=totalnumber
        lockc.release()
        edgethread=threading.Thread(target=edgetotalthread,args=(dnnport,dnn,totalnumberc,device))   #port,dnn,totalnumber,username
        edgethread.start()
        print(dnn," started ",device)
        #待添加，端口检查~~~
        communication.send_msg(conn, "ok")
        lockc.acquire()
        totalnumber+=1
        lockc.release()
        
args=parse_argse()
print(args)      
devicelist=args.devicelist.split(",")
for i in range(len(devicelist)):
    devicelist[i]=devicelist[i]
#设备，端口。
devicehost=["0.0.0.0"]
deviceportlist=args.portlist.split(",")
for i in range(len(deviceportlist)):
    deviceportlist[i]=int(deviceportlist[i])

starttime_all=time.time()
timestart=starttime_all
communicationlist=[]
#任务总数
totalnumber=0
import os

# os.system('rm -r latency/')
# os.system('mkdir latency')

for i in range(len(devicelist)):
    thread=threading.Thread(target=commuicatethread,args=(devicelist[i],deviceportlist[i]))
    thread.daemon=True
    thread.start()

timecounter=0
bandwidthlist=[20, 15, 5, 15, 7, 10, 10, 20, 30, 7, 20, 15, 7, 5, 5, 30, 7, 15, 10, 7, 7, 7, 5, 25, 20, 7, 15, 30, 10, 20, 25, 5]
startime_all=time.time()

while True:
    # try:
        time.sleep(2)





