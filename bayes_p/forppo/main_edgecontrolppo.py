
#25.2.25 改为PPO


#**************edge 主体部分**************#
#-----------------------------------#
import os
import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)
parent_parent_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(parent_parent_dir)

import torch
import argparse
import time
import threading
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
import joblib
import pandas as pd
from ppomodel import PPO

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

            time.sleep(0.001)

        except KeyboardInterrupt or TypeError or OSError:
            del model
            torch.cuda.empty_cache()
            time.sleep(2)
            print(dnn," edge closed now.")
            #taskifon[netnumer]=False
    del model
    torch.cuda.empty_cache()
    time.sleep(2)
    #taskifon[netnumer]=False
    print(dnn," edge closed now.")

def communicationthwithclient(host,port,netnumber,actotal,username):
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
    global cputotallist
    global memaverlist
    global latencysum
    global latencycount
    global cputotalperiod
    global cputotalperiodcount
    global black_box_readypp
    global suggestreadypp

    print(f'ready to connect{port}')
    communication=serverCommunication(host,port)
    conn, addr = communication.accept_conn()
    lock=threading.Lock()
    ddpartition=0
    while True:
        try:
          
            lock.acquire()
            ddpartition=ddpartitionlist[netnumber]
            lock.release()

            communication.send_msg(conn, ddpartition)
            lock.acquire()    
            black_box_readypp[netnumber]=ddpartition  #通知blackbox 分割点部署好了
            lock.release()

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
                #                              0                1             2         3            4            5                6          7        8
            #接受总延时数据   ladata=[real_latency_offrame,partitionPoint,real_client,real_trans,real_edgeinfer,cputotalsum,cpu_timelenth,cputotal,memaver]
            latencydata=communication.receive_msg(conn)  #[actuallatency,partitonpoint,actual_client,actual_trans,actual_edge,cpuu,cpu_timelenth,cputotal,memaver]
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
            cputotalsum[netnumber]=+latencydata[5]
            cputotallist[netnumber]=latencydata[7]
            memaverlist[netnumber]=latencydata[8]
            latencysum[netnumber]+=latencydata[0]
            latencycount[netnumber]+=1 ##！！！！！！！！！！！！！！
            cputotalperiod[netnumber].append(latencydata[7])
            cputotalperiodcount[netnumber].append(latencydata[6])
            lock.release()
        except  KeyboardInterrupt or TypeError or OSError:
            lock.acquire()
            taskifon[netnumber]=False
            lock.release()
            print("cclient end except")
           
    lock.acquire()
    taskifon[netnumber]=False
    lock.release()
    print("cclient end ",netnumber," ",taskifon[netnumber])
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
    # global predict_plist
    global devicednnnumber
    global taskchangeflag
    global taskchangecount
    
    actuallatencystepold=-1
    dnnname=dnn
    partitionmax=partitioncountdict[dnnname]
    # optimalp=optimaldic[username][dnn]

    pprf=str(parent_dir)+"/latency/pporeallatency"+str(dnn)+str(dnnnumber)+".txt"
    with open(pprf,'w') as prf:
        prf.write(f'real_latency                step       partitionpoint                 ')  
        prf.write(f'real_latencyclient            real_latencytrans           real_latencyedge      ')  
        prf.write(f'cpusum       cputotal    memaver_\n')
    lock=threading.Lock()
    mindatanum=3
    readyflag=False
    losssum=0
    lossnum=0
    step=0
    satisfynum=0
    totalnum=0

    for epoch in range(30000):
        ##等待网络完成一帧推理
        update=0
        while update==0 :
            update=0  
            if actuallatencystepold<actuallatencystep[dnnnumber]:
                update+=1  
            if(taskifon[dnnnumber]==False): #判断是否client要结束
                break 
            time.sleep(0.001)
        if(taskifon[dnnnumber]==False): #判断是否client要结束
                break 

        #获取时间延时和分割点
        lock.acquire()
        real_latency=actuallatencylist[dnnnumber]
        partitionpoint_use=partition_pointlistofthelatency[dnnnumber] #如果发送信息快了一步，就不对应了！所以，partition_pointlistofthelatency 更新须与actuallatencylist同步
        real_latencyclient=actuallatencyclient[dnnnumber]
        real_latencytrans=actuallatencytrans[dnnnumber]
        real_latencyedge=actuallatencyedge[dnnnumber]
        #获取cpuu,和powersum ~~~~
        cpusum=cputotalsum[dnnnumber]
        # powers_=powerlist[dnnnumber]
        cputotal_=cputotallist[dnnnumber]
        memaver_=memaverlist[dnnnumber]
        if partitionpoint_use>partitionmax:
                print("wrong",partitionpoint_use) 
        actuallatencystepold=actuallatencystep[dnnnumber]
        lock.release()

        # predict_plist[dnnnumber]=[]
       
        #记录
        if real_latency<1.5:
            satisfynum+=1
        totalnum+=1
        with open(pprf,'a') as prf:
            prf.write(f'{real_latency:<23}{str(step):<23}{str(partitionpoint_use):<23}')
            prf.write(f"{str(real_latencyclient):<23}{str(real_latencytrans):<23}{str(real_latencyedge):<23}")
            prf.write(f'{cpusum:<23}{str(cputotal_):<23}{str(memaver_):<23}')
            prf.write(f'\n')
        step+=1

    #清除devicednnnumber[device][totalnumber]=dnn
    print("gausshtread end ",dnnname)
    del devicednnnumber[username][dnnnumber]
    lock.acquire()
    taskchangeflag[username]=True
    taskchangecount[username]+=1
    lock.release()
    print(username," ",dnnname," ",dnnnumber,"satisfyrate ",satisfynum/totalnum)
    with open(pprf,'a') as prf:
        prf.write(f'"satisfyrate ",str{satisfynum/totalnum}\n')

#全局变量
HOST = '0.0.0.0'

#延时信息
dnnsnum=10
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
cputotallist=[0.0 for _ in range(dnnsnum)]
memaverlist=[0.0 for _ in range(dnnsnum)]
# predict_plist=[() for _ in range(dnnsnum)]  #用于存放最新的延时预测值
#平均延时，用于constrain
latencysum=[0.0 for _ in range(dnnsnum)]  #延时和
latencycount=[0.0 for _ in range(dnnsnum)] #计数
cputotalperiod=[[] for _ in range(dnnsnum)]   #用于blackbox统计cpu利用率，一段时间列表
cputotalperiodcount=[[] for _ in range(dnnsnum)] #用于blackbox统计cpu利用率计数，一段时间列表
black_box_readypp=[0 for _ in range(dnnsnum)]  #用于blackbox判断分割点是否已经部署上了

suggestreadypp={"k":True,"tx2":True,"ly":True,"u":True} #新决策是否做出。通信线程决定是否传输任务全局变量里的值，还是用历史最优值。

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
    th=threading.Thread(target=communicationthwithclient,args=(HOST,port,totalnumber,partitioncountdict[dnn],username))    
    th.setDaemon(True)
    th.start()

firstblakboxflag=True

###       (**next_point_to_probe) list netnumber:pp
def black_box_function(next_point_to_probe,numberlist,powermodel,username):#该设备上任务根据分割点做出决策，等待运行一段时间后,返回能耗
    global ddpartitionlist
    global cputotalsum
    global cputotallist
    #global cpu_timelenth
    global firstblakboxflag
    global latencysum
    global latencycount
    global cputotalperiod
    global cputotalperiodcount
    global black_box_readypp
    global taskifon
    global taskchangeflag
    lockc=threading.Lock()

    #1. 方法2：将决策分割点放入全局变量:ddpartitionlist 中，等待直至实际分割点都执行后，cpu清零。
    # bbstarttime=time.time()
    dnnsnum=len(numberlist)
    for i in range(dnnsnum):
        lockc.acquire()
        ddpartitionlist[numberlist[i]]=next_point_to_probe[i]
        lockc.release()

    readyflag=False
    while(readyflag==False):
        readyflag=True
        #for netnumber,pp in next_point_to_probe.items():
        for i in range(dnnsnum):   
            lockc.acquire()
            if black_box_readypp[numberlist[i]]!=next_point_to_probe[i]:
                readyflag=False
            lockc.release()
            if taskifon[numberlist[i]]==False:
                readyflag=True
                break
        time.sleep(0.001)

    # bbreadytime=time.time()
    # for netnumber,pp in next_point_to_probe.items():
    for i in range(dnnsnum):    
        lockc.acquire()
        cputotalperiod[numberlist[i]]=[]
        cputotalperiodcount[numberlist[i]]=[]
        lockc.release()
    #清空延时统计
    # for netnumber,pp in next_point_to_probe.items():
    for i in range(dnnsnum):    
        lockc.acquire()
        latencysum[numberlist[i]]=0
        latencycount[numberlist[i]]=0
        lockc.release()

    # bbwaitstarttime=time.time()
    #2.等待启动，等待执行一段时间
    if firstblakboxflag==True:
        time.sleep(10)
        firstblakboxflag=False
    else:
        time.sleep(4)
    
    dnnssum=len(next_point_to_probe)

    #3.返回这段时间的平均能耗,从全局变量拿出来，并清空。 等等，这里统计的应该是cpu利用率和时间,用power模型计算功耗！！
    #判断是否完成
    cpulist=[]  #black box 分割点部署期间 设备上各任务的 cpu 平均利用率
    cpucountlist=[] #cpu利用率 平均值 的时长
    finishedcount=0
    while (finishedcount != dnnssum):
        finishedcount=0
        time.sleep(0.001)
        for netnumber in numberlist:
            lockc.acquire()
            if (latencysum[netnumber]!=0):
                finishedcount+=1
            lockc.release()
        if taskchangeflag[username]==True:
            break
    
    lockc.acquire()
    for netnumber in numberlist:
        cpu=0.0
        cpusum=0    
        #cpucountsum=0
        for i in range(len(cputotalperiod[netnumber])):
            cpusum+=(cputotalperiod[netnumber][i]*cputotalperiodcount[netnumber][i])
        cpucountsum=sum(cputotalperiodcount[netnumber])
        if cpucountsum!=0:
            cpu=cpusum/cpucountsum
        cpucountlist.append(cpucountsum)
        cpulist.append(cpu)

    #返回平均延时
    latencylist=[]
    for netnumber in numberlist:
        if latencysum[netnumber]!=0:
            latencylist.append(latencysum[netnumber]/latencycount[netnumber])
        else:
            latencylist.append(4.0)

    #使用线性模型预测
    cpulist_=pd.DataFrame({"cputotal":cpulist})
    powerlist=powermodel.predict(cpulist_)
    lockc.release()
    #缩放，0~1
    result = [(30.0 / power - 2)/8.0 for power in powerlist]
    #print('balck box pp is :',next_point_to_probe,"powerlist: ",powerlist,"result: ",result)
    # bbendtime=time.time()
    
    return result,powerlist,latencylist,cpucountlist

def getreward(numberlist,powerresult,latencylist):

    #print("powerresult",powerresult)
    #print("latencylist",latencylist)
    latencyresult=0
    for la in latencylist:
        latencyresult+=la
    latencyresult=latencyresult/len(latencylist)
    reward=powerresult+0.5/latencyresult-1

    return reward

def overallcontrol(username:str): #每个设备上有一个 username，设备名称
    global taskchangeflag
    global devicednnnumber
    global partitioncountdict
    global taskchangeflag
    global taskchangecount
    global firstblakboxflag
    global poweraverofthedivice

    print("overallcontrol username: ",username)
    lockc=threading.Lock()
    cstep=0

    #加载功耗模型
    powermodelpath=str(parent_parent_dir)+"/models/powermodel/powermodel_"+username+".pkl"
    powermodel=joblib.load(powermodelpath)

    device = torch.device('cuda') if torch.cuda.is_available() \
                            else torch.device('cpu')
     
    # ----------------------------------------- #
    # 参数设置
    # ----------------------------------------- #
    num_episodes = 100  # 总迭代次数
    gamma = 0.9  # 折扣因子
    actor_lr = 1e-3  # 策略网络的学习率
    critic_lr = 1e-3  # 价值网络的学习率
    n_hiddens = 16  # 隐含层神经元个数
    env_name = 'CartPole-v1'
    return_list = []  # 保存每个回合的return
    n_states=1
    n_actions=1
    batchsize=16
    starttrain=32
    epocheach=10

    #设备功耗均值
    powersumlist=[]
    powercountlist=[]
    powertotal=0
    powersumtotal=0
   
    while True:
        print("poweraver",poweraverofthedivice)
        #****基础部分***#
        #等待任务变动
        # while taskchangeflag[username]==False:
        #     time.sleep(0.001)
        lockc.acquire()
        taskchangeflag[username]=False
        lockc.release()
        firstblakboxflag=True
        #获取设备的任务编号和名字，设置分割点范围
        pbounds_d={}
        numberlist=[]
        ppmaxlist=[]
        print("devicednnnumber: ",devicednnnumber)
        lockc.acquire()
        for number,dnn in devicednnnumber[username].items():   
            varible=number
            pbounds_d[varible]=(0,partitioncountdict[dnn])    ## pbounds   如  {0:(0,20),3:(0,31)}
            numberlist.append(number)
            ppmaxlist.append(partitioncountdict[dnn])
        lockc.release()
        print("pbounds_d: ",pbounds_d)

        n_states=len(numberlist)
        n_actions=0
        for number in numberlist:
            n_actions+=pbounds_d[number][1]+1
    
        # ----------------------------------------- #
        # 模型构建
        # ----------------------------------------- #
        agent = PPO(n_states=n_states,  # 状态数，任务分割点(=任务个数)。
                    n_hiddens=n_hiddens,  # 隐含层数
                    n_actions=n_actions,  # 动作数，任务分割点（31,31,31????）
                    actor_lr=actor_lr,  # 策略网络学习率
                    critic_lr=critic_lr,  # 价值网络学习率
                    lmbda = 0.95,  # 优势函数的缩放因子
                    epochs = epocheach,  # 一组序列训练的轮次
                    eps = 0.2,  # PPO中截断范围的参数
                    gamma=gamma,  # 折扣因子
                    device = device,
                    numberlist=numberlist,
                    pbounds_d=pbounds_d,
                    batchsize=batchsize
                    )
        print("init agent n_states",n_states,"n_actions",n_actions,"n_numberlist",numberlist)
        
        # ----------------------------------------- #
        #构造数据集，保存每个回合的状态数据
        # ----------------------------------------- #
        transition_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': [],
        }
        
        #记录
        ppof=str(parent_dir)+"/latency/ppolog"+username+str(taskchangecount[username])+".txt"
        with open(ppof,"w") as f:
            f.write("reward    ")
            for i in range(len(numberlist)):
                f.write(f'partition_point{str(numberlist[i])}       latency{str(numberlist[i])}  ')
            for i in range(len(numberlist)):
                f.write(f'power{str(numberlist[i])}         ')
            for i in range(len(numberlist)):
                f.write(f'powercount{str(numberlist[i])}         ')
            f.write("poweraver     suggest_time   blackbox_time    update_time") 
            # f.write("\n")

        next_point_to_probe=[]    # list ,对应numberlist
        state=np.array([0 for _ in range(n_states)],dtype=np.float32)
      
        #****循环部分***#
        while taskchangeflag[username]==False:

            cstep+=1
            
            # 1.ppo给出下一个分割点的决策
            sugstarttime=time.time()
            if cstep>starttrain:
                next_point_to_probe = agent.take_action(state)  # 动作选择
            else:
                next_point_to_probe=[random.randint(0, ppmaxlist[i]) for i in range(n_states)]
            sugendtime=time.time()
            lockc.acquire()
            suggestreadypp[username]=True
            lockc.release()
         
            #2.等待一段时间，获取设备能耗,任务延时
            powerresult,powerlist,latencylist,cpucountlist= black_box_function(next_point_to_probe,numberlist,powermodel,username)
            if taskchangeflag[username]==True:
                break
            powerr=0
            if len(powerresult)!=0:
                for powe in powerresult:
                    if powe>0:
                        powerr=powe
                        break

            upstarttime=time.time()

            #3.获取环境信息
            reward=getreward(numberlist,powerr,latencylist)
            next_state=np.array(next_point_to_probe,dtype=np.float32)
            done=1
            
            #4.环境更新
            # 保存每个时刻的状态\动作\...
            transition_dict['states'].append(state)
            transition_dict['actions'].append(next_point_to_probe)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            # 更新状态
            state = next_state

            #5.# 模型训练
            if len (transition_dict['states'])>starttrain:
                actorloss=agent.learn(transition_dict)
                #print("reward",reward,"actorloss",actorloss)

            upendtime=time.time()
            powertotal+=powerlist[0]*cpucountlist[0]
            powersumtotal+=cpucountlist[0]
            poweraver=powertotal/powersumtotal
            poweraverofthedivice[username]=poweraver
            with open(ppof,"a") as f:
                f.write("\n")
                f.write(str(reward))
                f.write("    ")
                for i in range(n_states):
                    f.write(str(round(next_point_to_probe[i])))
                    f.write("    ")
                    f.write(str(latencylist[i]))
                    f.write("    ")
                for power in powerlist:
                    f.write(str(power))
                    f.write("    ")
                for count in cpucountlist:
                    f.write(str(count))
                    f.write("    ")
                f.write(str(poweraver))
                f.write(f'      {str(sugendtime-sugstarttime):<30}{str(upstarttime-sugendtime):<30}{str(upendtime-upstarttime):<30}')
                   
#*****************************************#

def parse_argse():
    parser=argparse.ArgumentParser()
    parser.add_argument('--devicelist',dest='devicelist',help="device",default='k',type=str)
    parser.add_argument('--portlist',dest='portlist',help="portlist",default='8080',type=str)
    parser.add_argument("--bandrand",dest="bandrand",help="bandrand",action='store_true')
    args=parser.parse_args()
    return args

#与设备总控连接，共设备个数个线程，负责启动任务（接收到任务类型和端口号后）
def commuicatethread(device,port,lockcom):
    global totalnumber
    global devicednnnumber
    global taskchangeflag
    global taskchangecount
    communication=serverCommunication("0.0.0.0",port)
    conn, addr = communication.accept_conn()
    print("已连接设备：",addr," device: ",device)
    
    #这里，驱动贝叶斯决定线程，但是，没有动态调整功能  待~~~~
    oct=threading.Thread(target=overallcontrol,args=(device,))
    oct.daemon=True
    oct.start()
    while True:
        dnn,dnnport = communication.receive_msg(conn)  #dnn , port,
        # print("reveive: ",receive)
        #通知贝叶斯进程，有任务变动
        lockcom.acquire()
        devicednnnumber[device][totalnumber]=dnn   #devicednnnumber   "k":[dnnnumber:dnnname]  #存放各个设备上的任务编号及其任务类型
        taskchangeflag[device]=True
        taskchangecount[device]+=1
        print("comu devicednnnumber: ",devicednnnumber)
        #若收到请求信息，启动线程 
        totalnumberc=totalnumber
        edgethread=threading.Thread(target=edgetotalthread,args=(dnnport,dnn,totalnumberc,device))   #port,dnn,totalnumber,username
        edgethread.start()
        totalnumber+=1
        lockcom.release()
        print(dnn," started ",device)
        #待添加，端口检查~~~
        communication.send_msg(conn, "ok")
        

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

poweraverofthedivice={'k':-1,"u":-1,"tx2":-1,"ly":-1}
#任务总数
totalnumber=0
import os
os.system('rm -r latency/')
os.system('mkdir latency')
lockcom=threading.Lock()
for i in range(len(devicelist)):
    print(devicelist[i]," ",deviceportlist[i])
    thread=threading.Thread(target=commuicatethread,args=(devicelist[i],deviceportlist[i],lockcom))
    thread.daemon=True
    thread.start()
 
timecounter=0
bandwidthlist=[55, 55, 55, 15, 15, 15, 8, 8, 8, 5,5,5, 15, 15, 15, 8, 8, 8, 5, 5,5, 55, 55, 55 ]
# bandwidthlist=[55,55,8,8,15,15,5,5,8,8,15,15,55,55,15,15,5,5,8,8,8,8,5,5,55,55,55]  #多写了一个。
startime_all=time.time()

while True:
    try:
        time.sleep(0.01)
        if args.bandrand:
            if(time.time()-startime_all>60*(timecounter)):
                if timecounter<len(bandwidthlist):
                    bandwidth=1024*bandwidthlist[timecounter]
                    print(str(timecounter),'min   ',)
                    res_ch=os.system("echo 6 | sudo -S wondershaper clear wlx08beac2cb0aa ")  
                    strwidth="echo 6 | sudo -S wondershaper wlx08beac2cb0aa "+str(bandwidth)+" "+str(bandwidth)
                    res_ch=os.system(strwidth)
                    print("res_ch",res_ch," ", timecounter,' min  bandwidth ',bandwidth)
                    timecounter+=1
                else:
                    print(str(timecounter),'min  ')
                    res_ch=os.system("echo 6 | sudo -S wondershaper clear wlx08beac2cb0aa ")
                    if timecounter>=len(bandwidthlist)+1:
                        break
                    timecounter+=1
        elif timecounter==0:
            res_ch=os.system("echo 6 | sudo -S wondershaper clear wlx08beac2cb0aa ")
            timecounter+=1
    except KeyboardInterrupt or TypeError or OSError:
        res_ch=os.system("echo 6 | sudo -S wondershaper clear wlx08beac2cb0aa ")
        break





