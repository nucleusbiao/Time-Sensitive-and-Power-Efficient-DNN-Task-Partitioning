#24/9/4
#client 控制开启，停止，带宽。
#定时开启，先向edge 发送启动请求，收到确认信息后启动任务子线程。
#9/9 按照device 设置任务。

import time
# import threading
import argparse
import os
from communicationdeep import clientCommunication
import subprocess

import sys
from pathlib import Path

this_dir = str(Path(__file__).resolve().parent)
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)


def parse_argse():
    parser=argparse.ArgumentParser()
    parser.add_argument('--host',dest='host',help="server host",default='192.168.31.33',type=str)
    #parser.add_argument('--bandrand',dest="bandrand",action='store_true',help="if add this , bandrand.")
    parser.add_argument('--port',dest="port",help="sever port",default=8080,type=int)
    parser.add_argument('--hostp',dest='hostp',help="power host master",default='192.168.31.160',type=str)
    parser.add_argument('--portp',dest="portp",help="power port master",default=8060,type=int)
    parser.add_argument('--devicename',dest="devicename",default='k',type=str)
    args=parser.parse_args()
    return args

args=parse_argse()
print(args)
##》》》
#配置任务和开始时间，持续时间，端口！！！
dnnlist=["alexnet","resnet"]
sarttimelist=[3,5]
timelength=[2,2]
portlist=[9080,9081,9082,9083]
if args.devicename=='k':
    dnnlist=["resnet","yolo"]
    sarttimelist=[0,0,0,2,22]
    timelength=[30,30,8,4,10]
    portlist=[9050,9051,9052,9053,9054,9055,9056]
elif args.devicename=='u':
    dnnlist=["mobilenetv2","yolo","resnet","alexnet"]
    sarttimelist=[28,20,16,26]
    timelength=[4,6,10,6]
    portlist=[9064,9065,9066,9067]
elif args.devicename=='tx2':
    dnnlist=["yolo"]
    sarttimelist=[0,0,0,3,4,5]
    timelength=[30,30,20,2,2,2]
    portlist=[9030,9031,9032,9033,9034,9035]
elif args.devicename=='ly':   
    dnnlist=["yolo","mobilenetv2","alexnet"]
    sarttimelist=[6,12,10,28]
    timelength=[10,16,6,4]
    portlist=[9080,9081,9082]
else:
    print("please input the right device name")
##《《《

##》》》
#配置与power通信的端口
portplistsart=8090
##《《《

passwarddic={'k':"6","u":"6","tx2":"6","ly":"123"}  #passward 用于设置带宽
passward=passwarddic[args.devicename]

starttime_all=time.time()
timecounter=-1
timestart=starttime_all

#deltrickled()
#连接服务器server
communication = clientCommunication(args.host, args.port)
communication.connect()
print("has connected edge master")

# #连接power master
communication_pmaster = clientCommunication(args.hostp, args.portp)
communication_pmaster.connect()
print("has connected power master")

os.system('rm -r '+str(parent_dir)+"/latency")
print("rm ",str(parent_dir)+"/latency")
os.system('mkdir '+str(parent_dir)+"/latency")


startflag=1
while True:
    try:
        time.sleep(0.01)
        timenow=time.time()
        #定时启动任务
        if timenow-timestart>=(timecounter+1)*60:
            timecounter+=1
            print("time: ",timecounter)
            for i in range(len(dnnlist)):
                if(sarttimelist[i]==timecounter):
                    communication.send_msg([dnnlist[i],portlist[i]])  #发送启动任务信息
                    communication.receive_msg()  #server确认信息  待：设置端口
                    #启动
                    portp=portplistsart+i
                    startargs='python3 '+str(this_dir)+'/client_gauss_vp.py --host '+args.host+ ' --port='+str(portlist[i])+' --dnn_model='+str(dnnlist[i])+ ' --breaktime='+str(timelength[i])+\
                                ' --netnumber='+str(i)+' --hostp='+args.hostp+' --portp='+str(portp)
                    #print(startargs)
                    child1=subprocess.Popen(startargs,shell=True)    
                    print(dnnlist[i],"started at ",timecounter)
                    #给power端发送tcp启动请求
                    communication_pmaster.send_msg([portp,i])  #[port,netnumber]

    except KeyboardInterrupt or TypeError or OSError:
          #deltrickled() 
          break    

   
















