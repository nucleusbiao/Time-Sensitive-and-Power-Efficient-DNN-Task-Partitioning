
import time
import argparse
import os
import sys
from pathlib import Path

this_dir = str(Path(__file__).resolve().parent)
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from communicationdeep import clientCommunication
import subprocess

def parse_argse():
    parser=argparse.ArgumentParser()
    parser.add_argument('--host',dest='host',help="server host",default='192.168.31.33',type=str)
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
    dnnlist=["yolo"]
    sarttimelist=[0,0,0,9,12]
    timelength=[7,20,20,11,8]
    portlist=[9050,9051,9052,9053,9054,9055,9056]
elif args.devicename=='u':
    dnnlist=["yolo"]
    sarttimelist=[0,10,16,26]
    timelength=[7,16,10,6]
    portlist=[9064,9065,9066,9067]
elif args.devicename=='tx2':
    dnnlist=["resnet","yolo","alexnet"]
    sarttimelist=[0,0,0,9,12]
    timelength=[20,20,20,11,8]
    portlist=[9030,9031,9032,9033,9034,9035]
elif args.devicename=='ly':   
    dnnlist=["mobilenetv2"]
    sarttimelist=[0,0,10,28]
    timelength=[12,16,6,4]
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

#连接服务器server
communication = clientCommunication(args.host, args.port)
communication.connect()
print("has connected edge master")

os.system('rm -r '+str(this_dir)+"/latency")
print("rm ",str(this_dir)+"/latency")
os.system('mkdir '+str(this_dir)+"/latency")

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
                    startargs='python3 client_gauss.py --host '+args.host+ ' --port='+str(portlist[i])+' --dnn_model='+str(dnnlist[i])+ ' --breaktime='+str(timelength[i])+\
                                ' --netnumber='+str(i)+' --hostp='+args.hostp+' --portp='+str(portp)
                    child1=subprocess.Popen(startargs,shell=True)    
                    print(dnnlist[i],"started at ",timecounter)

    except KeyboardInterrupt or TypeError or OSError:
          break    

   
















