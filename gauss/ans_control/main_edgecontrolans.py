#edge 控制加载任务和关闭任务
# import maintagentbbatch_ljdan_powernew_guass_s
import time
import os
import subprocess
import argparse
from communication import serverCommunication
import threading
import sys
from pathlib import Path

this_dir = str(Path(__file__).resolve().parent)
sys.path.append(this_dir)
parent_dir = str(Path(__file__).resolve().parent.parent)
print(parent_dir)
sys.path.append(parent_dir)
parent_parent_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(parent_parent_dir)

def parse_argse():
    parser=argparse.ArgumentParser()
    parser.add_argument('--devicelist',dest='devicelist',help="device",default='k',type=str)
    parser.add_argument('--portlist',dest='portlist',help="portlist",default='8080',type=str)
    parser.add_argument("--bandrand",dest="bandrand",help="bandrand",action='store_true')
    args=parser.parse_args()
    return args
def commuicatethread(device,port):
    global totalnumber
    communication=serverCommunication("0.0.0.0",port)
    conn, addr = communication.accept_conn()
    print("已连接设备：",addr)
    while True:
        receive = communication.receive_msg(conn)  #dnn , port,
        # print("reveive: ",receive)
        #若收到请求信息，启动进程
        startargs='python3 '+str(this_dir)+ '/edge_server_main_s.py  --port='+str(receive[1])+' --dnn='+str(receive[0]+' --totalnumber='+str(totalnumber))
        child1=subprocess.Popen(startargs,shell=True)    
        print(receive[0]," started ",device)
        #待添加，端口检查~~~
        communication.send_msg(conn, "ok")
        totalnumber+=1

args=parse_argse()
print(args)      
devicelist=args.devicelist.split(",")
for i in range(len(devicelist)):
    devicelist[i]=devicelist[i]
#设备，端口。
devicehost=["0.0.0.0"]
portlist=args.portlist.split(",")
for i in range(len(portlist)):
    portlist[i]=int(portlist[i])

starttime_all=time.time()
timestart=starttime_all
communicationlist=[]
#任务总数
totalnumber=0

for i in range(len(devicelist)):
    thread=threading.Thread(target=commuicatethread,args=(devicelist[i],portlist[i]))
    thread.daemon=True
    thread.start()

timecounter=0
startime_all=time.time()
bandwidthlist=[5,10,15,20,25,30,35,40,45,50,50]
while True:
    try:
        time.sleep(0.01)
        if args.bandrand:
            if(time.time()-startime_all>60*(timecounter)):
                if timecounter<len(bandwidthlist):
                    bandwidth=1024*bandwidthlist[timecounter]
                    print(str(timecounter),'min   ',)
                    #echo 6, 6为用户密码。wlo1 查看ifconfig 配置。
                    res_ch=os.system("echo 6 | sudo -S wondershaper clear wlo1")  
                    strwidth="echo 6 | sudo -S wondershaper wlo1 "+str(bandwidth)+" "+str(bandwidth)
                    res_ch=os.system(strwidth)
                    print("res_ch",res_ch," ", timecounter,' min  bandwidth ',bandwidth)
                    timecounter+=1
                else:
                    print(str(timecounter),'min  ')
                    res_ch=os.system("echo 6 | sudo -S wondershaper clear wlo1 ")
                    if timecounter>=len(bandwidthlist)+1:
                        break
                    timecounter+=1
        elif timecounter==0:
            res_ch=os.system("echo 6 | sudo -S wondershaper clear wlo1 ")
            timecounter+=1
    except KeyboardInterrupt or TypeError or OSError:
        res_ch=os.system("echo 6 | sudo -S wondershaper clear wlo1 ")
        break





