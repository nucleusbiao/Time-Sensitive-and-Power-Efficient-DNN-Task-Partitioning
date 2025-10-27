#edge 控制加载任务和关闭任务
# import maintagentbbatch_ljdan_powernew_guass_s
import time
import sys
from pathlib import Path

this_dir = str(Path(__file__).resolve().parent)
sys.path.append(this_dir)
parent_dir = str(Path(__file__).resolve().parent.parent)
print(parent_dir)
sys.path.append(parent_dir)
parent_parent_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(parent_parent_dir)

import os
import subprocess
import argparse
from communicationdeep import serverCommunication
import threading

def parse_argse():
    parser=argparse.ArgumentParser()
    #parser.add_argument('--host',dest='host',help="server host",default='0.0.0.0',type=str)
    # parser.add_argument('--bandrand',dest="bandrand",action='store_true',help="if add this , bandrand.")
    # parser.add_argument('--port',dest="port",default=8080,type=int)
    parser.add_argument('--devicelist',dest='devicelist',help="device",default='k',type=str)
    parser.add_argument('--portlist',dest='portlist',help="portlist",default='8080',type=str)
    parser.add_argument("--mode",dest="mode",help="client,edge,optimal",default='client',type=str)
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
        startargs='python3 ' +str(this_dir)+'/maintagentbbatch_ljdan_powernew_ceo.py  --ports='+str(receive[1])+' --dnns='+str(receive[0]+\
                        ' --totalnumber='+str(totalnumber)+' --username='+device+' --mode '+args.mode+" --devicename "+device)
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
#bandwidthlist=[20, 15, 5, 15, 7, 10, 10, 20, 30, 7, 20, 15, 7, 5, 5, 30, 7, 15, 10, 7, 7, 7, 5, 25, 20, 7, 15, 30, 10, 20, 25, 5]
#bandwidthlist=[50,5,10,10,50,5,10,10,50,5,5,50,10,10,50]
#bandwidthlist=[50,20,10,10,50,20,10,10,50,7,7,50,10,10,50]
bandwidthlist=[50,20,10,20,7,10,50,20,50]
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





