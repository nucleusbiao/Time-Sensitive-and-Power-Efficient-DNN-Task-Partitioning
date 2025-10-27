#24/12/17  统计分割点期间的功率总和，总数，之后发送给client 端，

from communicationdeep import serverCommunication
import argparse
import subprocess
import time
import os
import threading
import minimalmodbus

def parse_args():
    # Parse input arguments
    desc = ''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--host', dest='host',
                        help='Ip address',
                        default='0.0.0.0', type=str)
    parser.add_argument('--ports', dest='ports',
                        help='Ip port',
                        default="8060", type=str)
    parser.add_argument('--dnnnum', dest='dnnnum',
                        help='dnns number',
                        default=6, type=int)
    args = parser.parse_args()
    return args

def e2_n(n):
    num=1
    for i in range(n):
        num=num*0.5
    return num

def tenTotwo(number):
    #定义栈
    s = []
    #binstring = ''
    while number > 0:
        #余数进栈
        rem = number % 2
        s.append(rem)
        number = number // 2
    while len(s)<16:
        s.append(0)
    s.reverse()
    #print(s)
    return s

def getpower():
    inst=minimalmodbus.Instrument("COM4",1)
    inst.serial.baudrate=9600
    inst.serial.timeout=1
    power=inst.read_registers(260,2,3)
    #print("power  ",str(power))
    bine=[]
    bine.extend(tenTotwo(power[0]))
    bine.extend(tenTotwo(power[1]))
   # print('bine ',bine)
    #print(len(bine))

    sum1=0
    for i in range(1,9):
        if bine[i]!=0:
            sum1+=(2**(8-i))
           # print(2**(8-i))
    #print("sum1 ",sum1)
    sum2=0
    for i in range(9,18):
        if bine[i]!=0:
            sum2+=(2**(8-i))
            #print(e2_n(i-8))
    sum2+=1
    #print("sum2 ",sum2)
    result=2**(sum1-127)*sum2
    # print(result)

    return result

def commuincatethread(host,port,threadnumber):
    global powersum_list
    global powercount_list
    global power_list
    global ifreadyflag
    global steplist
    # powerpath="power"+str(threadnumber)+".txt"
    communication = serverCommunication(host, port)
    conn, addr = communication.accept_conn()
    print("已连接",port)
    lock=threading.Lock()

    while True:        
        recv_data = communication.receive_msg(conn)
        if recv_data=="start":
            lock.acquire()   #对应的powerlist开始
            ifreadyflag[threadnumber]=True
            lock.release()
        elif recv_data=="end":  #统计，发送，清除置0，
            lock.acquire()  #把对应的powerlist开始
            ifreadyflag[threadnumber]=False
            powersum=powersum_list[threadnumber]
            powercount=powercount_list[threadnumber]
            powersum_list[threadnumber]=0
            powercount_list[threadnumber]=0
            lock.release()
            powerave=0
            if powercount_list[threadnumber]==0:
                powerave=powersum/powercount
            
            # if steplist[threadnumber]<50:
            #     filepath="latency/power"+"thread"+str(threadnumber)+"step"+str(steplist[threadnumber])+".txt"
            #     with open(filepath,"w") as f:
            #         for power in power_list[threadnumber]:
            #             f.write(str(power))
            #             f.write("\n")
            #     power_list[i]=[]
                # steplist[threadnumber]+=1
            print(powerave)
            communication.send_msg(conn, [powersum,powercount])
        elif  recv_data=="break":
            conn.close_channel()
            lock.acquire()  #把对应的powerlist开始
            ifreadyflag[threadnumber]=False
            lock.release()

def mastercommunication(host_m, port_m):

    communication_m = serverCommunication(host_m,port_m)
    conn_m, addr_m = communication_m.accept_conn()
    print("已连接master",port_m)
    while True:
        port,netnumber = communication_m.receive_msg(conn_m)  #[port,netnumber]
        #启动通信线程
        t=threading.Thread(target=commuincatethread,args=(args.host,int(port),netnumber))
        t.setDaemon(True)
        t.start()
        print("通信线程启动 port: ",port,"net ",netnumber)
        
args = parse_args()
print(args)
#portlist=args.ports.split(",")
# portlist=[args.ports+i for i in range(args.dnnnum)]
powersum_list=[0 for i in range(args.dnnnum)]
powercount_list=[0 for i in range(args.dnnnum)]
power_list=[[] for i in range(args.dnnnum)]
ifreadyflag=[False for i in range(args.dnnnum)]
steplist=[0 for i in range(args.dnnnum)]
print("hi")

if __name__ =="__main__":
    #master tcp
    port_m=8060
    t=threading.Thread(target=mastercommunication,args=(args.host,port_m))
    t.setDaemon(True)
    t.start()

    lock=threading.Lock()
    
    while True:
        power=getpower()
        lock=threading.Lock()
        for i in range(args.dnnnum):
            if ifreadyflag[i]==True:
                lock.acquire()
                powersum_list[i]+=power
                powercount_list[i]+=1
                # if steplist[i]<=60:
                #     power_list[i].append(power)
                lock.release()
        time.sleep(0.01)
        #print(power)

