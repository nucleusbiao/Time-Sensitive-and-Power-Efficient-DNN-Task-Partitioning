#24/9/4
#client 控制开启，停止，带宽。
#定时开启，先向edge 发送启动请求，收到确认信息后启动任务子线程。
#9/9 按照device 设置任务。

#************client infer 部分**************#

from pathlib import Path
import sys
parent_dir = str(Path(__file__).resolve().parent.parent)
print(parent_dir)
sys.path.append(parent_dir)
parent_parent_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(parent_parent_dir)

import threading
import argparse
import cv2
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
import time
import pickle
import json
# import random
import os
# import socket

from models.vgg16 import vgg16
from models.tiny_yolo import tinyYolo
from models.alexnet import alexnet
from models.resnet import resnet50
from models.mobilenetv2 import mobilenetv2
from models.mobileformer import MobileFormer
from utils.config import config_294
#from keyFrameDetection import KeyFrameDetection
from communicationdeep import clientCommunication
#from up_all2 import ql
from yolo_utils import load_class_names, get_boxes, plot_boxes_cv2



def prepare_image_vgg(frame):
    min_img_size = 224
    transform_pipeline = transforms.Compose([transforms.Resize((min_img_size, min_img_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    img = transform_pipeline(img)
    img = img.unsqueeze(0)
    return img

def prepare_image_yolo(frame):
    min_img_size = 416
    image = cv2.resize(frame, (min_img_size, min_img_size), interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    img = torch.from_numpy(image.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    return img

def prepare_image_alexnet(frame):
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    img=data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    return img

def prepare_image_mobilenetv2(frame):
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    img=data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    return img

def prepare_image_resnet(frame):
    transform_pipeline = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    img = transform_pipeline(img)
    img = img.unsqueeze(0)
    return img

def prepare_image_mobilenetformer(frame):
    transform_pipeline = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    img = transform_pipeline(img)
    img = img.unsqueeze(0)
    return img

def getVggLabelDic(class_file):
    with open(class_file, "r") as read_file:
        class_idx = json.load(read_file)
        labels = {int(key): value for key, value in class_idx.items()}
    return labels

def decodePrediction_vgg(res, labels):
    res = torch.autograd.Variable(res)
    label_index = torch.argmax(res).item()
    return labels[label_index]

def getActualDelay(action, model, preprocessed_image, totallayerNo, communication,device,dnn_model):
    if action == totallayerNo - 1: # local mobile process
        start_t=time.time()
        if dnn_model=='mobileformer':
            prediction = model(preprocessed_image.to(device),0,0,False,action)
            prediction=prediction[0]
        else:    
            prediction = model(preprocessed_image.to(device))
        prediction=prediction.data
        end_t=time.time()
        return 0, [prediction,0],end_t-start_t,0
    else:
        start_t=time.time()
        if dnn_model=='mobileformer':
            intermediate_output = model(preprocessed_image.to(device),0,0, server=False, partition=action)
        else:    
            intermediate_output = model(preprocessed_image.to(device), server=False, partition=action)
            intermediate_output=intermediate_output.data
        end_t=time.time()

    data_to_server = [action, intermediate_output]
    del intermediate_output

    start_time = time.time()
    communication.send_msg(data_to_server)
    result= communication.receive_msg()
    end_time = time.time()

    return end_time - start_time,  result, end_t-start_t, start_time

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def clientinfer(dnn_model,host,port,breaktime,netnumber):
    # global partitionpointlist

    device="cpu"
    print("using {} device.".format(device))

    if dnn_model == 'vgg':
        model = vgg16()
        model.eval()
        labels = getVggLabelDic(str(parent_parent_dir)+'/models/imagenet_class_index.json')
        acttotal=22
    elif dnn_model == 'yolo':
        model = tinyYolo()
        model.eval()
        labels = load_class_names(str(parent_parent_dir)+'/models/voc.names')
        acttotal=31
    elif dnn_model == 'alexnet':
        model=alexnet(num_class=5)
        model.eval()
        with open(str(parent_parent_dir)+"/models/alexnetclass_indices.json", "r") as read_file:
            class_idx = json.load(read_file)
        acttotal=11
    elif dnn_model == 'mobilenetv2':
        model=mobilenetv2()
        model.eval()
        with open(str(parent_parent_dir)+"/models/mobile_class_indices.json", "r") as read_file:
            class_idx = json.load(read_file)
        acttotal=21
    elif dnn_model == 'resnet':
        # create model
        model = resnet50(num_classes=5)
        weights_path = str(parent_parent_dir)+"/models/resNet50_e3_1_0.pth"
        # load model weights
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        # read class_indict
        json_path =str(parent_parent_dir)+ '/models/class_indices.json'
        #assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
        with open(json_path, "r") as f:
            class_indict = json.load(f)
        acttotal=20
    elif dnn_model=='mobileformer':
        model=MobileFormer(config_294)
        weights_path=str(parent_parent_dir)+'/models/mobileformer.pth.tar'
        if os.path.exists(weights_path) == False:
            weights_path = "../models/mobileformer.pth.tar"
        # load model weights
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device)['state_dict'])
        model.eval()
        acttotal=18
    else:
        print("please input the right net name")
        assert False

    #model.cuda()
    model.to(device)
    Action_num=acttotal+1

    #与edge 通信
    communication = clientCommunication(host, port)
    communication.connect()

    total_time = 0
    total_frame_num = 0

    filepath=str(parent_dir)+"/latency/latency_deeppred"+str(netnumber)+".txt"
    myfile=open(filepath,'w')
    myfile.write(f'actual_delay_all{"":5s}ppoint{"":4s}actual_df{"":12s}actual_de{"":13s}totaltime_lastframe{"":13s}')
   
    startime_all=time.time()
    totaltime_start=time.time()
    timecounter=0
    real_latency_offrame=0
    lock=threading.Lock()
    while True:
     try:
        totaltime=time.time()-totaltime_start
        totaltime_start=time.time()
        # partitionPoint= deep_s.receive_msg()
        partitionPoint=communication.receive_msg()

        img=cv2.imread(str(parent_parent_dir)+'/models/tulip.jpg')
        
        if dnn_model == 'vgg':
            preprocessed_image = prepare_image_vgg(img)
        elif dnn_model == 'yolo':
            preprocessed_image = prepare_image_yolo(img)
        elif dnn_model == 'alexnet':
            preprocessed_image = prepare_image_alexnet(img)
        elif dnn_model == 'mobilenetv2':
            preprocessed_image = prepare_image_mobilenetv2(img)
        elif dnn_model=='mobileformer':
            preprocessed_image = prepare_image_mobilenetformer(img)
        else:
            preprocessed_image = prepare_image_resnet(img)
        
        end2endtime_start = time.time()
        
        actual_delay, res, actual_df ,start_time= getActualDelay(partitionPoint, model, preprocessed_image, Action_num, communication,device,dnn_model)

        end2endtime_end = time.time()

        #w 如果分割点为最后分割点,给服务器发送数据，服务器记录。
        if partitionPoint==acttotal:
            communication.send_msg([partitionPoint,0])
            #communication.close_channel()

        if total_frame_num==0:
            print("已完成第一帧",netnumber)
        
        #给edge发送实际延时数据
        real_latency_offrame=end2endtime_end-end2endtime_start
        real_client=actual_df
        real_trans=actual_delay-res[1]
        real_edgeinfer=res[1]
        ladata=[real_latency_offrame,partitionPoint,real_client,real_trans,real_edgeinfer]
        communication.send_msg(ladata)  #
        #deep_s.send_msg((real_latency_offrame,partitionPoint))

        ###唤醒等待
        #event.set()

        #w增加，记录延时
        myfile.write("\n")
        myfile.write(f'{str(end2endtime_end-end2endtime_start):<21}')
        myfile.write(f'{str(partitionPoint):<5}')
        myfile.write(f'{str(actual_df):<23}')
        myfile.write(f'{str(actual_delay):<23}')
        myfile.write(f'{str(totaltime):<23}\n')                #上一帧的总时间

        total_frame_num = total_frame_num + 1
        if total_frame_num>1:
            total_time = total_time + (end2endtime_end - end2endtime_start)
            average_time = total_time/(total_frame_num-1)
             
        ##定时 1min    50max
        if total_frame_num==1:
            print("已完成第一帧",netnumber)
        if(time.time()-startime_all>(timecounter+1)*60):
            timecounter+=1
            print(timecounter,'min  total_frame_num: ',total_frame_num,"netnumber: ",netnumber)
            if(breaktime==timecounter):
                print('break at ',timecounter,'min net ',netnumber," ",dnn_model)
                break

     except KeyboardInterrupt or TypeError or OSError:
            #发送结束
            communication.send_msg("close")
            communication.close_channel()
            myfile.close()
            print(netnumber," ",dnn_model,' average_time:',average_time)
            del model
                

    #发送结束
    communication.send_msg("close")
    communication.close_channel()
    myfile.close()
    print(netnumber," ",dnn_model,' average_time:',average_time)
    del model


def parse_argse():
    parser=argparse.ArgumentParser()
    parser.add_argument('--host',dest='host',help="server host",default='192.168.31.33',type=str)
    parser.add_argument('--bandrand',dest="bandrand",action='store_true',help="if add this , bandrand.")
    parser.add_argument('--port',dest="port",help="sever port",default=8080,type=int)
    parser.add_argument('--devicename',dest="devicename",default='k',type=str)
    args=parser.parse_args()
    return args

def deltrickled():
    global passward
    pid=''
    killarg=''
    os.system('ps -ef | grep trickled | grep -v grep > trickled.txt')
    with open('trickled.txt','r') as tf:
        line=tf.readline().split()
        # print(line)
        if(len(line)>1):
            pid=line[1]
            killarg='echo '+passward+' | sudo -S kill '+pid
    os.system(killarg)

args=parse_argse()
print(args)
##》》》
#配置任务和开始时间，持续时间，端口！！！
if args.devicename=='k':
    dnnlist=["yolo","resnet"]
    sarttimelist=[0,8]
    timelength=[12,4]
    portlist=[9050,9051,9052,9053]
elif args.devicename=='u':
    dnnlist=["alexnet"]
    sarttimelist=[6]
    timelength=[6]
    portlist=[9060,9061,9062,9063]
elif args.devicename=='tx2':
    dnnlist=["mobilenetv2"]
    sarttimelist=[2]
    timelength=[10]
    portlist=[9070,9071,9072,9073]
elif args.devicename=='ly':   
    dnnlist=["vgg","mobileformer"]
    sarttimelist=[4,10]
    timelength=[8,2]
    portlist=[9080,9081,9082,9083]
else:
    print("please input the right device name")
##《《《
##《《《
passwarddic={'k':"6","u":"6","tx2":"6","ly":"123"}  #用于设置带宽
passward=passwarddic[args.devicename]

starttime_all=time.time()
timecounter=-1
timestart=starttime_all

deltrickled()
#连接服务器server
communication = clientCommunication(args.host, args.port)
communication.connect()
print("has connected edge")

startflag=1
while True:
    try:
        time.sleep(0.01)
        timenow=time.time()
        #定时启动任务
        if timenow-timestart>=(timecounter+1)*60:
            timecounter+=1
            for i in range(len(dnnlist)):
                if(sarttimelist[i]==timecounter):
                    communication.send_msg([dnnlist[i],portlist[i]])  #发送启动任务信息
                    communication.receive_msg()  #server确认信息  待：设置端口
                    t=threading.Thread(target=clientinfer,args=(dnnlist[i],args.host,portlist[i],timelength[i],i)) #启动任务线程
                    t.daemon=True
                    t.start()

            ##带宽控制
            if args.bandrand:
                bandwidthlist=[50,5,5,50,10,10,5,5,50,10,5]
                if timecounter<len(bandwidthlist):
                    bandwidth=bandwidthlist[timecounter]
                    #删除已有设置
                    deltrickled()
                    #设置新的
                    strwidths="trickled"+ " -d "+ str(bandwidth)+" -u "+ str(bandwidth)
                    res_ch=os.system(strwidths)
                    print("res_ch",res_ch, timecounter,' min  bandwidth is set to ',bandwidth)
                else:
                    print(str(timecounter),'min  total_frame_num: ')
                    deltrickled()
                    print("bandlist end now")
                    if (timecounter>len(bandwidthlist)):
                        break
    except KeyboardInterrupt or TypeError or OSError:
          deltrickled() 
          break
    
















