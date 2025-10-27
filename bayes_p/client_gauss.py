#************client infer 部分**************#
# import sys

import cv2
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
import time
import pickle
import json
import os
import argparse

import sys
from pathlib import Path

this_dir = str(Path(__file__).resolve().parent)
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from models.vgg16 import vgg16
from models.tiny_yolo import tinyYolo
from models.alexnet import alexnet
from models.resnet import resnet50
from models.mobilenetv2 import mobilenetv2
from communicationdeep import clientCommunication
from yolo_utils import load_class_names, get_boxes, plot_boxes_cv2
import psutil
import threading
WINDOW_NAME = 'CameraDemo'

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

def prepare_image_googlenet(frame):
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img=data_transform(frame)
    img = torch.unsqueeze(img, dim=0)
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

def getActualDelay(action, model, preprocessed_image, totallayerNo, communication,device):
    if action == totallayerNo - 1: # local mobile process
        start_t=time.time()
        prediction = model(preprocessed_image.to(device))
        end_t=time.time()
        return 0, [prediction.data,0],end_t-start_t,0
    else:
        start_t=time.time()
        intermediate_output = model(preprocessed_image.to(device), server=False, partition=action)
        end_t=time.time()

    data_to_server = [action, intermediate_output.data]
    del intermediate_output

    start_time = time.time()
    communication.send_msg(data_to_server)
    result= communication.receive_msg()
    end_time = time.time()

    return end_time - start_time,  result, end_t-start_t, start_time

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    

def show_preds(img, label, averageTime):
    x = 10
    y = 50

    font = cv2.FONT_HERSHEY_PLAIN

    pred = '{:20s}'.format(label[1])
    cv2.putText(img, pred, (x, y), font, 2, (0, 0, 240), 2, cv2.LINE_AA)
    y += 30
    timeShow = 'AvgTime: {:.4f}'.format(averageTime)
    cv2.putText(img, timeShow, (x, y), font, 2, (0, 0, 240), 2, cv2.LINE_AA)

    return img

def open_window(width, height):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, 'Camera Demo ucbp0')

#每0.05s获取当前进程的内存占用率。
def vmem():
    global memflag
    global mempercentsum
    global mempercentcount
    global step
    lock=threading.Lock()
    while True:
        if memflag==True:
            mem = psutil.virtual_memory()
            mempercent=mem.percent
            lock.acquire()
            mempercentsum+=mempercent
            mempercentcount+=1
            lock.release()
        time.sleep(0.05)


def clientinfer(dnn_model,host,port,hostp,portp,breaktime,netnumber):
    # global partitionpointlist
    global cpuflag
    global thispid
    global cpuusum
    global cpuucounter
    global memflag
    global mempercentsum
    global mempercentcount
    global step

    device="cpu"
    print("using {} device.".format(device))

    if dnn_model == 'vgg':
        model = vgg16()
        model.eval()
        labels = getVggLabelDic(str(parent_dir)+'/models/imagenet_class_index.json')
        acttotal=22
    elif dnn_model == 'yolo':
        model = tinyYolo()
        model.eval()
        labels = load_class_names(str(parent_dir)+'/models/voc.names')
        acttotal=31
    elif dnn_model == 'alexnet':
        model=alexnet(num_class=5)
        model.eval()
        with open(str(parent_dir)+"/models/alexnetclass_indices.json", "r") as read_file:
            class_idx = json.load(read_file)
        acttotal=11
    elif dnn_model == 'mobilenetv2':
        model=mobilenetv2()
        model.eval()
        with open(str(parent_dir)+"/models/mobile_class_indices.json", "r") as read_file:
            class_idx = json.load(read_file)
        acttotal=21
    elif dnn_model == 'resnet':
        # create model
        model = resnet50(num_classes=5)
        weights_path = str(parent_dir)+"/models/resNet50_e3_1_0.pth"
        # load model weights
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        # read class_indict
        json_path =str(parent_dir)+ '/models/class_indices.json'
        #assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
        with open(json_path, "r") as f:
            class_indict = json.load(f)
        acttotal=20
    else:
        print("please input the right net name")
        assert False

    model.to(device)
    Action_num=acttotal+1

    #与edge 通信
    communication = clientCommunication(host, port)
    communication.connect()

    total_time = 0
    total_frame_num = 0

    #w,延时记录文件
    filepath=str(this_dir)+"/latency/latency_actual"+str(netnumber)+".txt"
    myfile=open(filepath,'w')
    myfile.write(f'actual_delay_all{"":5s}ppoint{"":4s}actual_df{"":12s}actual_de{"":13s}totaltime_lastframe{"":13s}time{"":13s}timecounter')
   
    startime_all=time.time()
    totaltime_start=time.time()
    timecounter=0
    real_latency_offrame=0
    lock=threading.Lock()
    oldtime_cpu=time.time()
    # cap = cv2.VideoCapture(0)
    # if not cap.isOpened():
    #     print("无法打开视频")
    #     exit()
    # ret, img = cap.read()
    # height, width, channels = img.shape
    # print(height,width)
    # open_window(800, 800)
    # framec=0
    average_time=0
    print("every  frame")
    while True:
     try:
        totaltime=time.time()-totaltime_start
        totaltime_start=time.time()
        partitionPoint=communication.receive_msg()
        
        #内存开启
        lock.acquire() 
        memflag=True #
        lock.release()
        
        img=cv2.imread(str(parent_dir)+'/models/tulip.jpg')
        # ret, img = cap.read()
        # if not ret:
        #     break
     
        if dnn_model == 'vgg':
            preprocessed_image = prepare_image_vgg(img)
        elif dnn_model == 'yolo':
            preprocessed_image = prepare_image_yolo(img)
        elif dnn_model == 'alexnet':
            preprocessed_image = prepare_image_alexnet(img)
        elif dnn_model == 'mobilenetv2':
            preprocessed_image = prepare_image_mobilenetv2(img)
        elif dnn_model=="googlenet":
            img = Image.open('models/tulip.jpg')
            preprocessed_image = prepare_image_googlenet(img)
        else:
            preprocessed_image = prepare_image_resnet(img)
        
        end2endtime_start = time.time()
        
        actual_delay, res, actual_df ,start_time= getActualDelay(partitionPoint, model, preprocessed_image, Action_num, communication,device)

        end2endtime_end = time.time()

        #w 如果分割点为最后分割点,给服务器发送数据，服务器记录。
        if partitionPoint==acttotal:
            communication.send_msg([partitionPoint,0])

        if total_frame_num==0:
            print("已完成第一帧",netnumber)
        
        #给edge发送实际延时数据
        real_latency_offrame=end2endtime_end-end2endtime_start
        real_client=actual_df
        real_trans=actual_delay-res[1]
        real_edgeinfer=res[1]

        #w增加，记录延时
        myfile.write("\n")
        myfile.write(f'{str(end2endtime_end-end2endtime_start):<21}')
        myfile.write(f'{str(partitionPoint):<5}')
        myfile.write(f'{str(actual_df):<23}')
        myfile.write(f'{str(actual_delay):<23}')
        myfile.write(f'{str(totaltime):<23}')                #上一帧的总时间
        myfile.write(f'{str(time.time()-startime_all):<23}{str(timecounter):<23}')

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

        #获取cpu利用率
        cputotal=psutil.cpu_percent(interval=None)
        #增加计时
        cpu_timelenth=time.time()-oldtime_cpu        
        oldtime_cpu=time.time()
        cpuutotalsum=cputotal*cpu_timelenth
        
        #获取内存占用率，停止内存统计
        lock.acquire() 
        memflag=False
        if mempercentcount!=0:
            memaver=mempercentsum/mempercentcount
        else:
            memaver=0
        mempercentsum=0
        mempercentcount=0
        lock.release()

        ladata=[real_latency_offrame,partitionPoint,real_client,real_trans,real_edgeinfer,cpuutotalsum,cpu_timelenth,cputotal,memaver]  ##一帧总延时，分割点，设备延时，传输延时，边缘延时，cpuu,cputime,cputotal,memaver
        communication.send_msg(ladata)  #
        step+=1

        res=res[0]
        if dnn_model == 'vgg':
            label = decodePrediction_vgg(res, labels)
            img = show_preds(img, label, average_time)
        elif dnn_model == 'yolo':
            boxes = get_boxes(device,res, model, conf_thresh=0.5, nms_thresh=0.5)
            img = plot_boxes_cv2(img, boxes, class_names=labels)
        elif dnn_model == 'alexnet':
            output=torch.squeeze(res).cpu()  #???
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            print_res = "alex class: {}   prob: {:.3}".format(class_idx[str(predict_cla)],
                                                 predict[predict_cla].numpy())
            label=[0,print_res]
            img = show_preds(img,label,average_time)
        elif dnn_model == 'mobilenetv2':
            output=torch.squeeze(res).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            print_res = "mobile class: {}   prob: {:.3}".format(class_idx[str(predict_cla)],
                                                 predict[predict_cla].numpy())
            label=[0,print_res]
            img = show_preds(img,label,average_time)
        else:
            res = torch.squeeze(res).cpu()
            predict = torch.softmax(res, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
            label=[0,print_res]
            img = show_preds(img,label,average_time)
            
        key = cv2.waitKey(10)
        if key == 27: # ESC key: quit program
            break
        elif key == ord('H') or key == ord('h'): # toggle help message
            show_help = not show_help
        elif key == ord('F') or key == ord('f'): # toggle fullscreen
            full_scrn = not full_scrn
            if full_scrn:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)

     except KeyboardInterrupt or TypeError or OSError:
            #发送结束
            communication.send_msg("close")
            communication.close_channel()
            myfile.close()
            print(netnumber," ",dnn_model,' average_time:',average_time)
            del model
            # 释放摄像头资源
            # cap.release()
            # 关闭所有窗口
            cv2.destroyAllWindows()
                
    #发送结束
    communication.send_msg("close")
    communication.close_channel()
    myfile.close()
    print(netnumber," ",dnn_model,' average_time:',average_time)
     # 释放摄像头资源
    # cap.release()
    # 关闭所有窗口
    cv2.destroyAllWindows()
    del model

def parse_argse():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dnn_model',dest='dnn_model',help="dnn_model name",default='vgg',type=str)
    parser.add_argument('--host',dest='host',help="server host",default='192.168.31.33',type=str)
    parser.add_argument('--port',dest="port",help="sever port",default=8080,type=int)
    parser.add_argument('--hostp',dest='hostp',help="power host",default='192.168.31.160',type=str)
    parser.add_argument('--portp',dest="portp",help="power port",default=8060,type=int)
    parser.add_argument('--breaktime',dest="breaktime",default=50,type=int)
    parser.add_argument('--netnumber',dest="netnumber",default=0,type=int)
    args=parser.parse_args()
    return args

cpuflag=False
cpuusum=0
cpuucounter=0
memflag=False
mempercentsum=0
mempercentcount=0
thispid=os.getpid()
step=0
print("main pid",thispid)

if __name__ == "__main__":
    arg=parse_argse()
    t=threading.Thread(target=vmem)
    t.daemon=True
    t.start()
    clientinfer(dnn_model=arg.dnn_model,host=arg.host,port=arg.port,hostp=arg.hostp,portp=arg.portp,breaktime=arg.breaktime,netnumber=arg.netnumber)