import argparse
import torch
import time
import os
import json
import sys
from pathlib import Path

this_dir = str(Path(__file__).resolve().parent)
sys.path.append(this_dir)
parent_dir = str(Path(__file__).resolve().parent.parent)
print(parent_dir)
sys.path.append(parent_dir)
parent_parent_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(parent_parent_dir)

from models.vgg16 import vgg16
from models.tiny_yolo import tinyYolo
from models.mobilenetv2 import mobilenetv2
from models.resnet import resnet50
from models.alexnet import alexnet
from models.mobileformer import MobileFormer
from utils.config import config_294
from communication import serverCommunication

def parse_args():
    # Parse input arguments
    desc = 'ANS in edge server side'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dnn', dest='dnn_model',
                        help='vgg, yolo,resnet,alexnet,mobilenetv2',
                        default='yolo', type=str)
    parser.add_argument('--host', dest='host',
                        help='Ip address',
                        default='0.0.0.0', type=str)
    parser.add_argument('--port', dest='port',
                        help='Ip port',
                        default=8889, type=int)
    parser.add_argument('--totalnumber', dest='totalnumber',
                        help='totalnumber of all tasks all devices',
                        default=0, type=int)    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    print("read to connect")
    communication = serverCommunication(args.host, args.port)
    conn, addr = communication.accept_conn()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device="cpu"
    print("using {} device.".format(device))
    time1=time.time()

    if args.dnn_model == 'vgg':
        model = vgg16()
        model.eval()
        acttotal=22
    elif args.dnn_model == 'yolo':
        model = tinyYolo()
        model.eval()
        acttotal=31
    elif args.dnn_model == 'alexnet':
        model=alexnet()
        model.eval()
        acttotal=11
    elif args.dnn_model == 'mobilenetv2':
        model=mobilenetv2()
        model.eval()
        acttotal=21
    elif args.dnn_model == 'resnet':
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
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
        with open(json_path, "r") as f:
            class_indict = json.load(f)
        acttotal=20
    elif args.dnn_model=='mobileformer':
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

    model.to(device)
    print('模型加载时间： ',time.time()-time1)
    filepath=str(parent_dir)+"/latency/latency_server"+str(args.totalnumber)+args.dnn_model+".txt"
    myfile=open(filepath,'w')
    myfile.write("receive_time              infer_endtime     partition_point      sendendtime")

    while True:
        try:
                recv_data = communication.receive_msg(conn)
                if(recv_data=="close"):
                    break
                receive_time=time.time()
                partition_point = recv_data[0]

                if partition_point!=acttotal:
                    data = recv_data[1]

                    if args.dnn_model=='mobileformer':
                        if type(data[1])!=int:
                            zd=data[1].to(device)
                        prediction = model(x=data[0].to(device),z=zd,b=data[2],server=True, partition=partition_point)
                    else:
                        prediction = model(data.to(device), server=True, partition=partition_point)
                    res = prediction.data
        
                    res=res.to('cpu')
                
                    infer_endtime=time.time()
                    msg = communication.send_msg(conn, res)

                send_endtime=time.time()
                myfile.write("\n")
                if(partition_point!=acttotal):
                    myfile.write(f'{receive_time:<22}{infer_endtime:<22}{partition_point:<8}{send_endtime}')
                else:
                    myfile.write(f'{receive_time:<22}{receive_time:<22}{partition_point:<8}{send_endtime}')

        except KeyboardInterrupt or TypeError or OSError:
            communication.close_channel()
            myfile.close()        
    communication.close_channel()
    myfile.close()
    print(str(args.dnn_model)," ",str(args.totalnumber),"closed")