'''
一个DNN不同分割策略的实施延时预测模型
输入；分割点，三个延时
输出：预测延时
'''

#**5/6 环境不变时，不做update  （定期mbatch？）**#
# loss 小于近十个的平均+0.01，update. 
# 在update 期间增加上置信区间采样 
# 

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import collections
import random

# -------------------------------------- #
# 构造深度学习网络，输入状态s，得到各个动作的reward
# -------------------------------------- #
    
# class Net(nn.Module):
    # 构造有两个隐含层的网络
    # def __init__(self, dnnsmun,n_states, n_hidden=49, n_actions=2):
    #     super(Net, self).__init__()
    #     self.conv=nn.Conv1d(dnnsnum,)
    #     # [b,n_states]-->[b,n_hidden]  
    #     self.fc1 = nn.Linear(n_states, n_hidden)
    #     self.relu = nn.ReLU(inplace=True)
    #     #self.sigmoind=nn.Sigmoid()
    #     # [b,n_hidden]-->[b,n_actions]
    #     self.dropout=nn.Dropout(0.2)
    #     self.fc2 = nn.Linear(n_hidden,n_actions)
        
    #     print(f'hidden ,{str(n_hidden)}')
    # # 前传
    # def forward(self, x):  # [b,n_states]
    #     x = self.fc1(x)
    #     x = self.relu(x)
    #     x=self.dropout(x)
    #     #x=self.sigmoind(x)
    #     x = self.fc2(x)
    #     return x
    
class Net(nn.Module):
    # 构造有两个隐含层的网络
    def __init__(self, n_states, n_hidden=49, n_actions=2):
        super(Net, self).__init__()
        # [b,n_states]-->[b,n_hidden]  
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.relu = nn.ReLU(inplace=True)
        #self.sigmoind=nn.Sigmoid()
        # [b,n_hidden]-->[b,n_actions]
        #self.dropout=nn.Dropout(0.2)
        self.fc2 = nn.Linear(n_hidden,n_actions)
        print(f'hidden ,{str(n_hidden)}')
    # 前传
    def forward(self, x):  # [b,n_states]
        x = self.fc1(x)
        x = self.relu(x)
       #x=self.dropout(x)
        #x=self.sigmoind(x)
        x = self.fc2(x)
        return x

# -------------------------------------- #
# 构造预测的class
# -------------------------------------- #

class Cnet:
    #（1）初始化
    def __init__(self, 
                learning_rate,gamma,historylength,arg,device,dnnname,dnnsnum,partitioncountdic,latencyrangedic_low,latencyrangedic_high,mainnumber):
        
        #静态点构建,
        staticlatencyofppdic={"vgg":[0.135632798075676,1.84624034352601,1.82540737651289,0.555241552181542,1.01635658368468,
                                     1.03722334932536,0.364841743372381,0.609673753380775,0.623371617868543,0.636040900833905,
                                     0.271314739249647,0.452634233050048,0.460248411633074,0.459349850192666,0.260149923153222,
                                     0.26392425596714,0.269658954814076,0.275311009958386,0.218898234888911,0.182884123176336,
                                     0.191422046162188,0.199528632685542,0.185149817727506],# k
                        "yolo":[0.975209126472473,3.33165528914508,3.31918281316757,3.15575832605362,1.20401803255081,
                                1.98736628157194,1.78801726549864,1.67308484978146,0.844536070823669,1.38075566053391,
                                1.45484951972961,1.33564312696457,0.737530047893524,0.913929894163801,0.844109957980126,
                                0.916474266052246,0.703921020847477,0.857848876788293,0.877179711364036,0.706304615941541,
                                0.506522953510285,0.605130341053009,0.59843005657196,0.751966223830268,0.586034793532297,
                                0.938976843357086,0.911076673820837,0.751153078423925,1.0224683157603,0.903021311759949,
                                0.959697326024374,0.522044194887762,],#u yolo
                        "resnet":[0.421245429515839,1.54236973524094,0.665017247200012,1.54467415332794,1.66013082398309,
                                    1.67457192420959,1.19947110652924,1.48656114339829,1.70620789766312,1.63542807579041,
                                    1.48332567691803,1.67235898017883,1.69887313365936,1.63155946493149,1.66102375030518,
                                    1.71582610518844,2.2432828950882,2.51759289741516,2.86122235298157,2.44945328235626,
                                    2.2533167952841],#k alexnnet +kresnet
                        "alexnet":[0.486023323535919,0.446764903068542,0.250829844474792,0.387708013057709,0.254409642219543,
                                    0.274282796382904,0.282006018161774,0.275214760303497,0.293011493682861,0.27665135383606,
                                    0.297576904296875,0.120680902004242,],# y alexnet
                        "mobilenetv2":[0.171874882811207,0.394284939361831,0.236144591186006,0.184331348386862,0.213312064187001,
                                       0.189752437300601,0.19931587930453,0.209445201744467,0.205893714549178,0.213539685233165,
                                       0.21839482905501,0.223283892970974,0.233700247134192,0.242624638444286,0.251128067404537,
                                       0.245401188478631,0.254981380397991,0.260633371644101,0.278705225152484,0.323719448962454,
                                       0.268273777880911,0.258745250055345,],#k
                        "mobileformer":[0.317853421,0.397558938,0.689390739,0.3667144,0.20720081,
                                        0.214694272,0.165889878,0.170992569,0.150447699,0.160039575,
                                        0.177053933,0.191731317,0.166422329,0.173334078,0.1757034,
                                        0.252001102,0.164389527,0.164578103,0.140798422,]
                                        } 
        
        #静态表构建  ~~~~~~~~~~~~~~~~~~~~~~~~~~~（待处理，用得着吗,应该用不着了）
        staticlatencyofpplist=staticlatencyofppdic[dnnname]
        # for i in range(dnnsnum):    #归一化
        maxl=max(staticlatencyofpplist)
        minl=min(staticlatencyofpplist)
        lengthl=maxl-minl
        datanum=len(staticlatencyofpplist)
        for j in range(datanum):
            staticlatencyofpplist[j]=(staticlatencyofpplist[j]-minl)/lengthl
        self.staticlatencyofpplist=staticlatencyofpplist[:]
        
        # 属性分配
        self.learning_rate = learning_rate  # 训练时的学习率
        self.historylength=historylength
        self.device=device
        # self.dnnsum=dnnsnum
        #self.train=arg.train
        self.train=True
        self.ifcontinue=False
        # self.dnnslist=dnnslist
        self.mainnumber=mainnumber  #任务编号
        self.latencyrangelist_low=latencyrangedic_low[dnnname]
        latencyrangelist_high=latencyrangedic_high[dnnname]
        self.latencyrangelist_length=latencyrangelist_high-self.latencyrangelist_low
        
        #归一化范围【0，1】
        # for i in range(dnnsnum):
        # self.latencyrangelist_length.append((latencyrangelist_high[i]-self.latencyrangelist_low[i]))
        if self.latencyrangelist_length==0:
            print("latencyrangelist_length ",str(i),"is zero")
            exit(1)
        print("归一化范围【0，1】")
       
        #改，归一化范围【-1，1】

        #self.dnnslist=dnnslist
        self.partitioncount=partitioncountdic[dnnname]

        #historypool 
        self.historypool=[]
        # for i in range(dnnsnum):
        for pp in range(self.partitioncount+1):
            self.historypool.append(collections.deque(maxlen=historylength+1))   #[h11,h12,h13,r1],[]  #作为经验池，数据回放。
                # for j in range(historylength+1):
                #     self.historypool[i][pp].append(0)
        
        # self.historylatency_datalist=[]
        # for _ in range(dnnsnum):
        self.historylatency_datalist=collections.deque(maxlen=historylength)  #最近的三次历史数据
        
        # 自主决策模式,加载之前的historypool
        #if arg.train==False:
        #if arg.ifcontinue:
        if   self.ifcontinue:
            import pickle #historypoolyolo+"+ mainnumber+ ".pkl"
            hispoolpath="models/historypoolyolo+"+ mainnumber+ ".pkl"
            with open(hispoolpath,"rb") as f:
                self.historypool=pickle.load(f)
                print("加载了之前的 ",hispoolpath)
            staticlatencyofpplistpath="models/staticlatencyofppdic+"+ mainnumber+ ".pkl"
            with open(staticlatencyofpplistpath,"rb") as f:
                self.staticlatencyofpplist=pickle.load(f)
                print("加载了之前的  " ,staticlatencyofpplistpath)
            
        self.net = Net(n_states=dnnsnum+dnnsnum*historylength,n_actions=dnnsnum)
        self.net.to(device)
        #if arg.ifcontinue:
        if self.ifcontinue:
            # dnnnames=''
            # for dnnname in dnnslist:
            #     dnnnames+=dnnname
            #net_dictpath='models/predictlatency_model'+mainnumber+dnnnames+'step'+str(arg.step)+'.pt'
            net_dictpath='models/predictlatency_model'+mainnumber+'.pt'
            #net_dictpath='models/predictlatency_model'+mainnumber+dnnnames+'step'+str(arg.step)+'.pt'
            self.net.load_state_dict(torch.load(net_dictpath,map_location='cpu'))
            print("已加载模型 ",net_dictpath)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        print('historylength ',historylength)
        print('learning_rate ',learning_rate)
        
        self.mbatchsize=self.partitioncount+1
        # # for dnn in range(dnnsnum):
        #     self.mbatchsize=self.mbatchsize*(self.partitioncountlist[dnn]+1)
        print("mbatchsize ",self.mbatchsize)

        self.plistu=[i for i in range(self.partitioncount+1)]
        print("bbatch: ",self.plistu)


        pppfile="latency/ppp"+str(self.mainnumber)+".txt"
        with open(pppfile,"w") as pppf:
            # for i in range(self.dnnsum):
            pppf.write(f'p predict  actual    step')
            pppf.write("\n")

    #（2）多batch训练
    def mbatch(self):
        loss_function = nn.MSELoss(reduction='mean')
        self.net.train()
        self.optimizer.zero_grad()
        #多batch 训练
        state=[[] for _ in range(self.mbatchsize)]
        real_latencynor=[[] for _ in range(self.mbatchsize)]

        # for dnn in range(self.dnnsum):
        for batch in range(self.mbatchsize):
            pp=self.plistu[batch]   
            state[batch].extend([(self.historypool[pp][i]-self.latencyrangelist_low)/self.latencyrangelist_length for i in range(self.historylength)])
            #state[batch].append(pp/self.partitioncountlist[dnn])
            #state[batch].append(self.staticlatencyofpplist[dnn][pp])  #改成静态标准化的延时  #该点的延时,上一次的??? or 静态?  or 平均? 
            state[batch].append((self.staticlatencyofpplist[pp]-self.latencyrangelist_low)/self.latencyrangelist_length)
            real_latencynor[batch].append((self.historypool[pp][-1]-self.latencyrangelist_low)/self.latencyrangelist_length)
        real_latencynor= torch.tensor(real_latencynor).view(self.mbatchsize,-1).to(self.device)
        
        state1=torch.tensor(state)
        outputs1 =self.net(state1.to(self.device))
        loss1 = loss_function(outputs1, real_latencynor)
        loss1.backward()
        self.optimizer.step()

        # return (loss1.item()+loss2.item()+loss3.item())/2
        return loss1.item()
    
    def predict_latency(self,partitionpint):
        self.net.eval()
        ###当前分割点预测，训练
        state=[]
        # for dnn in range(self.dnnsum):
        state+=([(self.historylatency_datalist[i]-self.latencyrangelist_low)/self.latencyrangelist_length for i in range(self.historylength)])
            # state.append(partitionpint[dnn]/self.partitioncountlist[dnn])  #
            #state.append(self.staticlatencyofpplist[dnn][partitionpint[dnn]])  #改成静态标准化的延时
        state.append((self.staticlatencyofpplist[partitionpint]-self.latencyrangelist_low)/self.latencyrangelist_length)
        state=torch.tensor(state).view(1,-1)
        outputs =self.net(state.to(self.device))
        #分割点延时逆归一化
        # for i in range(self.dnnsum):
        outputs[0][0]=outputs[0][0]*self.latencyrangelist_length+self.latencyrangelist_low
        return outputs
    
    #（3）网络训练
    def update(self, real_latency,partitionpint,ifmbatch,averloss): 
        #loss_function = nn.L1Loss(reduction='mean')
        loss_function = nn.MSELoss(reduction='mean')
        self.net.train()
        self.optimizer.zero_grad()
        state=[]
        ###当前分割点预测，训练
        # for dnn in range(self.dnnsum):
        state+=([(self.historylatency_datalist[i]-self.latencyrangelist_low)/self.latencyrangelist_length for i in range(self.historylength)])
        # state.append(partitionpint[dnn]/self.partitioncountlist[dnn])
        #state.append(self.staticlatencyofpplist[dnn][partitionpint[dnn]])  #改成静态标准化的延时
        state.append((self.staticlatencyofpplist[partitionpint]-self.latencyrangelist_low)/self.latencyrangelist_length)  #改成静态标准化的延时
        state=torch.tensor(state).view(1,-1)
        outputs =self.net(state.to(self.device))
        real_latencynor=[]
        #for i in range(self.dnnsum):  #real_latency 归一化
        real_latencynor.append((real_latency-self.latencyrangelist_low)/self.latencyrangelist_length)
        real_latencynor= torch.tensor(real_latencynor).view(1,-1).to(self.device)
        loss = loss_function(outputs, real_latencynor)

        ifupdatenet=False
        #是否update 判断： ~~~!!!!限制到训练好之后啊！！
        if loss>averloss+0.01 or ifmbatch:
            loss.backward()
            self.optimizer.step()
            ifupdatenet=True

        #for i in range(self.dnnsum):  #预测延时逆归一化
        outputs[0][0]=outputs[0][0]*self.latencyrangelist_length+self.latencyrangelist_low
      
        ##多batch训练
        if ifmbatch:
            loss_mbatch=self.mbatch()    
        else : 
            loss_mbatch=0

        return loss.item(),outputs,loss_mbatch,ifupdatenet

    def ppptest(self,step):
        # print("plistu",self.plistu)
        pppfile="latency/ppp"+str(self.mainnumber)+".txt"
        with open(pppfile,"a") as pppf:
            # for i in range(self.dnnsnum):
            #     pppf.write(f'p{str(i)} predict{str(i)}  actual{str(i)}   step')
            # pppf.write("\n")
            for batch in range(self.mbatchsize):
                state=[]
                real_latency=[]
                # hhhplist=[11,31,20]
                # for dnn in range(self.dnnsum):
                # print("dnn",dnn)
                # print("batch",batch)
                pp=self.plistu[batch]   
                #****#   historypool
                state.extend([(self.historypool[pp][i]-self.latencyrangelist_low)/self.latencyrangelist_length for i in range(self.historylength)])
                #****#  固定
                #state.extend([(self.historylatency_datalist[dnn][i]-self.latencyrangelist_low[dnn])/self.latencyrangelist_length[dnn] for i in range(self.historylength)])
                #****#    histrypool 里的一组
                #state.extend([(self.historypool[dnn][hhhplist[dnn]][i]-self.latencyrangelist_low[dnn])/self.latencyrangelist_length[dnn] for i in range(self.historylength)])
                state.append((self.staticlatencyofpplist[pp]-self.latencyrangelist_low)/self.latencyrangelist_length)  #改成静态标准化的延时  #该点的延时,上一次的??? or 静态?  or 平均? 
                real_latency.append(self.historypool[pp][-1])

                state=torch.tensor(state).view(1,-1)
                outputs =self.net(state.to(self.device))
                # for i in range(self.dnnsum):  #预测延时逆归一化
                outputs[0][0]=outputs[0][0]*self.latencyrangelist_length+self.latencyrangelist_low
                # for i in range(self.dnnsum):
                pppf.write(f'{str(self.plistu[batch])}   {str(outputs[0][0].item())} {str(real_latency[0])}    ')
                pppf.write(str(step))    
                pppf.write("\n")

import argparse
def parse_args():
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnns', dest='dnns',
                        help='vgg, yolo,resnet,alexnet,mobilenetv2',
                        # default='yolo,resnet,alexnet', type=str)
                        default='alexnet,yolo,resnet', type=str)
    parser.add_argument('--continue', dest='ifcontinue',
                        help='contiune to train',
                        action='store_false')  #store_true  store_false
    parser.add_argument('--train', dest='train',
                        help='contiune to train',
                        action='store_false')  #store_true  store_false
    parser.add_argument('--step', dest='step',
                        help='continue from this step',
                        default=7001, type=int)
    parser.add_argument('--mainnumber', dest='mainnumber',
                        help='mainnumber',
                        default="", type=str)
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args=parse_args()
    print(args)
    dnnslist=args.dnns.split(",")
    dnnsnum=len(dnnslist)
    print(dnnslist)

    lr = 1e-2
    #lr = 2e-3  # 学习率
    gamma = 0.9  # 折扣因子
    historylen=3 #历史数据个数

    #mainnumber=""
    
    device='cpu'
   
    latencyrangedic_high={"vgg":2.3733,  "yolo":1.907, "resnet":2.144,"alexnet": 0.1988, "mobilenetv2":0.399}
    latencyrangedic_low={"vgg":0.3171,  "yolo":0.322, "resnet":0.107, "alexnet":0.036, "mobilenetv2":0.178}
    #vgg, yolo,resnet,alexnet,mobilenetv2 的分割点数量
    partitioncountdict={"vgg":22,"yolo": 31,"resnet":20,"alexnet":11,"mobilenetv2":21}

    CNet=Cnet(learning_rate=lr,gamma=gamma,historylength=historylen,arg=args,device=device,dnnslist=dnnslist,dnnsnum=dnnsnum,partitioncountdic=partitioncountdict,latencyrangedic_low=latencyrangedic_low,latencyrangedic_high=latencyrangedic_high,mainnumber=args.mainnumber)
    
    import pickle
    historypoolpath="models/historypoolyolo+"+args.mainnumber+".pkl"
    with open(historypoolpath,"rb") as f:
        CNet.historypool=pickle.load(f)
        print("加载了之前的  " ,historypoolpath)
    staticlatencyofpplistpath="models/train/staticlatencyofppdic+"+ args.mainnumber+ ".pkl"
    with open(staticlatencyofpplistpath,"rb") as f:
        CNet.staticlatencyofpplist=pickle.load(f)
        print("加载了之前的  " ,staticlatencyofpplistpath)
        
    # CNet.historylatency_datalist[0].append(0.449026584625244)
    # CNet.historylatency_datalist[0].append(0.87877631187439)
    # CNet.historylatency_datalist[0].append(0.87877631187439)
    # CNet.historylatency_datalist[0].append(0.87877631187439)
    # CNet.historylatency_datalist[1].append(1.0628080368042)
    # CNet.historylatency_datalist[1].append(1.99345278739929)
    # CNet.historylatency_datalist[1].append(0.962835073471069)
    # CNet.historylatency_datalist[1].append(0.962835073471069)
    # CNet.historylatency_datalist[2].append(1.0628080368042)
    # CNet.historylatency_datalist[2].append(1.99345278739929)
    # CNet.historylatency_datalist[2].append(0.962835073471069)
    # CNet.historylatency_datalist[2].append(0.962835073471069)
    # CNet.historylatency_datalist[0].append(0.449026584625244)
    # CNet.historylatency_datalist[0].append(1.62191438674927)
    # CNet.historylatency_datalist[0].append(1.86640167236328)
    # CNet.historylatency_datalist[0].append(0.732714176177979)
    # CNet.historylatency_datalist[1].append(1.84072160720825)
    # CNet.historylatency_datalist[1].append(1.85952019691467)
    # CNet.historylatency_datalist[1].append(0.940254211425781)
    # CNet.historylatency_datalist[1].append(0.936151266098022)
    # CNet.historylatency_datalist[2].append(0.140600442886353)
    # CNet.historylatency_datalist[2].append(0.095891714096069)
    # CNet.historylatency_datalist[2].append(0.166377305984497)
    # CNet.historylatency_datalist[2].append(0.331668376922607)

    # h=[[7.90022683143616,0.674335479736328,0.59386157989502,1.73618292808533],
    # [1.16858220100403,6.10432195663452,6.84672069549561,7.4188437461853],
    # [0.44202733039856,1.2173273563385,1.39243459701538,0.292113065719604]]

    # h=[[0.434957027435303,7.90022683143616,0.674335479736328,0.59386157989502],
    # [0.780951738357544,2.01064229011536,1.00651264190674,0.921807289123535],
    # [0.115871429443359,0.257956504821777,0.2787008285522460,0.116214752197266]]

    # h=[[0.1497793197631836,0.1781473159790039,0.15816640853881836,],
    # [0.44681596755981445,0.4469425678253174,0.44666504859924316,],
    # [1.8704204559326172,1.882544755935669,1.826906681060791]]


    # for i in range(dnnsnum):
    #     for j in range(historylen):
    #         CNet.historylatency_datalist[i][j]=h[i][j]

    CNet.net.eval()
    mbatchsize=1
    for dnn in dnnslist:
        mbatchsize=mbatchsize*(partitioncountdict[dnn]+1)
    with open("latency/ppp.txt","w") as pppf:
        for i in range(dnnsnum):
            pppf.write(f'p{str(i)} predict{str(i)}  actual{str(i)}   ')
        pppf.write("\n")
        for batch in range(mbatchsize):
            state=[]
            real_latency=[]
            hhhplist=[11,31,20]
            for dnn in range(CNet.dnnsum):
                pp=CNet.plistu[dnn][batch]   
                #****#   historypool
                state.extend([(CNet.historypool[dnn][pp][i]-CNet.latencyrangelist_low[dnn])/CNet.latencyrangelist_length[dnn] for i in range(CNet.historylength)])
                #****#  固定
                #state.extend([(CNet.historylatency_datalist[dnn][i]-CNet.latencyrangelist_low[dnn])/CNet.latencyrangelist_length[dnn] for i in range(CNet.historylength)])
                #****#    histrypool 里的一组
                #state.extend([(CNet.historypool[dnn][hhhplist[dnn]][i]-CNet.latencyrangelist_low[dnn])/CNet.latencyrangelist_length[dnn] for i in range(CNet.historylength)])
                
                state.append(CNet.staticlatencyofpplist[dnn][pp])  #改成静态标准化的延时  #该点的延时,上一次的??? or 静态?  or 平均? 
                real_latency.append(CNet.historypool[dnn][pp][-1])
            state=torch.tensor(state).view(1,-1)
            outputs =CNet.net(state.to(CNet.device))
            for i in range(dnnsnum):  #预测延时逆归一化
                outputs[0][i]=outputs[0][i]*CNet.latencyrangelist_length[i]+CNet.latencyrangelist_low[i]

            for i in range(dnnsnum):
                pppf.write(f'{str(CNet.plistu[i][batch])}   {str(outputs[0][i].item())} {str(real_latency[i])} ')
            pppf.write("\n")

        # for dnn in range(CNet.dnnsum):
        #     for i in range(historylen):
        #         print(" ",CNet.historypool[dnn][hhhplist[dnn]][i])
        #     print("\n")