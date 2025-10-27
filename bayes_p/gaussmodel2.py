#24.8.14
#高斯过程回归预测延时模型
#24.8.21
#增加两个最大一个最小(静态的，除了0)作为训练数据。

import collections
import numpy as np
from scipy.optimize import minimize
import pickle
import copy

import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)
print("guass parent dir ",parent_dir)

class Gaussmodel():    
    def __init__(self,netname,username,optimize=True,):

        self.netname=netname
        partitioncountdict={"vgg":22,"yolo": 31,"resnet":20,"alexnet":11,"mobilenetv2":21}
        self.partitionmax=partitioncountdict[netname]
        self.maxhistorylenth=80   #使用的历史数据个数
        # print("maxhistorylength ",self.maxhistorylenth)
        self.latencypool_total=collections.deque(maxlen=self.maxhistorylenth)
        self.latencypool_pp=collections.deque(maxlen=self.maxhistorylenth)

        self.latencypool_client=collections.deque(maxlen=self.maxhistorylenth)
        self.latencypool_trans=collections.deque(maxlen=self.maxhistorylenth)
        self.latencypool_edge=collections.deque(maxlen=self.maxhistorylenth)
        self.latencypool_clientpp=collections.deque(maxlen=self.maxhistorylenth)
        self.latencypool_transpp=collections.deque(maxlen=self.maxhistorylenth)
        self.latencypool_edgepp=collections.deque(maxlen=self.maxhistorylenth)
        self.step=0
        self.x=np.array([1.0,1.0,1.0])
      
        self.ppmax=partitioncountdict[netname]

        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 0.5, "sigma_f": 0.2}
        # print(self.params)
        self.optimize = optimize

        #静态测量的数据  client,trans,edge
        self.avergedatadic_net=[]
        pdata=""
        user=username   #!!!!!!!!!1
        if(pdata)!='':
            file=str(parent_dir)+'/models/pkl/'+pdata+'/'+netname +user+'.pkl'
        else:
            file=str(parent_dir)+'/models/pkl/'+ netname +user+'.pkl'
        print('file: ',file)
        with open (file, 'rb') as temp:
            pret= pickle.load(temp)
            self.avergedatadic_net.append(pret[0])  #clientaver
            self.avergedatadic_net.append(pret[2])  #transaver
            self.avergedatadic_net.append(pret[4])  #edgeaver

        #找出静态数据的最大最小值
        self.min_latencystatic=[]
        self.minsecond_latencystatic=[]
        self.max_latencystatic=[]
        self.maxsecond_latencystatic=[]
        #排序
        averdeta=copy.deepcopy(self.avergedatadic_net)
        for i in range(3): 
            for j in range(0,self.partitionmax):  #最大值
                for k in range(0,self.partitionmax-j):
                    if(averdeta[i][k]>averdeta[i][k+1]):
                        a=averdeta[i][k]
                        averdeta[i][k]=averdeta[i][k+1]
                        averdeta[i][k+1]=a

        for i in range(3):
            self.max_latencystatic.append(averdeta[i][-1])
            self.maxsecond_latencystatic.append(averdeta[i][-2])
            self.min_latencystatic.append(averdeta[i][1])
            self.minsecond_latencystatic.append(averdeta[i][2])
        
        self.updateold=0
        self.updatacount=0
        self.updataverage=0


    def fit(self, X, y):
        # store train data
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)
        self.is_fit = True
        # hyper parameters optimization
        # def negative_log_likelihood_loss(params):
        #     self.params["l"], self.params["sigma_f"] = params[0], params[1]
        #     Kyy = self.kernel(self.train_X, self.train_X) + 1e-8 * np.eye(len(self.train_X))
        #     loss = 0.5 * self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y) + 0.5 * np.linalg.slogdet(Kyy)[1] + 0.5 * len(self.train_X) * np.log(2 * np.pi)
        #     return loss.ravel()

        # if self.optimize:
        #     res = minimize(negative_log_likelihood_loss, [self.params["l"], self.params["sigma_f"]],
        #            bounds=((1e-4, 1e4), (1e-4, 1e4)),
        #            method='L-BFGS-B')
        #     self.params["l"], self.params["sigma_f"] = res.x[0], res.x[1]

    def predict_mucov(self, X):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return
        X = np.asarray(X)
        Kff = self.kernel(self.train_X, self.train_X)  # (N, N)
        Kyy = self.kernel(X, X)  # (k, k)
        Kfy = self.kernel(self.train_X, X)  # (N, k)
        Kff_inv = np.linalg.inv(Kff + 1e-8 * np.eye(len(self.train_X)))  # (N, N)
        mu = Kfy.T.dot(Kff_inv).dot(self.train_y)   #(k*3)
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)  #(k,k)  ?
        return mu, cov

    def kernel(self, x1, x2):
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return self.params["sigma_f"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * dist_matrix)
    
    def update_data(self,actual_latency,a_client,a_trans,a_edge,pp):

        #25.1.2增加，若连续选同一个分割点5次，而延时在其10%范围内，不更新数据！！
        if pp==self.updateold:
            if self.updatacount>=5: 
                if actual_latency<self.updataverage*1.3 and actual_latency>self.updataverage*0.7:
                    return
                self.updataverage=(self.updataverage*self.updatacount+actual_latency)/(self.updatacount+1)
            self.updatacount+=1
        else:
            self.updateold=pp
            self.updatacount=0
            self.updataverage=0

        #更新数据
        if pp==0:
            self.latencypool_trans.append(a_trans)
            self.latencypool_transpp.append(pp)
            self.latencypool_edge.append(a_edge)
            self.latencypool_edgepp.append(pp)
        elif pp==self.ppmax:
            self.latencypool_client.append(a_client)
            self.latencypool_clientpp.append(pp)
        else:
            self.latencypool_client.append(a_client)
            self.latencypool_clientpp.append(pp)
            self.latencypool_trans.append(a_trans)
            self.latencypool_transpp.append(pp)
            self.latencypool_edge.append(a_edge)
            self.latencypool_edgepp.append(pp)
        self.latencypool_total.append(actual_latency)
        self.latencypool_pp.append(pp)
        return

    def predict(self):
        mu_sum,cov_sum,mu=self.predictall()
        #找出最小延时分割点
        ppmin=0
        latencymin=mu_sum[0]  #(ppmax+1,0) ??
        for j in range(1,len(mu_sum)):
            if mu_sum[j]<latencymin:
                ppmin=j
                latencymin=mu_sum[j]
        covindepend=[]
        for i in range(len(cov_sum)):  
            covindepend.append(cov_sum[i][i])
        covindepend=np.array(covindepend)
        #负值处理，静态值。
        for i in range(len(mu_sum)):   #mu_sum array (ppmax+1,)
            if mu_sum[i]<0:
                mu_sum[i]=self.avergedatadic_net[0][i]+self.avergedatadic_net[1][i]+self.avergedatadic_net[2][i]
        return ppmin,mu_sum,covindepend,mu
 
    def predictall(self):
        partitionpoint=[i for i in range(self.partitionmax+1)]
        mu=[i for i in range(3)]
        cov=[i for i in range(3)]
        train_x=[]
        length=self.maxhistorylenth
        for i in range(3):
            train_y=[]
            pplist=[]
            if i==0:
                length=len(self.latencypool_client)  #历史数据长度
                train_y=[self.latencypool_client[i] for i in range(length)]
                pplist=self.latencypool_clientpp
            elif i==1:
                length=len(self.latencypool_trans)
                train_y=[self.latencypool_trans[i] for i in range(length)]
                pplist=self.latencypool_transpp
            else:
                length=len(self.latencypool_edge)
                train_y=[self.latencypool_edge[i] for i in range(length)]
                pplist=self.latencypool_edgepp
            train_x=[self.avergedatadic_net[i][pp] for pp in pplist]
            
            #增加静态数据，为稳定性。
            train_y.append(self.max_latencystatic[i])
            train_x.append(self.max_latencystatic[i])
            train_y.append(self.min_latencystatic[i])
            train_x.append(self.min_latencystatic[i])
            train_y.append(self.maxsecond_latencystatic[i])
            train_x.append(self.maxsecond_latencystatic[i])
            train_y.append(self.minsecond_latencystatic[i])
            train_x.append(self.minsecond_latencystatic[i])
            train_y=np.array(train_y).reshape(-1,1)
            train_x=np.array(train_x).reshape(-1,1)

            self.fit(train_x,train_y)
            test_X=np.zeros((len(partitionpoint),1))
            for pp in partitionpoint:
                test_X[pp][0]=self.avergedatadic_net[i][pp]
            test_X=np.array(test_X).reshape(-1,1)
            mu[i], cov[i] = self.predict_mucov(test_X) #mu[3,ppmax+1,1] 3:client ,trans, edge
        mu=np.array(mu)  #(3,ppmax+1,1)
        mu[2][self.ppmax][0]=0 #强制，最后分割点的edge延时为0。尝试。
        mu[1][self.ppmax][0]=0
        mu[0][0][0]=0     #强制，完全卸载，client 延时为0
        mu_sum=mu.sum(0)  #(ppmax+1,1)
        mu_sum=mu_sum.squeeze()  #(ppmax+1,)

        cov=np.array(cov) #(3,ppmax+1,1)
        cov_sum=cov.sum(0) #(ppmax+1,1)
        cov_sum=cov_sum.squeeze()
        return mu_sum,cov_sum,mu
    

    #用于和贝叶斯优化结合，限制条件概率预测值
    def predict_w(self,X):
        mu_sum,cov_sum,mu=self.predictall(X)   #注意，
        #找出最小延时分割点
        ppmin=0
        latencymin=mu_sum[0]  #(ppmax+1,0) ??
        for j in range(1,len(mu_sum)):
            if mu_sum[j]<latencymin:
                ppmin=j
                latencymin=mu_sum[j]
        covindepend=[]
        for i in range(len(cov_sum)):  #应该拿对角线？？？
            covindepend.append(cov_sum[i][i])
        covindepend=np.array(covindepend)
        return mu_sum,covindepend
    
    def predictall_w(self,X):
        partitionpoint=[i for i in range(self.partitionmax+1)]
        mu=[i for i in range(3)]
        cov=[i for i in range(3)]
        train_x=[]
        length=self.maxhistorylenth
        for i in range(3):
            train_y=[]
            pplist=[]
            if i==0:
                length=len(self.latencypool_client)  #历史数据长度
                train_y=[self.latencypool_client[i] for i in range(length)]
                pplist=self.latencypool_clientpp
            elif i==1:
                length=len(self.latencypool_trans)
                train_y=[self.latencypool_trans[i] for i in range(length)]
                pplist=self.latencypool_transpp
            else:
                length=len(self.latencypool_edge)
                train_y=[self.latencypool_edge[i] for i in range(length)]
                pplist=self.latencypool_edgepp
            train_x=[self.avergedatadic_net[i][pp] for pp in pplist]
            
            #增加静态数据，为稳定性。
            train_y.append(self.max_latencystatic[i])
            train_x.append(self.max_latencystatic[i])
            train_y.append(self.min_latencystatic[i])
            train_x.append(self.min_latencystatic[i])
            train_y.append(self.maxsecond_latencystatic[i])
            train_x.append(self.maxsecond_latencystatic[i])
            train_y.append(self.minsecond_latencystatic[i])
            train_x.append(self.minsecond_latencystatic[i])
            train_y=np.array(train_y).reshape(-1,1)
            train_x=np.array(train_x).reshape(-1,1)

            self.fit(train_x,train_y)
            test_X=np.zeros((len(X),1))
            for x in X:
                test_X[int(x)][0]=self.avergedatadic_net[i][int(x)]
            test_X=np.array(test_X).reshape(-1,1)
            mu[i], cov[i] = self.predict_mucov(test_X) #mu[3,ppmax+1,1] 3:client ,trans, edge
        mu=np.array(mu)  #(3,ppmax+1,1)
        mu[2][self.ppmax][0]=0 #强制，最后分割点的edge延时为0。尝试。
        mu[1][self.ppmax][0]=0
        mu[0][0][0]=0     #强制，完全卸载，client 延时为0
        mu_sum=mu.sum(0)  #(ppmax+1,1)
        mu_sum=mu_sum.squeeze()  #(ppmax+1,)

        cov=np.array(cov) #(3,ppmax+1,1)
        cov_sum=cov.sum(0) #(ppmax+1,1)
        cov_sum=cov_sum.squeeze()
        return mu_sum,cov_sum,mu

####给定数据测试 8.19
if __name__=='__main__':
    import time
    lsm_f='latency/lsmlatency.txt'
    with open (lsm_f,"w") as f: 
        # f.write('total_real          predicted_T     pp      latency_c       predict_c       latency_trans       predict_trans       latency_edge         predict_edge\n')  #total
        f.write('pp          predict_c       predict_trans                predict_edge     total \n')
    netname="yolo"
    Gausspro=Gaussmodel(netname,"tx2",True)
    partitioncountdict={"vgg":22,"yolo": 31,"resnet":20,"alexnet":11,"mobilenetv2":21}
    partitionmax=partitioncountdict[netname]
    #while True:
    for _ in range(3):
        clientlist=[6.87E-05,0.011907816,0.010550499,0.025974751,0.027619362,0.037201643,0.048042774,0.058775663,0.056913853,0.081011772,0.097543716,0.096423626]
        translist=[0.135524273,0.124430418,0.041548014,0.142946482,0.034973621,0.042278767,0.031532288,0.038624763,0.017763615,0.015559435,0.012260675,0]
        edgelist=[0.005579472,0.004529476,0.003862143,0.003388643,0.002053738,0.00276041,0.002789736,0.00165534,0.001583099,0.001640797,0.001092672,0]
        
        givenclient=[0.135986567,0.149464369,0.161713362,0.260403872,0.218372822,
        0.12071538,0.172060966,0.141236305,0.184698582,0.214035034,
        0.13676095,0.215865374,0.192050457,0.199114561,0.19761467,
        0.152778625,0.12016654,0.228905201,0.147747278,0.132423639,
        0.240341663,0.141989231,0.137560844,0.154010534,0.218547344,
        0.147313356,0.182492495,0.243707418,0,0,0,
        0.182394028,0.20702529,0.268782139,0.23579669,0.143007278,
        0.143087864,0.17071414,0.071658134,0.314146519,]
        givenpplist=[12,12,12,12,12,
                     12,12,12,12,12,
                     12,12,12,12,12,
                     12,12,12,12,12,
                     12,12,12,12,12,
                     12,12,12,31,31,
                     31,12,12,12,12,
                     12,12,16,20,0]
        for i in range(40):
            pp=givenpplist[i]
            a_client=givenclient[i]
            a_trans=1
            a_edge=1
            actual_latency=a_client+a_trans+a_edge
            Gausspro.update_data(actual_latency,a_client,a_trans,a_edge,pp)
        if len(Gausspro.latencypool_pp)>=7:
            ppmin,mu_sum,mu=Gausspro.predict()
            print(ppmin)
            with open(lsm_f,"a") as f:
                for pi in range(partitionmax+1):
                    f.write(f"{str(pi):<8}{str(mu[0][pi][0]):<25}{str(mu[1][pi][0]):<25}{str(mu[2][pi][0]):<25}{str(mu_sum[pi][0]):<25}")
                    f.write("\n")
        time.sleep(0.01)










    
    


    





        















 





