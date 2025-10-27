#24.12.19
#遗传算法GA

import random
import numpy as np

class GA():
    '''
    Parameters
    ----------
    N: int.
        number of tasks.

    size: int.
        Population size.

    gene_length: int
        The length of an individual's genetic sequence
    
    '''
    def __init__(self,
                size: int,
                gene_length: int,
                dnnnamelist:list,
                ):
        self.N=len(dnnnamelist)
        self.poplist=self.initialize_population(self.N, size, gene_length)  #初始化N个种群
        self.size=size
        self.gene_length=gene_length
        self.bound=self.setbound(dnnnamelist)  #十进制种群参数范围

    def initialize_population(self,N,size, gene_length):
        # N: 任务个数
        # size: 种群的大小
        # gene_length: 单位个体基因序列的长度
        # 生成初始种群，每个个体由随机生成的基因序列组成
        poplist = [np.random.randint(2, size=(size, gene_length)) for _ in range(N)] #matrix (size, gene_length*N)
        return poplist
    
    def translateDNA(self): #pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
        #x_pop = pop[:,1::2]#奇数列表示X
        #y_pop = pop[:,::2] #偶数列表示y
        #for pop in self.poplist:
        tresult=[]
        for i in range(self.N):
            #pop:(POP_SIZE,DNA_SIZE)*(DNA_SIZE,1) --> (POP_SIZE,1)
            x = self.poplist[i].dot(2**np.arange(self.gene_length)[::-1])/float(2**self.gene_length-1)*(self.bound[i][1]-self.bound[i][0])+self.bound[i][0]
            #y = y_pop.dot(2**np.arange(DNA_SIZE)[::-1])/float(2**DNA_SIZE-1)*(Y_BOUND[1]-Y_BOUND[0])+Y_BOUND[0]
            x=x.astype(int)
            tresult.append(x)
        return tresult
    
    def setbound(self,dnnnamelist):
        pbound=[]   #种群的上界
        partitioncountdict={"vgg":22,"yolo": 31,"resnet":20,"alexnet":11,"mobilenetv2":21,"googlenet":18}
        for dnn in dnnnamelist:
            pbound.append([0,partitioncountdict[dnn]])
        return pbound
    
    def mutation(self,child, MUTATION_RATE=0.20):
        if np.random.rand() < MUTATION_RATE: 				#以MUTATION_RATE的概率进行变异
            mutate_point = np.random.randint(0, self.gene_length)	#随机产生一个实数，代表要变异基因的位置
            child[mutate_point] = child[mutate_point]^1 	#将变异点的二进制为反转

    def crossover_and_mutation(self, CROSSOVER_RATE = 0.8):
        for i in range(self.N):  #对每个种群分别交叉变异
            pop=self.poplist[i]
            new_pop = pop.copy()
            #print(new_pop is pop)
            for j in range(self.size):
                father=pop[j]		#遍历种群中的每一个个体，将该个体作为父亲
                child = father.copy()		#孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
                #print(child is father)
                if np.random.rand() < CROSSOVER_RATE:			#产生子代时不是必然发生交叉，而是以一定的概率发生交叉
                    mother = pop[np.random.randint(self.size)]	#再种群中选择另一个个体，并将该个体作为母亲
                    cross_points = np.random.randint(low=0, high=self.gene_length)	#随机产生交叉的点
                    child[cross_points:] = mother[cross_points:]		#孩子得到位于交叉点后的母亲的基因
                self.mutation(child)	#每个后代有一定的机率发生变异
                #print((father==child).all())
                new_pop[j]=child.copy()
            self.poplist[i]=new_pop.copy()
     
    def select(self, fitness):    # nature selection wrt pop's fitness
        self.bestlist=[]
        new_pop=[]
        for i in range(self.N):
            pop=self.poplist[i]
            idx = np.random.choice(np.arange(self.size), size=self.size, replace=True,
                                p=(fitness[i])/(abs(fitness[i].sum())) )
            #bestid=np.max(fitness)
            #self.bestlist.append(pop[bestid])
            new_pop=pop[idx]
            self.poplist[i]=new_pop.copy()
        return self.translateDNA()
    
    def getfitness(self,power,p_latency):
        #由power 和 latency 计算得分
        fitness=power*p_latency
        return fitness
    
# def mu(cc):
#     cc[0][0]=100
def getfitness(nextDNN,taskname):
    fitness=[]
    # for dnn in nextDNN:
    #     fit=nextDNN
    pmaxlist={'vgg':22,"yolo":31}
    for i in range(len(nextDNN)):
        pmax=pmaxlist[taskname[i]]
        power=(nextDNN[i])*4/pmax+3
        #power=-(nextDNN[i])*4/pmax+7
        fit=100.0/power-10
        fitness.append(fit.tolist())
    #print("fitness ",fitness)
    return np.array(fitness)

if __name__ == "__main__":
    taskname=['vgg','yolo']
    ga=GA(8,20,taskname)   #(8,16),  (8,20) (10,20) 0.2, 0.8
    #print(ga.poplist)
    print(ga.translateDNA())
    nextDNN=ga.translateDNA()
    for i in range(40):
        ga.crossover_and_mutation()
        #print("cm: ",ga.translateDNA())
        nextDNN=ga.translateDNA()
        fitness=getfitness(nextDNN,taskname)
        print("select: ",ga.select(fitness))

    #fitness=np.array([[-1,-1,-2,-2,-100,-100],[-1,-1,-2,-2,-100,-100]])
    # fitness=np.array([1,1,2,2,10,10])
    

    # old=np.array([[1,2],[3,4]])
    # for ll in old:
    #     ll=np.array([0,0])
    #     print(ll is old[0])
    # print(old)
    # for i in range(len(old)):
    #     old[i]=np.array([0,0])
    # mu(old)
    # print(old)
 


















