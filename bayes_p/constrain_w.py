import numpy as np

class ConstraintModel_w:

    def __init__(
            self,
        ) -> None:
        pass

    def predict(self,X):  #X (size，N)     size为自变量范围内的采样个数10000,N 为自变量个数
        newresult = np.ones(X.shape)  # (size，N)
        for j in range(X.shape[0]):
            for i in range(X.shape[1]):
                newresult[j][i]=self.result[i][round(X[j][i])]
        probility=newresult[:,0]
        for i in range(1,X.shape[1]):
            aa=newresult[:,i]
            probility=probility*aa
        return np.array(probility)

    def update(self,data):
        if data is not None:
            self.result=data  #这里需要各个任务各个分割点满足时间限制的概率：[parray1,parray2]  
            #print("self.result",self.result)

if __name__ =="__main__":
    consmodel=ConstraintModel_w()
    a1=np.array([1,2,3,4,5,6,7,8,9,10])
    a2=np.array([11,12,13,14,15,16,17,18,19,20])
    constrainpro=[]
    constrainpro.append(a1)
    constrainpro.append(a2)
    consmodel.update(constrainpro)
    #X=np.random.rand(10,2)
    #X=np.array([[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9]])
    #X=np.array([[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9]])
    X=np.array([[0,9],[1,8],[2,7],[3,6],[4,5],[5,4],[6,3],[7,2],[8,1],[9,0]])
    print(consmodel.predict(X))
    






