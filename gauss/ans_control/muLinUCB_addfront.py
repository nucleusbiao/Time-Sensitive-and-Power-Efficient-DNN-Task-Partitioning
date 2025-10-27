# 更改掉错误的预测延时返回顺序
# 去掉最后一分割点不更新策略  23.10.17

import numpy as np

def fillThetaContext(layerInfo, theta_context_dim):
    Action_num = len(layerInfo)
    x_theta = np.zeros((theta_context_dim, Action_num))
    actionList = []
    for i in range(Action_num):
        x_theta[0][i] = layerInfo[i][3]
        x_theta[1][i] = layerInfo[i][0]

        x_theta[2][i] = layerInfo[i][4]
        x_theta[3][i] = layerInfo[i][1]

        x_theta[4][i] = layerInfo[i][5]
        x_theta[5][i] = layerInfo[i][2]
        if theta_context_dim==7:
            x_theta[6][i] = layerInfo[i][6]
            actionList.append(layerInfo[i][7])
        else:
            actionList.append(layerInfo[i][6])
    return x_theta, actionList

def getCx(x_thetae, Action_num):
    listC_x = []
    for i in range(Action_num):
        temp = np.sqrt(np.matmul(x_thetae[:, [i]].T, x_thetae[:, [i]]))
        listC_x.append(temp[0][0])
    Cx = pow(max(listC_x), 2)
    return Cx

class muLinUCB():
    def __init__(self, mu, layerInfof, layerInfoe):
        self.mu = mu
        self.numOfAction = len(layerInfoe)
        self.thetaContextDim_f = 6
        self.thetaContextDim_e = 7
        self.x_thetaf, self.actionListf = fillThetaContext(layerInfof, self.thetaContextDim_f)
        self.x_thetae, self.actionListe = fillThetaContext(layerInfoe, self.thetaContextDim_e)
        #self.x_thetae_another, self.actionListe_another = fillThetaContext(layerInfoe_another, self.thetaContextDim_e)
        #self.x_thetae_add=self.x_thetae+self.x_thetae_another
        #self.actionListe_add=self.actionListe+self.actionListe_another


        self.C_x = getCx(self.x_thetae, self.numOfAction)
       # self.frontDelay = frontDelay

        self.frameNum = 200
        self.delta = 0.1
        self.C_noise = 0.05
        self.l_key = 0.8
        self.l_nonkey = 0.2
        self.C_theta = 1
        self.Af = np.diag(np.random.randint(1, 9, size=self.thetaContextDim_f))
        self.bf = np.zeros((self.thetaContextDim_f, 1))
        self.Ae = np.diag(np.random.randint(1, 9, size=self.thetaContextDim_e))
        self.be = np.zeros((self.thetaContextDim_e, 1))
        #w  加了一个系数
        self.alphaf = 0.1*(self.C_theta + np.sqrt(np.log((1 + self.frameNum * self.C_x * self.C_x)/self.delta) * self.thetaContextDim_f)*self.C_noise)/(1 - self.l_key)
        print(self.alphaf.item())
        print(self.alphaf.item())
        print(self.alphaf.item())
        #self.alpha =0.85

        self.alphae = 0.1*(self.C_theta + np.sqrt(np.log((1 + self.frameNum * self.C_x * self.C_x)/self.delta) * self.thetaContextDim_e)*self.C_noise)/(1 - self.l_key)
        print(self.alphae.item())
        print(self.alphae.item())
        print(self.alphae.item())
        #self.alpha =0.85

        self.forceSamplingRate = 0.25
        self.forceSampleFrame = np.ceil(np.power(self.frameNum, self.forceSamplingRate))
        print('forceSampleFrame:', self.forceSampleFrame)

    def updateDoublingTrickFrameNum(self, current_frame):
        if current_frame > self.frameNum:
            self.frameNum = self.frameNum * 2
            # #w 加了一个系数
            self.alphaf = 0.025*(self.C_theta + np.sqrt(np.log((1 + self.frameNum * self.C_x * self.C_x)/self.delta) * self.thetaContextDim_f)*self.C_noise)/(1 - self.l_key)
            self.alphae = 0.025*(self.C_theta + np.sqrt(np.log((1 + self.frameNum * self.C_x * self.C_x)/self.delta) * self.thetaContextDim_e)*self.C_noise)/(1 - self.l_key)
            # print("alpha:",self.alpha.item())
            # print("alpha:",self.alpha.item())

            self.forceSampleFrame = np.ceil(np.power(self.frameNum, self.forceSamplingRate))
            return True
        return False

    def getEstimationActionfront(self, key_frame):
        Af_inv = np.linalg.inv(self.Af)
        thetaf = np.matmul(Af_inv, self.bf)
                     
        if key_frame:
            L = self.l_key
        else:
            L = self.l_nonkey

        estimate_delay_f = []
        temp1=[]
        temp2=[]


        for action_index in range(self.numOfAction):
            x_1 = np.copy(self.x_thetaf[:, [action_index]])
            x_2 = np.copy(self.x_thetaf[:, [action_index]])

            temp_1 = np.matmul(thetaf.T, x_1)
            temp_2 = self.alphaf * np.sqrt((1 - L) * np.matmul(np.matmul(x_1.T, Af_inv), x_2))
            temp1.append(temp_1)
            temp2.append(temp_2)
            
            estimate_delay_f.append(temp_1 - temp_2 )  #悲观

        return estimate_delay_f,temp1,temp2


    def getEstimationActionback(self, key_frame):
        Ae_inv = np.linalg.inv(self.Ae)
        thetae = np.matmul(Ae_inv, self.be)
        
        if key_frame:
            L = self.l_key
        else:
            L = self.l_nonkey

        estimate_delay_e = []
        temp1=[]
        temp2=[]

        for action_index in range(self.numOfAction):
            x_1 = np.copy(self.x_thetae[:, [action_index]])
            x_2 = np.copy(self.x_thetae[:, [action_index]])

            temp_1 = np.matmul(thetae.T, x_1)
            temp_2 = self.alphae * np.sqrt((1 - L) * np.matmul(np.matmul(x_1.T, Ae_inv), x_2))
            temp1.append(temp_1)
            temp2.append(temp_2)

            
            estimate_delay_e.append(temp_1 - temp_2 )  #悲观

        return estimate_delay_e,temp1,temp2

    def getEstimationAction(self, key_frame, current_frame,total_frame_num):

        estimate_delay_f,tempf1,tempf2=self.getEstimationActionfront(key_frame)
        
        estimate_delay_e,tempe1,tempe2=self.getEstimationActionback(key_frame)

        estimate_delay = []
        for action_index in range(self.numOfAction):
            estimate_delay.append(estimate_delay_f[action_index]+estimate_delay_e[action_index])  #悲观

        if current_frame % self.forceSampleFrame == 0:
            estimate_action = estimate_delay.index(min(estimate_delay[1:-1]))  #!!!!!
        else:
            estimate_action = estimate_delay.index(min(estimate_delay))
        #return estimate_action
        #
        #print(self.alpha.item())

        
        #每_帧输出一次各个点的预测值
        # if(total_frame_num!=0):
        #  if(total_frame_num<20 or total_frame_num%100==0  or (total_frame_num<500 and total_frame_num%20==0) or (total_frame_num>300 and total_frame_num<320) ):
        #     file1=open("two_de_temp_1.txt",'a')
        #     file2=open("two_temp2.txt",'a')
        #     file1.write(f'{str(total_frame_num)}')
        #     file1.write("\n")
        #     #file2.write(f'str(total_frame_num)}{":"}')
        #     #file2.write("\n")
        #     #print('item1:')

        #     # for i in range(self.numOfAction):
        #     #     file1.write(f'{str(tempf1[i].item()):<23}{str(tempe1[i].item())}')
        #     #     file1.write("\n")
        #     # file1.close()

        #     for i in range(self.numOfAction):
        #         file2.write(f'{str(tempf1[i].item()):<23}{str(tempf2[i].item())}')
        #         file2.write("\n")
        #     file2.close()

        #     for i in range(self.numOfAction):
        #         file1.write(f'{str(tempe1[i].item()):<23}{str(tempe2[i].item())}')
        #         file1.write("\n")
        #     file1.close()


        force=(current_frame % self.forceSampleFrame ==0)
        #return estimate_action,estimate_delay[estimate_action],force,estimate_delay_f[estimate_action],tempf1[estimate_action].item(),tempf2[estimate_action].item(),estimate_delay_e[estimate_action],tempe1[estimate_action].item(),tempe2[estimate_action].item(),
        return estimate_action,estimate_delay[estimate_action],force, estimate_delay_f[estimate_action],tempf1[estimate_action].item(),tempf2[estimate_action].item(),\
                estimate_delay_e[estimate_action],tempe1[estimate_action].item(),tempe2[estimate_action].item()
               


    def updateA_b(self, estimate_action, actual_de,actual_df):
        
        self.Ae = self.Ae + np.matmul(self.x_thetae[:, [estimate_action]], self.x_thetae[:, [estimate_action]].T)
        self.be = self.be + self.x_thetae[:, [estimate_action]] * actual_de
        self.Af = self.Af + np.matmul(self.x_thetaf[:, [estimate_action]], self.x_thetaf[:, [estimate_action]].T)
        self.bf = self.bf + self.x_thetaf[:, [estimate_action]] * actual_df

if __name__ == '__main__':
    partitionInfo = {
        0: [13, 3, 24, 15346630656, 123633664, 26208256, 4818272],
        1: [12, 3, 23, 15259926528, 123633664, 22996992, 102761824],
        2: [11, 3, 22, 13410238464, 123633664, 19785728, 102761824],
        3: [11, 3, 21, 13410238464, 123633664, 16574464, 25691488],
        4: [10, 3, 20, 12485394432, 123633664, 13363200, 51381600],
        5: [9, 3, 19, 10635706368, 123633664, 10151936, 51381600],
        6: [9, 3, 18, 10635706368, 123633664, 8546304, 12846432],
        7: [8, 3, 17, 9710862336, 123633664, 6940672, 25691496],
        8: [7, 3, 16, 7861174272, 123633664, 5335040, 25691496],
        9: [6, 3, 15, 6011486208, 123633664, 4532224, 25691496],
        10: [6, 3, 14, 6011486208, 123633664, 3729408, 6423912],
        11: [5, 3, 13, 5086642176, 123633664, 2926592, 12846440],
        12: [4, 3, 12, 3236954112, 123633664, 2123776, 12846440],
        13: [3, 3, 11, 1387266048, 123633664, 1320960, 12846440],
        14: [3, 3, 10, 1387266048, 123633664, 919552, 3212648],
        15: [2, 3, 9, 924844032, 123633664, 518144, 3212648],
        16: [1, 3, 8, 462422016, 123633664, 417792, 3212648],
        17: [0, 3, 7, 0, 123633664, 317440, 3212648],
        18: [0, 3, 6, 0, 123633664, 217088, 3212648],
        19: [0, 3, 4, 0, 123633664, 16384, 804200],
        20: [0, 2, 2, 0, 20873216, 12288, 804200],
        21: [0, 1, 0, 0, 4096000, 0, 132416],
        22: [0, 0, 0, 0, 0, 0, 0]
    }

    frontDelay = [0 for index in range(len(partitionInfo))]
    muLinUCB = muLinUCB(0.25, partitionInfo, frontDelay)











