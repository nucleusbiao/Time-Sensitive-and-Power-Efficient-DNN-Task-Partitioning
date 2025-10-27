import random
from itertools import count
# from tensorboardX import SummaryWriter
# import gym
from collections import deque
import numpy as np
from torch.nn import functional as F
import torch
import torch.nn as nn
class Dueling_DQN(nn.Module):
    def __init__(self, state_dim, action_dim,dnn_sum,ppmaxlist):
        super(Dueling_DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ppmaxlist=ppmaxlist
        self.dnn_sum=dnn_sum

        self.f1 = nn.Linear(state_dim, 512)
        self.f2 = nn.Linear(512, 256)

        self.val_hidden = nn.Linear(256, 128)
        self.adv_hidden = nn.Linear(256, 128)

        # ppsum=0
        # for ppmax in ppmaxlist:
        #     ppsum+=ppmax+1
        self.val=nn.Linear(128, 1)
        self.adv=nn.Linear(128, action_dim)

    def forward(self, x):

        x = self.f1(x)
        x = F.relu(x)
        x = self.f2(x)
        x = F.relu(x)

        val_hidden = self.val_hidden(x)
        val_hidden = F.relu(val_hidden)

        adv_hidden = self.adv_hidden(x)
        adv_hidden = F.relu(adv_hidden)

        val = self.val(val_hidden)  #

        adv = self.adv(adv_hidden)

        adv_ave = torch.mean(adv, dim=1, keepdim=True)

        x = adv + val - adv_ave

        return x

    def select_action(self, state):
        with torch.no_grad():
            # print(state)
            Q = self.forward(state) #(1,n_actions)
            action_indexlist=[]
            left=0
            right=0
            # print("Q    ",Q) #
            # print('Q[0]',Q[0])#(n_actions,)
            # print('Q[0][0:5]',Q[0][0:5])  #(5,)
            for i in range(self.dnn_sum):
                right+=self.ppmaxlist[i]+1
                #print(left,right)
                action_indexlist.append(torch.argmax(Q[0][left:right]).item())
                left=right
        # return action_index.item()
        print("action_indexlist",action_indexlist) # list (dnn_sum,)
        return action_indexlist

 
class Memory(object):
    def __init__(self, memory_size:int):
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > self.size():
            batch_size = self.size()
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()
