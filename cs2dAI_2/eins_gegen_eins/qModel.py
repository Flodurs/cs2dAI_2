import torch
from torch import nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import copy

import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, inputNum, outputNum):
        super().__init__()
        
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(inputNum, 1000),
            nn.LeakyReLU(0.1),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.1),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.1),
            nn.Linear(1000, outputNum))
        
        self.hidden_1 = nn.Linear(inputNum, 1000)
        self.hidden_2 = nn.Linear(1000, 1000)
        self.reLu = nn.ReLU()
        self.value = nn.Linear(1000,1)
        self.advantage = nn.Linear(1000,outputNum)
            
        

    def forward(self, x):
        work = self.hidden_1(x)
        work = self.hidden_2(self.reLu(work))

        adv = self.advantage(self.reLu(work))
        val = self.value(self.reLu(work))

        avg = torch.mean(adv, dim=-1, keepdim=True)

        Q = val + adv - avg
        return Q

        
class qModel:


    def __init__(self, input_num, output_num):
        self.REPLAY_MEMORY_SIZE = 300000
        self.MIN_REPLAY_MEMORY_SIZE = 1000
        self.MINI_BATCH_SIZE = 64
        self.DISCOUNT = 0.995
        self.UPDATE_TARGET_INTERVAL = 2
        self.LEARNING_RATE = 0.0001
        self.TAU = 0.005
        self.avg_td_error = 0.0
        self.AVG_FACTOR = 100
      
        self.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
        
        self.policy_model = NeuralNetwork(input_num, output_num).to(self.device)
        self.target_model = NeuralNetwork(input_num, output_num).to(self.device)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        self.optimizer = optim.AdamW(self.policy_model.parameters(), lr=self.LEARNING_RATE, amsgrad=True)

    def saveModel(self, path):
        pass

    def set_model(self, params):
        self.policy_model.load_state_dict(params)

    def get_model(self):
        return self.policy_model.state_dict()

    def _forward_policy_model(self, input):
        return self.policy_model(torch.FloatTensor(input).to(self.device)).cpu()
    
    def _forward_target_model(self, input):
        return self.policy_model(torch.FloatTensor(input).to(self.device)).cpu()
    
    def forward_policy_model(self, input):
        with torch.no_grad():
            return self.policy_model(torch.FloatTensor(input).to(self.device)).cpu().detach().numpy()
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        
    def train(self, terminalState):
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return
        
        
        #sample batch for training
        batch = random.sample(self.replay_memory, self.MINI_BATCH_SIZE) 
        state_batch = [transition[0] for transition in batch]
        action_batch = [transition[1] for transition in batch]
        reward_batch = [transition[2] for transition in batch]
        next_state_batch = [transition[3] for transition in batch]
        done_batch = [transition[4] for transition in batch]

        

        policy_model_q_values = self._forward_policy_model(state_batch)
        policy_model_q_values_copy = policy_model_q_values.detach().clone()

        

        with torch.no_grad():
            next_state_target_model_q_values = self._forward_target_model(next_state_batch)
            next_state_model_q_values = self.forward_policy_model(next_state_batch)
        
        
        y = []
        for i in range(self.MINI_BATCH_SIZE):
            if done_batch[i] == True:
                y_i = policy_model_q_values_copy[i]
                y_i[action_batch[i]] = reward_batch[i]
                y.append(y_i)
            else:
                y_i = policy_model_q_values_copy[i]
                #td_error = abs(y_i[action_batch[i]] - (reward_batch[i] + self.DISCOUNT * max(next_state_target_model_q_values[i])))

                # Q-learniing
                #y_i[action_batch[i]] = reward_batch[i] + self.DISCOUNT * max(next_state_target_model_q_values[i])

                # DDQN
                #print(np.argmax(next_state_model_q_values[i]))
                y_i[action_batch[i]] = reward_batch[i] + self.DISCOUNT * next_state_target_model_q_values[i][np.argmax(next_state_model_q_values[i])]
                
                #if td_error >= self.avg_td_error*self.AVG_FACTOR:
                    #print(td_error, self.avg_td_error)
                    #self.update_replay_memory([state_batch[i],action_batch[i],reward_batch[i],next_state_batch[i],done_batch])
                    #print("SURPRISE MOTHAFUCKA")

                #self.avg_td_error=(self.avg_td_error+td_error)/2
                y.append(y_i)
        

        criterion = nn.SmoothL1Loss()
        loss = criterion(policy_model_q_values, torch.stack(y))
        """ if random.uniform(0,1) < 0.1:
            print("---------------")
            print(loss) """

        
        loss.backward()
        
        torch.nn.utils.clip_grad_value_(self.policy_model.parameters(), 30)
        self.optimizer.step()
        self.optimizer.zero_grad()
            
        target_net_state_dict = self.target_model.state_dict()
        policy_net_state_dict = self.policy_model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_model.load_state_dict(target_net_state_dict)
     
            



    
    
    
    
    
    
    