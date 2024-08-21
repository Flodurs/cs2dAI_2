import cs2d.baseAgent
import os
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import eins_gegen_eins.qModel


import random
import time
import math

import itertools

import copy 

import common.livePlot

from collections import deque

class combatLearningAgent(cs2d.baseAgent.baseAgent):
    def __init__(self,pos):
        super().__init__(pos)
        
        self.lP = common.livePlot.livePlot(1)
        self.q_plot = common.livePlot.livePlot(2)
        
        self.START_EPSILON = 1
        self.END_EPSILON = 0.05
        self.DECAY_EPSILON = 200000
        self.FLAT_EPSILON = 0.05
        self.FRAME_SKIPING_INTERVAL = 10
        self.FRAMES_EXPOSED = 10
        
        self.inputNumPerFrame = 10
        self.actionNum = 8
        self.inputNum = (self.inputNumPerFrame*self.FRAMES_EXPOSED)
        
        self.epsilon = self.START_EPSILON
        
        self.q = eins_gegen_eins.qModel.qModel(self.inputNum,self.actionNum)
  
        self.step = 0
        self.global_step = 0
     
        self.lastAction = 0

        self.last_y = 0
        
        self.stateCollection = deque(maxlen=self.FRAMES_EXPOSED)
        self.lastStateCollection = []

        self.reward = 0
        self.done = False

        self.learning = False
        
        

        
        for i in range(self.FRAMES_EXPOSED):
            self.stateCollection.append([0.0 for i in range(self.inputNumPerFrame)])
            self.lastStateCollection.append([0.0 for i in range(self.inputNumPerFrame)])
            
        
    def think(self,world):
        if self.global_step%self.FRAME_SKIPING_INTERVAL != 0:
            self.global_step+=1
            self.epsilon = self.END_EPSILON + (self.START_EPSILON - self.END_EPSILON) * math.exp(-1.0 * self.global_step / self.DECAY_EPSILON)
            return


        
        #choose Action
        action = 0
        
        
        current_state = self.getViewList()
        current_state = self.processInputs(current_state)
        current_state.append(self.step/1000)
        
        self.stateCollection.append(current_state)

        if self.learning == False:
         
            self.global_step+=1
            if random.uniform(0,1) < self.epsilon:
                action = random.randrange(0,self.actionNum)
                #print("Random")
                
            else:
                q_values = self.q.forward_policy_model(list(itertools.chain.from_iterable(self.stateCollection)))
                action = np.argmax(q_values)
                

            self.executeAction(action)
            return 
        

        self.lP.update()
        reward = self.reward
        done = self.done
    
        self.reward = 0
        self.done = False

      
        

        
        #update Replay Memory 
        if self.global_step != 0:
            self.q.update_replay_memory([list(itertools.chain.from_iterable(self.lastStateCollection)),self.lastAction,reward,list(itertools.chain.from_iterable(self.stateCollection)),done])
        
        if done:
            self.stateCollection = deque(maxlen=self.FRAMES_EXPOSED)
            self.lastStateCollection = []
            for i in range(self.FRAMES_EXPOSED):
                self.stateCollection.append([0.0 for i in range(self.inputNumPerFrame)])
                self.lastStateCollection.append([0.0 for i in range(self.inputNumPerFrame)])
            print("Epsilon: " + str(self.epsilon))
            print(reward)
            self.lP.addData(reward)
            self.lP.drawAvgLast()
            self.q_plot.draw()
            self.q_plot.clear()

        
       
        if random.uniform(0,1) < self.epsilon:
            action = random.randrange(0,self.actionNum)
            #print("Random")
        else:
            q_values = self.q.forward_policy_model(list(itertools.chain.from_iterable(self.stateCollection)))
          
            self.q_plot.addData(max(q_values))
            
            

        #store for transition
        self.lastAction = action
        self.lastState = []
        self.lastStateCollection = copy.deepcopy(self.stateCollection)
       
        self.executeAction(action)
        
       
        self.q.train(done)
            
        self.step+=1
        self.global_step+=1

    def set_reward(self, r):
        self.reward = r

    def set_done(self, d):
        self.done = d

    def set_net(self, params):
        self.q.set_model(params)

    def get_net(self):
        return self.q.get_model()
     
    def reset(self):
        self.stateCollection = deque(maxlen=self.FRAMES_EXPOSED)
        for i in range(self.FRAMES_EXPOSED):
            self.stateCollection.append([0.0 for i in range(self.inputNumPerFrame)])
            self.lastStateCollection.append([0.0 for i in range(self.inputNumPerFrame)])
        self.resetPhysics()

    def get_learning(self):
        return self.learning
    
    def set_learning(self, l):
        self.learning = l


        

        
       
      
        