import cs2d.baseAgent
import os
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import reinforcement.qModel
import reinforcement.episode

import random
import time
import math

import itertools

import copy 

import common.livePlot

from collections import deque

class reinforcementLearningAgent2(cs2d.baseAgent.baseAgent):
    def __init__(self,pos):
        super().__init__(pos)
        
        self.lP = common.livePlot.livePlot(1)
        self.q_plot = common.livePlot.livePlot(2)
        
        self.START_EPSILON = 1
        self.END_EPSILON = 0.05
        self.DECAY_EPSILON = 200000
        self.FLAT_EPSILON = 0.05
        
        self.FRAMES_EXPOSED = 10
        
        self.inputNumPerFrame = 10
        self.actionNum = 7
        self.inputNum = (self.inputNumPerFrame*self.FRAMES_EXPOSED)
        
        self.epsilon = self.START_EPSILON
        
        self.q = reinforcement.qModel.qModel(self.inputNum,self.actionNum)
  
        self.step = 0
        self.global_step = 0
     
        self.lastAction = 0

        self.last_y = 0
        
        self.stateCollection = deque(maxlen=self.FRAMES_EXPOSED)
        self.lastStateCollection = []
        
        for i in range(self.FRAMES_EXPOSED):
            self.stateCollection.append([0.0 for i in range(self.inputNumPerFrame)])
            self.lastStateCollection.append([0.0 for i in range(self.inputNumPerFrame)])

    def reset(self):
        self.resetPhysics()
            
        
    def think(self,world):
        self.lP.update()
        
        current_state = self.getViewList()
        current_state = self.processInputs(current_state)
        current_state.append(self.step/1000)
        
        self.stateCollection.append(current_state)
        
        
        reward = 0
        done = False
        
        #adjust eps
        self.epsilon = self.END_EPSILON + (self.START_EPSILON - self.END_EPSILON) * math.exp(-1.0 * self.global_step / self.DECAY_EPSILON)
        #self.epsilon = self.FLAT_EPSILON


        #check for reward
   


        if self.pos[1] > 300:
            reward = 1
            self.pos = [400,100]
          
            self.step = 0
            self.rotation = 0
            self.resetPhysics()
            done = True
            
            print("Epsilon: " +str(self.epsilon))
            
            self.lP.addData(reward)
            self.lP.drawAvgLast()
            self.q_plot.draw()
            self.q_plot.clear()
            
            if self.epsilon > self.END_EPSILON:
                self.epsilon *= self.DECAY_EPSILON

            
            
        if self.step > 500:
            
            reward = 0
            self.pos = [400,100]
            
            self.rotation = 0
            self.resetPhysics()
            done = True
            print("Epsilon: " + str(self.epsilon))
            self.step = 0
          
            self.lP.addData(reward)
            self.lP.drawAvgLast()
            self.q_plot.draw()
            self.q_plot.clear()
           
            if self.epsilon > self.END_EPSILON:
                self.epsilon *= self.DECAY_EPSILON
        
        #update Replay Memory 
        if self.global_step != 0:
            self.q.update_replay_memory([list(itertools.chain.from_iterable(self.lastStateCollection)),self.lastAction,reward,list(itertools.chain.from_iterable(self.stateCollection)),done])
        
        if done:
            self.stateCollection = deque(maxlen=self.FRAMES_EXPOSED)
            self.lastStateCollection = []
            for i in range(self.FRAMES_EXPOSED):
                self.stateCollection.append([0.0 for i in range(self.inputNumPerFrame)])
                self.lastStateCollection.append([0.0 for i in range(self.inputNumPerFrame)])
            print("done")

        #choose Action
        action = 0
       
        if random.uniform(0,1) < self.epsilon:
            action = random.randrange(0,self.actionNum)
            #print("Random")
        else:
            q_values = self.q.forward_policy_model(list(itertools.chain.from_iterable(self.stateCollection)))
            action = np.argmax(q_values)
     
            self.q_plot.addData(max(q_values))
            
            

        #store for transition
        self.lastAction = action
        self.lastState = []
        self.lastStateCollection = copy.deepcopy(self.stateCollection)
       
        self.executeAction(action)
        
       
        self.q.train(done)
            
        self.step+=1
        self.global_step+=1
        

        
       
      
        