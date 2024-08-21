import cs2d.baseWorld
import playground.human_controlled
import numpy as np

class playgroundLvl(cs2d.baseWorld.baseWorld):
    def __init__(self):
        super().__init__()
        self.agents.append(playground.human_controlled.human_controlled(np.array([200.0,100.0])))
        self.aWalls.append([200,200,400,100])
        self.aWalls.append([200,500,400,100])
        
        self.aWalls.append([0,0,30,800])
        self.aWalls.append([770,0,30,800])