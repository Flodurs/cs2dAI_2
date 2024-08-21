import cs2d.baseWorld
import reinforcement.reinforcementLearningAgent2
import numpy as np

class reinforcementLvl(cs2d.baseWorld.baseWorld):
    def __init__(self):
        super().__init__()
        self.agents.append(reinforcement.reinforcementLearningAgent2.reinforcementLearningAgent2(np.array([400.0,100.0])))
        self.agents.append(reinforcement.reinforcementLearningAgent.reinforcementLearningAgent(np.array([400.0,800.0])))
        self.aWalls.append([200,200,400,100])
        self.aWalls.append([200,500,400,100])
        
        self.aWalls.append([0,0,30,800])
        self.aWalls.append([770,0,30,800])

        self.match_step = 0
        self.reward_A = 0
        self.reward_B = 0

    def update_match(self):
        match_step+=1
        self.reward_A = 0
        self.reward_B = 0

        if match_step == 1000:
            dist_A = np.linalg.norm(self.agents[0].getPos()-np.array([400,400]))
            dist_B = np.linalg.norm(self.agents[1].getPos()-np.array([400,400]))

            if dist_A > dist_B:
                reward_A = 1
            else:
                reward_B = 1




