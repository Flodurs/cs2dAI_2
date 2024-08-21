import cs2d.baseWorld
import eins_gegen_eins.combat_learning_agent
import numpy as np

class einsGegenEinsLvl(cs2d.baseWorld.baseWorld):
    def __init__(self):
        super().__init__()
        self.agents.append(eins_gegen_eins.combat_learning_agent.combatLearningAgent(np.array([400.0,100.0])))
        self.agents.append(eins_gegen_eins.combat_learning_agent.combatLearningAgent(np.array([400.0,800.0])))


        self.aWalls.append([200,200,400,100])
        self.aWalls.append([200,500,400,100])
        
        self.aWalls.append([0,0,30,800])
        self.aWalls.append([770,0,30,800])

        self.match_step = 0
        self.global_step = 0

        self.agents[1].set_net(self.agents[0].get_net())

        self.reset_lvl()


        self.agents[0].set_learning(True)

        self.UPDATE_OPPONENT_FREQ = 10000



    def update_match(self):
        self.match_step+=1
        self.global_step += 1

        if self.match_step >= 1000:
            dist_A = np.linalg.norm(self.agents[0].getPos()-np.array([400,400]))
            dist_B = np.linalg.norm(self.agents[1].getPos()-np.array([400,400]))
            self.reset_lvl()

            if dist_A < dist_B:
                print("A won!")
                self.agents[0].set_reward(1)
                self.agents[0].set_done(True)
            else:
                print("B won")
                self.agents[0].set_done(True)


        if self.global_step%self.UPDATE_OPPONENT_FREQ == 0:
            self.agents[1].set_net(self.agents[0].get_net())
            print("Updating Opponent")

    def reset_lvl(self):

        self.match_step = 0

        for agent in self.agents:
            agent.reset()

        if np.random.uniform(0,1) >= 0.5:
            self.agents[0].setPos(np.array([400.0,100.0]))
            self.agents[1].setPos(np.array([400.0,800.0]))

            self.agents[0].setRotation(0)
            self.agents[1].setRotation(np.pi)
        else:
            self.agents[1].setPos(np.array([400.0,100.0]))
            self.agents[0].setPos(np.array([400.0,800.0]))

            self.agents[0].setRotation(np.pi)
            self.agents[1].setRotation(0)

