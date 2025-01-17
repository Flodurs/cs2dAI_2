import reinforcement.combatLvL
import reinforcement.reinforcementLearningAgent

import pygame
import cs2d.zeichner

class reinforcementMain:
    def __init__(self):
        self.bRunning = True 
        self.w = reinforcement.combatLvl.combatLvl()
        self.renderer = cs2d.zeichner.zeichner()
        pygame.init()

    def run(self):
        while self.bRunning == True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.bRunning = False
            self.update()
            self.render()
            #time.sleep(0.02)
            
            
    def update(self):
        self.w.update()
            
    def render(self):
        self.renderer.render(self.w)