import eins_gegen_eins.eins_gegen_eins_lvl
import reinforcement.reinforcementLearningAgent

import time
import pygame
import cs2d.zeichner

class einsGegenEinsMain:
    def __init__(self):
        self.bRunning = True 
        self.w = eins_gegen_eins.eins_gegen_eins_lvl.einsGegenEinsLvl()
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
        self.w.update_match()
            
    def render(self):
        self.renderer.render(self.w)