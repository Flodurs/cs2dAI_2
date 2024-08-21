import cs2d.baseAgent
import pygame
import numpy as np
import math

class human_controlled(cs2d.baseAgent.baseAgent):
    def __init__(self,pos):
        super().__init__(pos)

    def think(self,world):
        current_state = self.getViewList()
        current_state = self.processInputs(current_state)
        print(str(current_state[6]) + " " + str(self.rotation))
        

        keys=pygame.key.get_pressed()
        if keys[pygame.K_s]:
            self.executeAction(0)
        if keys[pygame.K_w]:
            self.executeAction(1)
        if keys[pygame.K_a]:
            self.executeAction(3)
        if keys[pygame.K_d]:
            self.executeAction(2)

        if keys[pygame.K_k]:
            self.executeAction(4)

        if keys[pygame.K_l]:
            self.executeAction(5)

        #mouse_x, mouse_y = pygame.mouse.get_pos()
        #pos = np.array([mouse_x,mouse_y])
        #vector_to_mouse = pos - self.pos
        #self.rotation = math.atan2(vector_to_mouse[0],vector_to_mouse[1])