import pygame as pg
import math
     
class Shot(object):
    def __init__(self, img, angle, position, vel_x, vel_y, name):

        #The projectile was fired from which fighter
        # 0 == player 1, 1 == player 2
        self.id = name
                    
        #Image details
        self.s_img = img
        self.s_rect = self.s_img.get_rect()
        self.s_center = self.s_rect.center
        
        #Mask
        self.mask = pg.mask.from_surface(self.s_img, 10)
        
        #Velocity of the shot
        if self.id == 0:
            self.vel_x = 25 * math.cos(-angle * math.pi) + vel_x
            self.vel_y = 25 * math.sin(-angle * math.pi) + vel_y
        elif self.id == 1:
            self.vel_x = 25 * math.cos(-angle * math.pi) - vel_x
            self.vel_y = 25 * math.sin(-angle * math.pi) - vel_y
        
        #Position of the shot
        self.pos_x = position[0]
        self.pos_y = position[1]
        
        #Distance to target
        self.dtt_x = 0
        self.dtt_y = 0
        self.c_dtt = 0
        self.p_dtt = 0
        
    def __del__(self):
        pass