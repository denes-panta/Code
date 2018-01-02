import pygame as pg
import math
     
class shot(object):
    def __init__(self, angle, position, vel_x, vel_y, name):

        #The projectile was fired from which fighter
        # 0 == player 1, 1 == player 2
        self.id = name
                    
        #Image details
        self.s_img = self.img_load()
        self.s_rect = self.s_img.get_rect()
        self.s_center = self.s_rect.center
        
        #Mask
        self.mask = pg.mask.from_surface(self.s_img, 10)
        
        #Velocity of the shot
        if self.id == 0:
            self.velocity_x = 20 * math.cos(-angle * math.pi) + vel_x
            self.velocity_y = 20 * math.sin(-angle * math.pi) + vel_y
        elif self.id == 1:
            self.velocity_x = 20 * math.cos(-angle * math.pi) - vel_x
            self.velocity_y = 20 * math.sin(-angle * math.pi) - vel_y
        
        #Position of the shot
        self.pos_x = position[0]
        self.pos_y = position[1]
        
        #Distance to target
        self.dtt_x = None
        self.dtt_y = None
        self.dtt = None
        
    def __del__(self):
        pass
    
    def img_load(self):
        temp_img = pg.image.load("F:/Code/AI/images/shot.png")
        temp_img = pg.transform.smoothscale(temp_img, 
                                            (int(temp_img.get_width() * 0.21), 
                                             int(temp_img.get_height() * 0.21))
                                            )
        return temp_img