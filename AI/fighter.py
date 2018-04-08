import math
import pygame as pg

class Sa23e(object):
    _ids = 0
    
    def __init__(self, pos_x, pos_y, img_name, side):
        #ID    
        self.id = Sa23e._ids
        Sa23e._ids += 1
        
        #Round
        self.round = 0
        
        #Number of controls
        self.n_ctrl = 8
        
        #Initial position of the spacecraft
        self.x = pos_x
        self.y = pos_y
        
        #Initial velocity of the spacecraft
        self.vel_x = 0.0
        self.vel_y = 0.0
        
        #Thrust & turn engine power
        self.thrust = 2
        self.turn = 0.125
        self.max_speed = 10
        
        #Initial angle of the spacecraft
        self.angle = 0

        #Previous size of the spacecraft image
        self.model = self.img_load(img_name)
        self.p_rect = self.model.get_rect()
        self.p_center = self.model.get_rect().center
        
        #After transformation size of the spacecraft image
        self.r_model = self.model
        self.r_rect = self.p_rect
        self.r_center = self.p_center
        
        #Collision mask
        self.col_mask()
        
        #Coordinates of the gun port
        if side == "left":
            self.gun = [50, 15]
        elif side == "right":
            self.gun = [-80, 15]
            
        self.ship_center = [self.x + self.r_center[0], 
                            self.y + self.r_center[1]]
        
        self.gun_port = [0, 0]
        self.g_port()
        
        #Gun status
        self.gun_cdown = 0
        self.cdown_rate = 0.5
        self.weapon_fired = False
        
        #Score & Damage
        self.score = 0
        self.p_damage = 0
        self.e_damage = 0
        
        #Data
        self.behav = []
        self.get_data()
        self.incoming = []
        self.memory = []
        
        #Commands
        self.commands = []
        
    def __del__(self):
        pass
        
    def fitness(self, mode):   
        if mode == "agressive":
            fitness = -self.e_damage - self.p_damage + 2 * self.score
        elif mode == "defensive":
            fitness = -self.e_damage - 2 * self.p_damage + self.score
        elif mode == "normal":
            fitness = -self.e_damage - self.p_damage + self.score            

        return fitness
    
    def counter_null(self):
        Sa23e._ids = 0
        
    #Create the positional variables
    def get_data(self):
        self.behav.clear()
        self.behav.append(self.angle)
        self.behav.append(self.ship_center[0])
        self.behav.append(self.ship_center[1])
        self.behav.append(self.vel_x)
        self.behav.append(self.vel_y)        
        
    #Load and transform the spacecraft image
    def img_load(self, img_name):
        temp_img = pg.image.load("F:/Code/AI/images/" + img_name)
        temp_img = pg.transform.smoothscale(temp_img, 
                                            (int(temp_img.get_width() * 0.6), 
                                             int(temp_img.get_height() * 0.6))
                                            )
        return temp_img
    
    #Change velocity of the spacecraft
    #Using the cos and sin of the unit circle
    def vel_forward(self):
        if self.thrust * math.cos(-self.angle * math.pi) > 0:
            if self.vel_x < self.max_speed * math.cos(-self.angle * math.pi):
                self.vel_x += self.thrust * math.cos(-self.angle * math.pi)
        if self.thrust * math.cos(-self.angle * math.pi) < 0:
            if self.vel_x > self.max_speed * math.cos(-self.angle * math.pi):
                self.vel_x += self.thrust * math.cos(-self.angle * math.pi)

        if self.thrust * math.sin(-self.angle * math.pi) > 0:
            if self.vel_y < self.max_speed * math.sin(-self.angle * math.pi):
                self.vel_y += self.thrust * math.sin(-self.angle * math.pi)
        if self.thrust * math.sin(-self.angle * math.pi) < 0:
            if self.vel_y > self.max_speed * math.sin(-self.angle * math.pi):
                self.vel_y += self.thrust * math.sin(-self.angle * math.pi)
    
    def vel_backward(self):
        if self.thrust * math.cos(-self.angle * math.pi) > 0:
            if self.vel_x > -self.max_speed * math.cos(-self.angle * math.pi):
                self.vel_x -= self.thrust * math.cos(-self.angle * math.pi)
        if self.thrust * math.cos(-self.angle * math.pi) < 0:
            if self.vel_x < -self.max_speed * math.cos(-self.angle * math.pi):
                self.vel_x -= self.thrust * math.cos(-self.angle * math.pi)

        if self.thrust * math.sin(-self.angle * math.pi) > 0:
            if self.vel_y > -self.max_speed * math.sin(-self.angle * math.pi):
                self.vel_y -= self.thrust * math.sin(-self.angle * math.pi)
        if self.thrust * math.sin(-self.angle * math.pi) < 0:
            if self.vel_y < -self.max_speed * math.sin(-self.angle * math.pi):
                self.vel_y -= self.thrust * math.sin(-self.angle * math.pi)

    def vel_up(self):
        if self.thrust * math.sin(self.angle * math.pi) > 0:
            if self.vel_x > -self.max_speed * math.sin(self.angle * math.pi):
                self.vel_x -= self.thrust * math.sin(self.angle * math.pi)
        if self.thrust * math.sin(self.angle * math.pi) < 0:
            if self.vel_x < -self.max_speed * math.sin(self.angle * math.pi):
                self.vel_x -= self.thrust * math.sin(self.angle * math.pi)

        if self.thrust * math.cos(self.angle * math.pi) > 0:
            if self.vel_y > -self.max_speed * math.cos(self.angle * math.pi):
                self.vel_y -= self.thrust * math.cos(self.angle * math.pi)
        if self.thrust * math.cos(self.angle * math.pi) < 0:
            if self.vel_y < -self.max_speed * math.cos(self.angle * math.pi):
                self.vel_y -= self.thrust * math.cos(self.angle * math.pi)
    
    def vel_down(self):
        if self.thrust * math.sin(self.angle * math.pi) > 0:
            if self.vel_x < self.max_speed * math.sin(self.angle * math.pi):
               self.vel_x += self.thrust * math.sin(self.angle * math.pi)
        if self.thrust * math.sin(self.angle * math.pi) < 0:
            if self.vel_x > self.max_speed * math.sin(self.angle * math.pi):
                self.vel_x += self.thrust * math.sin(self.angle * math.pi)

        if self.thrust * math.cos(self.angle * math.pi) > 0:
            if self.vel_y < self.max_speed * math.cos(self.angle * math.pi):
                self.vel_y += self.thrust * math.cos(self.angle * math.pi)
        if self.thrust * math.cos(self.angle * math.pi) < 0:
            if self.vel_y > self.max_speed * math.cos(self.angle * math.pi):
                self.vel_y += self.thrust * math.cos(self.angle * math.pi)
                
    #Turn Port
    def turn_l(self):
        #Change the angle with the turning rate        
        self.angle += self.turn
        
        #Cap the max angle at -2.0
        if self.angle >= 2.0:
            self.angle = 0    
            
        #Turn the model and get the new rectangle and centre    
        self.r_model = pg.transform.rotate(self.model, 
                                           math.degrees(self.angle * math.pi)
                                           )
        self.r_rect = self.r_model.get_rect() 
        self.r_center = self.r_rect.center
        
        #Update the position variables to retain the original centre
        self.x -= (self.r_rect[2] - self.p_rect[2]) / 2
        self.y -= (self.r_rect[3] - self.p_rect[3]) / 2
        
        #Save the new rectangle and centre coordinates
        self.p_rect = self.r_rect
        self.p_center = self.r_center

        #Recreate the mask
        self.col_mask()
    
        #Calculate the new gun port coordinates
        self.g_port()
    
    #Turn Starboard
    def turn_r(self):
        #Change the angle with the turning rate
        self.angle -= self.turn
        
        #Cap the min angle at -2.0
        if self.angle <= -2.0:
            self.angle = 0
        
        #Turn the model and get the new rectangle and centre
        self.r_model = pg.transform.rotate(self.model, 
                                           math.degrees(self.angle * math.pi)
                                           )
        self.r_rect = self.r_model.get_rect() 
        self.r_center = self.r_rect.center
        
        #Update the position variables to retain the original centre
        self.x -= (self.r_rect[2] - self.p_rect[2]) / 2
        self.y -= (self.r_rect[3] - self.p_rect[3]) / 2
        
        #Save the new rectangle and centre coordinates
        self.p_rect = self.r_rect
        self.p_center = self.r_center

        #Recreate the mask
        self.col_mask()

        #Calculate the new gun port coordinates
        self.g_port()

    #Check to see if any speed update, violated the max speed constant
    def vel_check(self):
        if self.vel_x >= self.max_speed * abs(math.cos(self.angle * math.pi)):
            self.vel_x = self.max_speed * abs(math.cos(self.angle * math.pi))

        if self.vel_x <= -self.max_speed * abs(math.cos(self.angle * math.pi)):
            self.vel_x = -self.max_speed * abs(math.cos(self.angle * math.pi))

        if self.vel_y >= self.max_speed * abs(math.sin(self.angle * math.pi)):
            self.vel_y = self.max_speed * abs(math.sin(self.angle * math.pi))

        if self.vel_y <= -self.max_speed * abs(math.sin(self.angle * math.pi)):
            self.vel_y = -self.max_speed * abs(math.sin(self.angle * math.pi))
            
    #Calculate the new gun port coordinates
    def g_port(self):       
        self.gun_port[0] = self.ship_center[0] + \
                           (self.gun[0]) * math.cos(-self.angle * math.pi) + \
                           (self.gun[1]) * math.cos(-self.angle * math.pi) + \
                           20 * math.sin(self.angle * math.pi)
                           
        self.gun_port[1] = self.ship_center[1] + \
                           (self.gun[0]) * math.sin(-self.angle * math.pi) + \
                           (self.gun[1]) * math.sin(-self.angle * math.pi) + \
                           20 * math.cos(self.angle * math.pi)

    #Create mask for the image model    
    def col_mask(self):
        self.mask = pg.mask.from_surface(self.r_model, 10)
    
    #Cut engines and initate stop
    def stand(self):
        if self.vel_x > 0:
            self.vel_x -= self.thrust
        if self.vel_x < 0:
            self.vel_x += self.thrust
        if self.vel_x < self.thrust and self.vel_x > -self.thrust:
            self.vel_x = 0

        if self.vel_y > 0:
            self.vel_y -= self.thrust
        if self.vel_y < 0:
            self.vel_y += self.thrust
        if self.vel_y < self.thrust and self.vel_y > -self.thrust:
            self.vel_y = 0
                