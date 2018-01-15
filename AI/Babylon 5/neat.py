import pygame as pg
import math
from operator import itemgetter
import fighter as f
import shot as sh
import neuralnet as nn
import time
import mcu
import impexp as iop

class Neat(object):
    #Define the color constants
    FONT = (204, 204, 204)
    DISJOINT = 1  
    EXESS = 1 
    MATCHED = 0.4
    
    def __init__(self, pop_size, export = False):
        
        #Create directory for save files
        if export == True:
            self.main = iop.IO.create_dir("F:/Code/AI/")
        else:
            self.main = None

        #Initialise the control variables
        self.running = True
        
        #Max epoch
        self.epoch = 0
        self.max_epoch = 20
        
        #Background
        self.space = self.bg()
        self.shot = self.shot()
        
        #Initialise the pygame Display
        pg.init()
        
        #Set screen size and load background
        self.sc_w = pg.display.Info().current_w
        self.sc_h = pg.display.Info().current_h
        self.sc_a = math.ceil(math.sqrt(self.sc_w**2 + self.sc_h**2))
        
        self.screen = pg.display.set_mode((self.sc_w, self.sc_h), pg.FULLSCREEN)
        self.screen.blit(self.space, (0, 0))
        pg.display.set_caption("Starfury")
        self.fnt = pg.font.SysFont("monospace", 18)

        #Edge masks
        self.vertical = pg.mask.from_surface(self.space)
        self.vertical = self.vertical.scale((1, self.sc_h))
        self.vertical.fill()
        self.horizontal = pg.mask.from_surface(self.space)
        self.horizontal = self.horizontal.scale((self.sc_w, 1))
        self.horizontal.fill()
        
        #Create the fighters and fighter variables
        self.f1 = f.Sa23e(100, 
                          int(self.sc_h / 2),
                          "starfury1.png",
                          "left"
                          )    
        self.f2 = f.Sa23e(1700, 
                          int(self.sc_h / 2),
                          "starfury2.png",
                          "right"
                          )
        
        #Settings
        pg.key.set_repeat(1, 50)    
        pg.mouse.set_visible(False)
        
        #Munition registry
        self.shots = []
        
        #Get the screen edge coordinates
        self.get_constants()
        self.collect_data(self.f1, self.f2)
        self.collect_data(self.f2, self.f2)
                
        #Initialize the Neat variables
        #Number of Inputs and Outputs
        self.i = len(self.f1.memory)
        self.o = 8
        
        #Innovation ID counter
        self.f1_innNum = 0
        self.f2_innNum = 0
        
        #NodeID
        self.f1_nodeID = 0
        self.f2_nodeID = 0
        
        #Innovation Dictonary
        self.f1_innDict = dict()
        self.f2_innDict = dict()
        
        #Species data dictionary
        self.f1_specDict = dict()
        self.f2_specDict = dict()
        
        #List of active species
        self.f1_specList = []
        self.f2_specList = []
        
        #Species target
        self.specTarget = 5

        #Species compatibility
        self.f1_th_comp = 0.26
        self.f2_th_comp = 0.26
        
        #Generation counter
        self.generation = 0
        
        #Genome counter
        self.gen = 0
        
        #Species counter
        self.f1_species_count = 1
        self.f2_species_count = 1
        
        #Population size
        self.n = pop_size
        
        #Mutation/replacement probabilities/rates
        self.mr_weight = 0.2
        self.rp_weight = 0.1
        
        #Innovation probability
        self.ip_node = 1
        self.ip_link = 1
        self.ip_rlink = 0.5
        
        #Crossover rate
        self.crossover = 0.7
        
        #Species probabilities
        self.th_oldage = 50
        self.py_oldage = 0.7
        self.th_youth = 10
        self.bs_youth = 1.3
        
        #Create populations of NNs for both fighters
        self.f1_pop, self.f1_innDict, self.f1_innNum, self.f1_nodeID = \
        self.pop_init(self.i, 
                      self.o, 
                      self.f1_innDict, 
                      self.f1_innNum, 
                      self.f1_nodeID
                      )

        self.f2_pop, self.f2_innDict, self.f2_innNum, self.f2_nodeID = \
        self.pop_init(self.i, 
                      self.o, 
                      self.f2_innDict, 
                      self.f2_innNum, 
                      self.f1_nodeID
                      )

        #Run the engine
        self.engine()
        
    ### Neat functions ###
    #Initiate the population
    def pop_init(self, i, o, innDict, innNum, nodeID):
        #Population
        population = []
        
        #Create the initial population
        for pop in range(self.n):
            
            #Create the Brain for the gen
            population.append(nn.Neuralnet(i, o))
            innDict, innNum, nodeID = \
            population[pop].create_net(innDict, innNum, nodeID)
            
        return population, innDict, innNum, nodeID 
    
    #Reset the game variables to default
    def reset(self):
        #Delete the fighter instances
        self.f1.__del__()
        self.f2.__del__()
        
        #Reset counter of the sa23e object
        f.Sa23e.counter_null(self)
        
        #Create the new instances for the fighters
        self.f1 = f.Sa23e(100, 
                          int(self.sc_h / 2),
                          "starfury1.png",
                          "left"
                          )
        self.f2 = f.Sa23e(1700, 
                          int(self.sc_h / 2),
                          "starfury2.png",
                          "right"
                          )      

        #Clear shots
        for ind, obj in enumerate(self.shots):
            self.shots.pop(ind).__del__
        self.shots.clear()
            
    #Collect the input data for each ship
    def collect_data(self, own, opp):
        own.memory = []
        own.memory += own.behav
        own.memory += opp.behav
        own.memory += self.env
        
        #fill up so that there are 5 number of shots data points
        #using the opponent's position and the screen size accross
        while len(own.incoming) < 5:
            #self.sc_a to make sure that the fired shots are on top of the list
            own.incoming.append([self.sc_a, 
                                 opp.gun_port[0], 
                                 opp.gun_port[1], 
                                 0, 
                                 0]
                                )
        
        #Sort the projectiles by distance to target
        own.incoming = sorted(own.incoming, key = itemgetter(0))
        
        #Get the x,y coordinates of the 5 closest shots        
        for ind, lst in enumerate(own.incoming):
            own.memory.append(lst[1])
            own.memory.append(lst[2])
            own.memory.append(lst[3])
            own.memory.append(lst[4])
        
    #Get the screen edge coordinates
    def get_constants(self):
        self.env = []
        self.env.append(1)
        self.env.append(1)
        self.env.append(self.sc_h)
        self.env.append(self.sc_w)
    
    #The conversion table for the fighters
    def conversion(self, fighter, y):
        if y == 1: fighter.vel_forward()
        if y == 2: fighter.vel_backward()
        if y == 3: fighter.vel_down()
        if y == 4: fighter.vel_up()
        if y == 5: fighter.stand()
        if y == 6: fighter.turn_l()
        if y == 7: fighter.turn_r()
        if y == 8: self.shoot(fighter)
            
    #Display scores
    def status(self):
        #Number of epochs
        l_bluesc = self.fnt.render("Epoch: %d" % (self.epoch), 
                                   True, 
                                   (self.FONT))
        self.screen.blit(l_bluesc, (20, self.sc_h - 140))
        
        #Fitness Score for the Blus ship
        l_bluesc = self.fnt.render("Blue fitness: %d" % \
                                   (self.f1.fitness("normal")), 
                                   True, 
                                   (self.FONT))
        self.screen.blit(l_bluesc, (20, self.sc_h - 120))

        #Fitness Score for the Red ship
        l_redesc = self.fnt.render("Red fitness: %d" % \
                                   (self.f2.fitness("normal")), 
                                   True, 
                                   (self.FONT))
        self.screen.blit(l_redesc, (20, self.sc_h - 100))

        #Current generation 
        l_curgen = self.fnt.render("Current generation: %d" % \
                                   (self.generation + 1), 
                                   True, 
                                   (self.FONT))
        self.screen.blit(l_curgen, (20, self.sc_h - 80))
        
        #Current genome
        l_curgnom = self.fnt.render("Current genome: %d" % (self.gen + 1), 
                                   True, 
                                   (self.FONT))
        self.screen.blit(l_curgnom,(20, self.sc_h - 60))
        
    #Adjust the threshold for speciation
    def adjust_threshold(self, specList, specTh, specTarget):
        if (len(specList) + 1) > specTarget:
            specTh += 0.03
        elif (len(specList) + 1) < specTarget and specTh > 0.3:
            specTh -= 0.03
            
        return specTh
    
    ### Engine ###
    #Engine
    def engine(self):
        while self.running == True:
            #Once X iterations have passed, reset to beginning
            #and load the next neural network
            if self.epoch == self.max_epoch:
                #Get fitness scores
                self.f1_pop[self.gen].r_fitness = self.f1.fitness("normal")      
                self.f2_pop[self.gen].r_fitness = self.f2.fitness("normal")
            
                #Reset the game objects
                self.reset()
                
                #Zero the epoch
                self.epoch = 0
                
                #Check if we reached the last genome, do crossover
                if self.gen == (self.n - 1):
                    #Calculate the adjusted fitness scores
                    self.f1_pop = mcu.evolution.adjust_fitness(self.f1_pop,
                                                               self.th_youth,
                                                               self.bs_youth,
                                                               self.th_oldage, 
                                                               self.py_oldage
                                                               )
                    self.f2_pop = mcu.evolution.adjust_fitness(self.f2_pop,
                                                               self.th_youth,
                                                               self.bs_youth,
                                                               self.th_oldage, 
                                                               self.py_oldage
                                                               )
                    
                    #Get Species Score Cards
                    self.f1_specDict, self.f1_specList = \
                    mcu.evolution.get_cards(self.f1_pop, 
                                            self.f1_specDict, 
                                            self.f1_specList
                                            )

                    self.f2_specDict, self.f2_specList = \
                    mcu.evolution.get_cards(self.f2_pop, 
                                            self.f2_specDict, 
                                            self.f2_specList
                                            )

                    #Evolve the new generation of genomes
                    self.f1_pop, self.f1_species_count = \
                    mcu.evolution.spawn(self.f1_pop,
                                        self.f1_specDict, 
                                        self.f1_specList,
                                        self.f1_th_comp,
                                        self.f1_species_count
                                        )
                    
                    self.f2_pop, self.f2_species_count = \
                    mcu.evolution.spawn(self.f2_pop,
                                        self.f2_specDict, 
                                        self.f2_specList,
                                        self.f2_th_comp,
                                        self.f2_species_count
                                        )
                    
                    for member in range(len(self.f1_pop)):                        
                        #Add Node to fighter 1
                        self.f1_innDict, self.f1_innNum, self.f1_nodeID = \
                        self.f1_pop[member].add_node(self.ip_node, 
                                                     50, 
                                                     self.f1_innDict, 
                                                     self.f1_innNum,
                                                     self.f1_nodeID
                                                     )

                        #Add Node to fighter 2
                        self.f2_innDict, self.f2_innNum, self.f2_nodeID = \
                        self.f2_pop[member].add_node(self.ip_node, 
                                                     50, 
                                                     self.f2_innDict, 
                                                     self.f2_innNum,
                                                     self.f2_nodeID                                                       
                                                     )

                        #Add link to fighter 1
                        self.f1_innDict, self.f1_innNum = \
                        self.f1_pop[member].add_link(self.ip_link,
                                                     self.ip_rlink, 
                                                     50, 
                                                     25, 
                                                     self.f1_innDict,
                                                     self.f1_innNum
                                                     )

                        #Add link to fighter 2
                        self.f2_innDict, self.f2_innNum = \
                        self.f2_pop[member].add_link(self.ip_link,
                                                     self.ip_rlink,
                                                     50,
                                                     25,
                                                     self.f2_innDict,
                                                     self.f2_innNum
                                                     )

                        #Replace weight
                        self.f1_pop[member].mutate_weight(self.mr_weight, 
                                                            self.rp_weight, 
                                                            "COLD"
                                                            )
                        self.f2_pop[member].mutate_weight(self.mr_weight, 
                                                            self.rp_weight, 
                                                            "COLD"
                                                            )
                

                        #Mutate weight
                        self.f1_pop[member].mutate_weight(self.mr_weight, 
                                                            self.rp_weight, 
                                                            "GAUSS"
                                                            )
                        self.f2_pop[member].mutate_weight(self.mr_weight, 
                                                            self.rp_weight, 
                                                            "GAUSS"
                                                            )
                        
                    #Adjust species threshold
                    self.f1_th_comp = self.adjust_threshold(self.f1_specList, 
                                                            self.f1_th_comp,
                                                            self.specTarget
                                                            )

                    self.f2_th_comp = self.adjust_threshold(self.f2_specList, 
                                                            self.f2_th_comp,
                                                            self.specTarget
                                                            )

                    #Set genome counter to 0
                    self.gen = 0

                    #Increment generation
                    self.generation += 1

                    #Export f1 & f2
                    if self.main != None:
                        iop.IO.exp(self.main, 
                                   "f1", 
                                   str(self.generation), 
                                   self.f1_innDict,
                                   self.f1_pop,
                                   self.f1_specList
                                   )
                        iop.IO.exp(self.main, 
                                   "f2", 
                                   str(self.generation), 
                                   self.f2_innDict, 
                                   self.f2_pop,
                                   self.f2_specList
                                   )
                    
                else:
                    #Increase genome counter
                    self.gen += 1
            
            #If nothing anomalistic happens, move to the next iteration
            else:
                self.epoch += 1
                
            #Event handling
            for event in pg.event.get():
                #Shell events
                #Event - Quit
                if event.type == pg.QUIT:
                    pg.quit()
                
                #Event - Quit
                if event.type == pg.KEYUP:
                    if (event.key == pg.K_ESCAPE):
                        pg.quit()
                        
            #Update behaviour of fighters
            self.f1.get_data()
            self.f2.get_data()
            self.collect_data(self.f1, self.f2)
            self.collect_data(self.f2, self.f1)
            
            #Update the neural network with the new data
            self.f1.commands = \
            self.f1_pop[self.gen].update(self.f1.memory, "active")
            self.f2.commands = \
            self.f2_pop[self.gen].update(self.f2.memory, "active")
            
            #Translate the commands into actions for the ship
            for cnd in range(0, self.o):
                self.conversion(self.f1, self.f1.commands[cnd] * (cnd + 1))
                self.conversion(self.f2, self.f2.commands[cnd] * (cnd + 1))
                
                #Move the objects and do the calculations            
                self.iterate()
                
                #Update the screen
                pg.display.update()
            
    ### Game functions ###
    #Background
    def shot(self):
        #Import background image
        shot = pg.image.load("F:/Code/AI/images/shot.png")
        shot = pg.transform.smoothscale(shot, 
                                        (int(shot.get_width() * 0.21), 
                                         int(shot.get_height() * 0.21))
                                        )
                                        
        return(shot)

    def bg(self):
        #Import background image
        background = pg.image.load("F:/Code/AI/images/bg.jpg")
        
        return(background)    

    #Edge correction
    def edge_check(self, fighter):
        #edge variable to see if edge correction was implemented
        edge = False

        #Left side        
        #Calculate the offsets
        off_x = int(fighter.x) - 0
        off_y = int(fighter.y) - 0

        #Check ot see if the fighter is offscreen
        if self.vertical.overlap(fighter.mask, (off_x, off_y)) != None:
            while self.vertical.overlap(fighter.mask, (off_x, off_y)) != None: 
                fighter.x += 1
                fighter.ship_center[0] += 1        
                fighter.gun_port[0] += 1
                off_x = int(fighter.x) - 0
                off_y = int(fighter.y) - 0
            fighter.vel_x = -fighter.vel_x / 2
            edge = True

        #Right side        
        #Calculate the offsets            
        off_x = int(fighter.x) - self.sc_w
        off_y = int(fighter.y) - 0

        #Check ot see if the fighter is offscreen
        if self.vertical.overlap(fighter.mask, (off_x, off_y)) != None:
            while self.vertical.overlap(fighter.mask, (off_x, off_y)) != None: 
                fighter.x -= 1
                fighter.ship_center[0] -= 1        
                fighter.gun_port[0] -= 1
                off_x = int(fighter.x) - self.sc_w
                off_y = int(fighter.y) - 0
            fighter.vel_x = -fighter.vel_x / 2
            edge = True

        #Top side        
        #Calculate the offsets            
        off_x = int(fighter.x) - 0
        off_y = int(fighter.y) - 0
 
        #Check ot see if the fighter is offscreen        
        if self.horizontal.overlap(fighter.mask, (off_x, off_y)) != None:
            while self.horizontal.overlap(fighter.mask, (off_x, off_y)) != None: 
                fighter.y += 1
                fighter.ship_center[1] += 1        
                fighter.gun_port[1] += 1
                off_x = int(fighter.x) - 0
                off_y = int(fighter.y) - 0
            fighter.vel_y = -fighter.vel_y / 2
            edge = True

        #Bottom side        
        #Calculate the offsets            
        off_x = int(fighter.x) - 0
        off_y = int(fighter.y) - self.sc_h

        #Check ot see if the fighter is offscreen
        if self.horizontal.overlap(fighter.mask, (off_x, off_y)) != None:
            while self.horizontal.overlap(fighter.mask, (off_x, off_y)) != None: 
                fighter.y -= 1
                fighter.ship_center[1] -= 1        
                fighter.gun_port[1] -= 1
                off_x = int(fighter.x) - 0
                off_y = int(fighter.y) - self.sc_h 
            fighter.vel_y = -fighter.vel_y / 2
            edge = True
        
        #if an adjustment has been made, recalculate the gun port and the mask
        if edge == True:
            fighter.ship_center = [fighter.x + fighter.r_center[0], 
                                   fighter.y + fighter.r_center[1]]
            fighter.g_port()
            fighter.col_mask()
            fighter.e_damage += 10
        else:
            fighter.e_damage -= 0.125
            
    #Calculate the new positions
    def iterate(self):
        #Clear screen with wallpaper
        self.screen.blit(self.space, (0, 0))        
        
        #Check screen edges
        self.edge_check(self.f1)
        self.edge_check(self.f2)
        
        #Move the ships
        self.move(self.f1)
        self.move(self.f2)
        
        #Move and display the shots
        self.move_projectiles()

        #Check for collision
        self.collision_detect(self.f1, self.f2)
        
        #Display ships
        self.display_ship(self.f1)
        self.display_ship(self.f2)
        
        #Display status
        self.status()
        
    #Shoot
    def shoot(self, fighter):
        if time.time() - fighter.gun_cdown > fighter.cdown_rate: 
            self.shots.append(sh.Shot(self.shot,
                                      fighter.angle, 
                                      fighter.gun_port, 
                                      fighter.vel_x,
                                      fighter.vel_y,
                                      fighter.id))
            fighter.gun_cdown = time.time()
        
    #Move and display ship
    def move(self, fighter):
        #Update the position of the ship
        fighter.x += fighter.vel_x
        fighter.y += fighter.vel_y
        
        #Update the position of the ship centre
        fighter.ship_center[0] += fighter.vel_x
        fighter.ship_center[1] += fighter.vel_y
        
        #Update the position of the gun port
        fighter.gun_port[0] += fighter.vel_x
        fighter.gun_port[1] += fighter.vel_y
        
    #Display the fighter model
    def display_ship(self, fighter):        
        #Display the ship
        self.screen.blit(fighter.r_model, (fighter.x, fighter.y))
        
    #Check for collision between shots and fighter
    def impact_detect(self, fighter_1, fighter_2, st, ind):
        #Calculate offsets for the mask
        off_x = int(fighter_1.x) - int(st.pos_x) + 13
        off_y = int(fighter_1.y) - int(st.pos_y) + 13
        
        #Collision check
        if st.mask.overlap(fighter_1.mask, (off_x, off_y)) != None:
            
            #Delete the shot
            self.shots.pop(ind).__del__
            
            #Add damage and score
            fighter_1.p_damage += int(self.max_epoch / 4)
            fighter_2.score += int(self.max_epoch / 4)
        
        return fighter_1, fighter_2
        
    #Check for collision between fighters
    def collision_detect(self, fighter_1, fighter_2):
        #Calculate offsets for the mask
        off_x = int(fighter_1.x) - int(fighter_2.x)
        off_y = int(fighter_1.y) - int(fighter_2.y)
        
        #Collision check
        if fighter_2.mask.overlap(fighter_1.mask, (off_x, off_y)) != None:     
            
            #Reverse and halv the speeds
            fighter_1.vel_x = -fighter_1.vel_x / 2 
            fighter_1.vel_y = -fighter_1.vel_y / 2
            fighter_2.vel_x = -fighter_2.vel_x / 2
            fighter_2.vel_y = -fighter_2.vel_y / 2
            
            #Add damage
            fighter_1.e_damage += 10
            fighter_2.e_damage += 10
            
            #Move the fighters
            self.move(fighter_1)
            self.move(fighter_2)
                
    #Move and display the shots
    def move_projectiles(self):
        self.f1.incoming.clear()
        self.f2.incoming.clear()
        
        if self.shots:
            for ind, obj in enumerate(self.shots):
                #Move the projectiles and calculate the distance to target
                if obj.id == 0:
                    obj.pos_x += obj.vel_x
                    obj.pos_y += obj.vel_y
                    obj.dtt_x = abs(self.f2.x - obj.pos_x)
                    obj.dtt_y = abs(self.f2.y - obj.pos_y)                    
                elif obj.id == 1:
                    obj.pos_x -= obj.vel_x
                    obj.pos_y -= obj.vel_y
                    obj.dtt_x = abs(self.f1.x - obj.pos_x)
                    obj.dtt_y = abs(self.f1.y - obj.pos_y)
                
                #Calculate the previous and current distance
                obj.p_dtt = obj.c_dtt
                obj.c_dtt = math.sqrt(obj.dtt_x**2 + obj.dtt_y**2)
                
                #See if the distance is closer or farther
                if obj.p_dtt >= obj.c_dtt:
                    obj.incoming = True
                elif obj.p_dtt < obj.c_dtt:
                    obj.incoming = False
                    
                #Display the shot
                self.screen.blit(self.shot, 
                                 (obj.pos_x - obj.s_center[0], 
                                  obj.pos_y - obj.s_center[0])
                                 )
                                 
                #Check for hits
                self.impact_detect(self.f1, self.f2, obj, ind)
                self.impact_detect(self.f2, self.f1, obj, ind)
                
                #Delete the out of screen projectiles
                if obj.pos_x <= 0 or \
                obj.pos_x >= self.sc_w or \
                obj.pos_y <= 0 or \
                obj.pos_y >= self.sc_h:
                    self.shots.pop(ind).__del__
                
                #Else, append the data to the fighter incoming table
                elif obj.incoming == True:
                    if obj.id == 0:
                        self.f2.incoming.append([obj.c_dtt, 
                                                 obj.pos_x, 
                                                 obj.pos_y,
                                                 obj.vel_x,
                                                 obj.vel_y])
                    elif obj.id == 1:
                        self.f1.incoming.append([obj.c_dtt, 
                                                 obj.pos_x, 
                                                 obj.pos_y,
                                                 obj.vel_x,
                                                 obj.vel_y])

if __name__ == "__main__":
    fight = Neat(20, export = True)
