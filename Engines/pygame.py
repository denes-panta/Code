import pygame as pg

class starfury(object):
    WHITE = (255, 255, 255)
    
    def __init__(self):
        #Initialise the control variables
        self.engine = True
        self.start = False
        self.done = False

        #Initialize the state variables
        self.generation = 0
        self.mutation = 0.01

        
        #Initialise the pygame Display
        pg.init()
        self.screen = pg.display.set_mode((800, 600))
        self.screen.fill(self.WHITE)
        self.engine()
        pg.display.set_caption("Starfury")
    
    def engine(self): 
        while self.engine == True:
            #Event handling
            for event in pg.event.get():
                #Event - Quit - Window close button
                if event.type == pg.QUIT:
                    pg.quit()
                
                #Event - Quit - Button q
                if event.type == pg.KEYUP:
                    if (event.key == pg.K_q):
                        pg.quit()
                
                #Event - Start - Button s
                if event.type == pg.KEYUP:
                    if (event.key == pg.K_s):
                        if self.start == False:
                            self.start = True
                            
                #Event - Reset - Button r
                if event.type == pg.KEYUP:
                    if (event.key == pg.K_r):
                        #Reset/clear variable
                        self.start = False
                        self.done = False
                        self.generation = 0
                        self.mutation = 0.01
                                                                        
if __name__ == "__main__":
    fight = starfury()