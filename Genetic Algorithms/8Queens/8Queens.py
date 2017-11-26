import pygame as pg
import numpy as np
import math
import time

class queens(object):
    #Defining colors as constants
    BROWN = (102, 51, 0)
    WHITE = (255, 255, 255)
    LIGHT = (182, 155, 76)
    BLACK = (0, 0, 0)
    
    #Geneartion count
    generation = 0
    #Number of times the population is initialised
    restart = 1
    #Size of the squers
    size = 50
    #Number of squares on the boards
    boardLength = 8 
    #Starttime for the timer
    starttime = 0
    #List for square coordinates    
    centers = []    
    #Population of the species     
    population = []
    #Fitness scores for the species
    scores = [] 
    #Paragon index and score
    paragon_is = [0, 0]  
    #Maximum fitnes score = 28, because 2 under 8 = 8*7/2
    max_score = 28
    #Initial mutation probability
    mutation = 0.01
    #Proportions of near solutions
    prob_27 = 0.0 

    
    #The total possible values calculated for reference
    bruteforce = int(math.factorial(64) / (math.factorial(8) * math.factorial(56)))
    
    #Import and resize the image of the queen    
    queen = pg.image.load("F:/Code/Genetic Algorithms/8Queens/queen.png")
    queen = pg.transform.scale(queen, (50, 50))
    
    #np.random.seed(117)
    
    def __init__(self, pop, max_mutaProb, max_spike, exponent, max_gen, display):
        #Input variables
        #Size of the population
        self.n = pop
        #Ceiling of mutation probability
        self.max_mutation = max_mutaProb
        #Population reinitialised after gen: max_gen
        self.max_generation = max_gen
        #Every n-th generation displayed
        self.display_generation = display
        #The exponent for the fitness scores
        self.exp = exponent 
        #Mutation spike control
        self.max_spike = max_spike
        self.mutaspike = int(math.floor(self.n / max_spike))
        #Initialise the pygame Display
        pg.init()
        self.fnt = pg.font.SysFont("monospace", 18)
        
        self.screen = pg.display.set_mode((800, 600))
        self.screen.fill(self.WHITE)
        pg.display.set_caption("ChessBoard - 8Queens")

        #Initialise the initial values
        #Variable to see if we found the solution
        self.done = False
        #Variable to start/stop the evolutions
        self.start = False
        #Fill in the posible x and y coordinates 
        self.center_fill()
        #Create the first generation
        self.init_pop()
        #Calculate the initial scores
        self.scores = self.calc_scores() 

        #Static Labels
        self.static_labels()

        #Varing Labels
        self.labels()
        self.status_label("Stopped")

        #Initiate the board and the starting queen positions
        self.board()
        self.init_queens()
        
        #Start running the engine
        self.evolve()

    #Run the engine
    def evolve(self):
        self.engine = True

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
                            
                            #Update label
                            self.status_label("Searching")
                            
                            #Record time
                            self.starttime = time.time()

                #Event - Reset - Button r
                if event.type == pg.KEYUP:
                    if (event.key == pg.K_r):
                        #Reset/clear variable
                        self.start = False
                        self.done = False
                        self.generation = 0
                        self.restart = 1
                        self.mutation = 0.01
                        
                        #Reinitialise the board and starting positions
                        self.board()
                        self.init_queens()
                        
                        #Reinitialise population and calculate scores
                        self.init_pop()
                        self.scores = self.calc_scores()

                        #Update labels
                        self.labels()
                        self.status_label("Stopped")
                        
                        pg.draw.rect(self.screen, self.WHITE, [460, 420, 340, 20])
                        
            #The actual evolutions         
            if self.done == False and self.start == True:

                #Create new population and calculate fittness scores and muta-rate
                self.mutation_rate()
                self.population = self.crossover()
                self.scores = self.calc_scores()
                
                #Check for generation limit
                if self.generation % self.max_generation == 0:
                    self.restart += 1
                    self.init_pop()
                
                #Display best speciment for every n-th generation
                if self.generation % self.display_generation == 0:
                    self.board()
                    self.display_paragon()     
                
                #Check if we have a solutions
                if self.paragon_is[1] == self.max_score:
                    #Check time
                    duration = time.time() - self.starttime
                    
                    #Display the solution & modify label
                    self.board()
                    self.display_paragon()     
                    self.done = True
                    self.status_label("Done")
                    
                    #Display timer
                    l_timer = self.fnt.render("Runtime: %.2f minutes " % (duration / 60), 
                                               True, 
                                               (self.BLACK))
                    self.screen.blit(l_timer, (460, 420))

                #Update labels
                self.labels()

            pg.display.update()                   
                
    #Draw the board
    def board(self):
        #Draw the squares
        for x in range(1, self.boardLength+1):
            #Create a variable to mix up the coloring
            if x % 2 == 0:
                cnt = 0
            else:
                cnt = 1
            
            for y in range(1, self.boardLength+1):
                if cnt % 2 == 0:
                    pg.draw.rect(self.screen, 
                                     self.LIGHT, 
                                     [self.size*x, self.size*y, 
                                      self.size, self.size]
                                     )
                else:
                    pg.draw.rect(self.screen, 
                                     self.BROWN, 
                                     [self.size*x, self.size*y, 
                                      self.size, self.size]
                                     )
                cnt += 1
 
        #Draw the Border
        pg.draw.rect(self.screen, 
                         self.BLACK, 
                         [self.size, 
                          self.size, 
                          self.boardLength*self.size, 
                          self.boardLength*self.size], 1)
 
    #Populate the list with center coordinates           
    def center_fill(self):
        for x in range(1, self.boardLength+1):
            self.centers.append(x*self.size)

    #Create the first population and calculate the fittness scores
    def init_pop(self):     
        self.population = []

        for speciment in range(self.n):
            self.population.append([np.random.choice(self.centers, size = 8), 
                                    np.random.choice(self.centers, size = 8)])           
    
    #Put up the pieces on the board
    def init_queens(self):
        for piece in range(1, 9):
            self.screen.blit(self.queen, (piece*self.size, self.size))
    
    #Display the speciment with the best score
    def display_paragon(self):
        paragon = self.paragon_is[0]

        for piece in range(8):
            self.screen.blit(self.queen, (self.population[paragon][0][piece], 
                                          self.population[paragon][1][piece]))
        
    #Calculate Scores    
    def calc_scores(self):
        self.scores = []
        
        #Temporary array for the new scores
        new_scores = [] 
        #Reset the paragon index and score to 0
        self.paragon_is = [0, 0] 
        count_27 = 0
        
        #Calculate the scores for each speciment, 
        #using the board as a coordinate system
        for speciment in range(self.n):
            score = 0
            
            #Vertical
            score += abs(len(self.population[speciment][0]) - \
                         len(set(self.population[speciment][0])))
            
            #Horizontal
            score += abs(len(self.population[speciment][1]) - \
                         len(set(self.population[speciment][1])))
            
            #Accross
            cross1 = []
            cross2 = []
            
            for x in range(8):
                cross1.append(self.population[speciment][0][x] - \
                              self.population[speciment][1][x])
                cross2.append(self.population[speciment][0][x] + \
                              self.population[speciment][1][x])
            
            score += abs(len(cross1) - len(set(cross1)))
            score += abs(len(cross2) - len(set(cross2)))
            
            score = (self.max_score - score)
            
            #Count the near solutions
            if score == 27:
                count_27 += 1
            
            #Increase the score to the power of the input exponent
            #In order to select suitable parents with higher probability
            new_scores.append(score**self.exp)
            
            #Save the highest score for future reference
            if score > self.paragon_is[1]:
                self.paragon_is[0] = speciment            
                self.paragon_is[1] = int(score)
 
       #Get proportion on near perfect solutions       
        self.prob_27 = count_27 / self.n
        
        #Calculate the relative probabilities
        score_sum = sum(new_scores)
            
        for index, score in enumerate(new_scores):
            new_scores[index] = score / score_sum

        return new_scores      
        
    #Procreation
    def crossover(self):
        #Increase the generation number
        self.generation += 1 
        #List for the new generation   
        new_generation = [] 
        
        #Create new generation based on 2 parents
        for speciment in range(self.n):
            #List for the contenders for fairness
            contenders = [] 
            
            #Keep selecting viable parents until there are at least 2
            while len(contenders) < 2:
                for index, speciment in enumerate(self.scores):
                    if speciment - np.random.rand() >= 0:
                        contenders.append(self.population[index])
            
            #Select 2 parents from the contenders
            parents = np.random.randint(len(contenders), size=2)

            #Create a mask from the parents for the pieces (queens)
            mask = np.random.choice(parents, size = 8)
            
            #Empty array for children
            children = [[], []] 
            
            #Mutate the piece/procreate from parents based on mutation probability
            for piece in range(8):
                if self.mutation - np.random.rand() >= 0:
                    children[0].append((np.random.randint(8, size=1)[0]+1) * self.size)
                    children[1].append((np.random.randint(8, size=1)[0]+1) * self.size)
                else:
                    children[0].append(contenders[mask[piece]][0][piece])
                    children[1].append(contenders[mask[piece]][1][piece])

            new_generation.append(children)
    
        return new_generation

    #Automatic tuning of mutation rate
    def mutation_rate(self):
        length = len(set(self.scores))
        
        #Set the mutation rate to within the regular range
        if self.mutation > self.max_mutation:
            self.mutation = 0.05
        
        #Increase/Decrease the mutation rate to keep diverse population
        if length < 12 and self.mutation < (self.max_mutation - 0.01):
            self.mutation += 0.01
        if length >= 12 and self.mutation > 0.01:
            self.mutation -= 0.01  
            
        #Increase mutation rate, in case of being stuck at local optimum
        if self.prob_27 > 0.05:
            if self.mutaspike >= self.n:
                self.mutaspike = int(math.floor(self.n / self.max_spike))
            else:
                self.mutaspike += int(math.floor(self.n / self.max_spike))
            self.mutation = self.prob_27 * (self.n / self.mutaspike)

    #Status label
    def status_label(self, status):
        pg.draw.rect(self.screen, self.WHITE, [460, 400, 340, 20])
        l_status = self.fnt.render("Current status: %s" % (status), 
                                   True, 
                                   (self.BLACK))
        self.screen.blit(l_status, (460, 400))
    
    #Update labels
    def labels(self):
        pg.draw.rect(self.screen, self.WHITE, [460, 280, 340, 20])
        l_curgen = self.fnt.render("Current generation: %d" % (self.generation), 
                                   True, 
                                   (self.BLACK))
        self.screen.blit(l_curgen, (460, 280))

        pg.draw.rect(self.screen, self.WHITE, [460, 300, 340, 20])
        l_curmuta = self.fnt.render("Current mutation rate: %.1f%%" % (self.mutation * 100), 
                                    True, 
                                    (self.BLACK))
        self.screen.blit(l_curmuta, (460, 300))

        pg.draw.rect(self.screen, self.WHITE, [460, 320, 340, 20])
        l_retries = self.fnt.render("Number of rounds: %d" % (self.restart), 
                                    True, 
                                    (self.BLACK))
        self.screen.blit(l_retries, (460, 320))

        pg.draw.rect(self.screen, self.WHITE, [460, 340, 340, 20])
        l_fitscore = self.fnt.render("Current best score: %d" % (self.paragon_is[1]), 
                                     True, 
                                     (self.BLACK))
        self.screen.blit(l_fitscore, (460, 340))
    
    #Static labels
    def static_labels(self):
        l_start = self.fnt.render("Press 's' to start the evolution", 
                                  True, 
                                  (self.BLACK))
        self.screen.blit(l_start, (50, 460))
        
        l_reset = self.fnt.render("Press 'r' to reset to starting point", 
                                  True, 
                                  (self.BLACK))
        self.screen.blit(l_reset, (50, 480))
        
        l_quit = self.fnt.render("Press 'q' to quit the application", 
                                 True, 
                                 (self.BLACK))
        self.screen.blit(l_quit, (50, 500))
        
        l_total = self.fnt.render("%d possible setups" % (self.bruteforce), 
                                  True, 
                                  (self.BLACK))
        self.screen.blit(l_total, (100, 550))
        
        l_input = self.fnt.render("Input variables", 
                                  True, 
                                  (self.BLACK))
        self.screen.blit(l_input, (460, 50))
        
        l_pop = self.fnt.render("Population size: %d" % (self.n), 
                                True, 
                                (self.BLACK))
        self.screen.blit(l_pop, (460, 80))
        
        l_genlim = self.fnt.render("Generation limit: %d" % (self.max_generation), 
                                   True, 
                                   (self.BLACK))
        self.screen.blit(l_genlim, (460, 100))
        
        l_mutalim = self.fnt.render("Mutation limit: %.1f%%" % (self.max_mutation * 100), 
                                    True, 
                                    (self.BLACK))
        self.screen.blit(l_mutalim, (460, 120))
        
        l_exp = self.fnt.render("Fittness score exponent: %d" % (self.exp), 
                                   True, 
                                   (self.BLACK))
        self.screen.blit(l_exp, (460, 140))
        
        l_mutalim = self.fnt.render("Display every %d generations" % (self.display_generation), 
                                    True, 
                                    (self.BLACK))
        self.screen.blit(l_mutalim, (460, 160))

        l_evol = self.fnt.render("Evolutionary variables", 
                                 True, 
                                 (self.BLACK))
        self.screen.blit(l_evol, (460, 250))


if __name__ == "__main__":
    game = queens(pop = 300, 
                  max_mutaProb = 0.1,
                  max_spike = 10,
                  exponent = 4,
                  max_gen = 1000,
                  display = 1
                  )
