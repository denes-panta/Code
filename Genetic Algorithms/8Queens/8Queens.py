import pygame
import numpy as np
import math
import time

class queens(object):
    #Defining colors as constants
    BROWN = (102, 51, 0)
    WHITE = (255, 255, 255)
    LIGHT = (182, 155, 76)
    BLACK = (0, 0, 0)
    
    generation = 0 #geneartion count
    restart = 0 #Number of times the population is initialised
    
    size = 50 #size of the squers
    boardLength = 8 #number of squares on the boards
    starttime = 0
    
    centers = [] #list for square centres        
    population = [] #population of the species
    scores = [] #fittness scores for the species
    paragon_is = [0, 0] #paragon index and score 
    
    max_score = 28 #28, because 2 under 8 = 8*7/2
    mutation = 0.01 #Initial mutation probability
    
    #The total possible values calculated for reference
    bruteforce = int(math.factorial(64) / (math.factorial(8) * math.factorial(56)))
    
    #Import and resize the image of the queen    
    queen = pygame.image.load("F:/Code/Genetic Algorithms/8Queens/queen.png")
    queen = pygame.transform.scale(queen, (50, 50))
    
    #np.random.seed(117)
    
    def __init__(self, pop, max_mutaProb, weight, max_gen, display):
        #Input variables
        self.n = pop #size of the population
        self.max_mutation = max_mutaProb #ceiling of mutation probability
        self.max_generation = max_gen #Population reinitialised after gen: max_gen
        self.display_generation = display #Every n-th generation displayed
        self.w = weight #The power weight for the fitness scores
        
        #Initialise the pyGame Display
        pygame.init()
        self.fnt = pygame.font.SysFont("monospace", 18)
        
        self.screen = pygame.display.set_mode((800, 600))
        self.screen.fill(self.WHITE)
        pygame.display.set_caption("ChessBoard - 8Queens")

        #Initialise the initial values
        self.done = False #Variable to see if we found the solution
        self.start = False #Variable to start/stop the evolutions
        self.center_fill() #Fill in the posible x and y coordinates 
        self.init_pop() #Create the first generation
        self.scores = self.calc_scores() #Calculate the initial scores

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
            for event in pygame.event.get():
                #Event - Quit - Window close button
                if event.type == pygame.QUIT:
                    pygame.quit()
                
                #Event - Quit - Button q
                if event.type == pygame.KEYUP:
                    if (event.key == pygame.K_q):
                        pygame.quit()

                #Event - Start - Button s
                if event.type == pygame.KEYUP:
                    if (event.key == pygame.K_s):
                        self.start = True
                        
                        #Update label
                        self.status_label("Searching")
                        
                        #Record time
                        self.starttime = time.time()

                #Event - Reset - Button r
                if event.type == pygame.KEYUP:
                    if (event.key == pygame.K_r):
                        #Reset/clear variable
                        self.start = False
                        self.done = False
                        self.generation = 0
                        self.restart = 0
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
                        
                        pygame.draw.rect(self.screen, self.WHITE, [460, 420, 340, 20])
                        
            #The actual evolutions         
            if self.done == False and self.start == True:
                
                #Check for generation limit
                if self.generation % self.max_generation == 0:
                    self.restart += 1
                    self.init_pop()
                
                #Craete new population and calculate fittness scores
                self.population = self.crossover()
                self.scores = self.calc_scores()
                
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
                    l_timer = self.fnt.render("Runtime: %.2f minutes " % (duration/60), 
                                               True, 
                                               (self.BLACK))
                    self.screen.blit(l_timer, (460, 420))
                
                #Update labels and mutation rate
                self.labels()
                self.mutation_rate()
            
            pygame.display.update()                   
                
    #Draw the board and get the centers
    def board(self):
        
        for x in range(1, self.boardLength+1):
            #Create a variable to mix up the coloring
            if x % 2 == 0:
                cnt = 0
            else:
                cnt = 1
                
            for y in range(1, self.boardLength+1):
                if cnt % 2 == 0:
                    pygame.draw.rect(self.screen, 
                                     self.LIGHT, 
                                     [self.size*x, self.size*y, 
                                      self.size, self.size]
                                     )
                else:
                    pygame.draw.rect(self.screen, 
                                     self.BROWN, 
                                     [self.size*x, self.size*y, 
                                      self.size, self.size]
                                     )
                cnt += 1
 
        #Border
        pygame.draw.rect(self.screen, 
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

        new_scores = [] #Temporary array for the new scores
        self.paragon_is = [0, 0] #Reset the paragon index and score to 0
        
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
            
            #Increase the score to the power of the input weight
            #In order to select suitable parents with higher probability
            new_scores.append((self.max_score - score)**self.w)
            
            #Save the highest score for future reference
            if (self.max_score - score) > self.paragon_is[1]:
                self.paragon_is[0] = speciment            
                self.paragon_is[1] = int(self.max_score - score)
        
        #Calculate the relative probabilities
        score_sum = sum(new_scores)
        new_scores = np.asarray(new_scores)
        rel_scores = new_scores / score_sum

        return rel_scores.tolist()      
        
    #Procreation
    def crossover(self):
        self.generation += 1 #Increase the generation number
             
        new_generation = [] #List for the new generation
        
        #Create new generation based on 2 parents
        for speciment in range(self.n):
            contenders = [] #List for the contenders for fairness
            
            #Keep selecting viable parents until there are at least 2
            while len(contenders) < 2:
                for index, speciment in enumerate(self.scores):
                    if speciment - np.random.rand() >= 0:
                        contenders.append(self.population[index])
            
            #Select 2 parents from the contenders
            parents = np.random.randint(len(contenders), size=2)

            #Create a mask from the parents for the pieces (queens)
            mask = np.random.choice(parents, size = 8)
            
            children = [[],[]] #Empty array for children
            
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

    #Change mutation rate to prevent inbreading or rapid mutation
    def mutation_rate(self):
        length = len(set(self.scores))
        
        if length < 12 and self.mutation < (self.max_mutation - 0.01):
            self.mutation += 0.01
        if length >= 12 and self.mutation > 0.01:
            self.mutation -= 0.01  
    
    #Status label
    def status_label(self, status):
        pygame.draw.rect(self.screen, self.WHITE, [460, 400, 340, 20])
        l_status = self.fnt.render("Current status: %s" % (status), 
                                   True, 
                                   (self.BLACK))
        self.screen.blit(l_status, (460, 400))
    
    #Update labels
    def labels(self):
        pygame.draw.rect(self.screen, self.WHITE, [460, 280, 340, 20])
        l_curgen = self.fnt.render("Current generation: %d" % (self.generation), 
                                   True, 
                                   (self.BLACK))
        self.screen.blit(l_curgen, (460, 280))

        pygame.draw.rect(self.screen, self.WHITE, [460, 300, 340, 20])
        l_curmuta = self.fnt.render("Current mutation rate: %.1f%%" % (self.mutation * 100), 
                                    True, 
                                    (self.BLACK))
        self.screen.blit(l_curmuta, (460, 300))

        pygame.draw.rect(self.screen, self.WHITE, [460, 320, 340, 20])
        l_retries = self.fnt.render("Number of rounds: %d" % (self.restart), 
                                    True, 
                                    (self.BLACK))
        self.screen.blit(l_retries, (460, 320))

        pygame.draw.rect(self.screen, self.WHITE, [460, 340, 340, 20])
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
        
        l_weight = self.fnt.render("Exponent weight: %d" % (self.w), 
                                   True, 
                                   (self.BLACK))
        self.screen.blit(l_weight, (460, 140))
        
        l_mutalim = self.fnt.render("Display every %d generation" % (self.display_generation), 
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
                  weight = 4,
                  max_gen = 500,
                  display = 1
                  )
    
