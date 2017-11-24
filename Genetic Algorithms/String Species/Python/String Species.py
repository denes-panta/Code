import random
import string
import numpy as np
import time

class population(object): #Population - single gender species

    def __init__(self, n, target, evol = 'split', mutaprob = 0.01):

        #Population size
        self.n = n 
        #Evolution type
        self.evol_type = evol
        #Mutation probabilty
        self.mutation = mutaprob 
        #Starting Fittness Scores
        self.fit = [0] * n 
        #Target for the evolution
        self.apex = list([target, len(target)]) 
        #Building blocks of the String Species
        self.cells = list(string.ascii_letters) + \
                     list(string.digits) + \
                     list(' ') + \
                     list('!@#$%&*()[]{}=_-/-+.\",') 
        #Empty list for the population
        self.population = [] 
        #Generation counter
        self.generation = 0
        
        #Creating the initial population of "String creatures"
        for speciment in range(n): 
            self.population.append(''.join(random.sample(self.cells, 
                                                         len(target))))
    
    #Does the evolving
    def engine(self):
        start_time = time.time()
        
        while self.paragon() != self.apex[0]:
            self.fit_scores()
            self.population = self.evolution()
            self.generation += 1
        
            #Print Status of the evolution
            if (self.generation+1) % 10 == 0: 
                    print('Generation: %d: ' % (self.generation+1) + \
                          str(self.paragon()))
            
        duration = (time.time() - start_time) / 60
        
        #Print the evolution report
        print('')
        print('Final Speciment: %s' % (self.paragon()))
        print('Number of Generations: %d' % (self.generation))
        print('Population size: %d' % (self.n))
        print('Mutation probability: %0.1f%%' % (self.mutation * 100))
        print('Number of parents: %d' % (self.k_parent))
        print('Evolution type: %s' % (self.evol_type))
        print('Time of evolution: %0.2f' % (duration) + ' minutes.')
    
    #Calculates the Fitness Score
    def fitness_function(self, score, max_score): 
        #score = fitness score
        #max_score = score of the perfect species / length of the target
        result = score**2/max_score

        return result
    
    #Calculates the Fittness Scores for each species
    def fit_scores(self):
        self.fit = [] 
        
        for i, speciment in enumerate(self.population):
            score = 0 
            
            for cell_t, cell_c in zip(list(self.apex[0]), speciment): 
                if cell_t == cell_c: score += 1
            
            self.fit.append(self.fitness_function(score, self.apex[1])) 
    
    #Evolution    
    def evolution(self):
        new_generation = []

        #Number of parents needed for reproduction
        self.k_parent = 2 
        
        #Calculating the relative probability based on Fitness Scores
        self.rel_prob = np.nan_to_num(self.fit / np.nansum(self.fit)).tolist()
        
        #Candidate pool for procreation
        candidates = [] 
    
        #Randomly chooses two speciments based on the relative probabilities
        for speciments in range(self.n): 
            
            #Selects k number of candidates based on relative probability
            candidates = np.random.choice(self.population, 
                                          self.k_parent, 
                                          p = self.rel_prob, 
                                          replace = False) 
            
            #Evolve using the split method
            if self.evol_type == 'split': 
                splits = int(np.round(self.apex[1]/self.k_parent))
                child = candidates[0][0:splits] + \
                        candidates[1][splits:self.apex[1]]
            
            #Evolve using random genes of parents
            if self.evol_type == 'rand': 
                child = candidates[0]
                parent = np.random.binomial(1, 0.5, self.apex[1])
                
                #Takes random genes from each parent
                for cell in range(self.apex[1]):  
                    list(child)[cell] = list(candidates[parent[cell]])[cell]
                child = ''.join(child)    
            
            #Random mutation based on the mutaprob
            for i, cell in enumerate(child): 
                if np.random.binomial(1, self.mutation, 1) == 1:
                    lab = list(child)
                    lab[i] = ''.join(random.sample(self.cells, 1))
                    child = ''.join(lab)
                    
            new_generation.append(child)
            
        return new_generation       

    #Returns the speciment with the highest Fitness Score.
    def paragon(self): 
        fit_scores = np.asarray(self.fit)
        index = np.argmax(fit_scores)
        
        return self.population[index]


if __name__ == "__main__":
    strevol = population(n = 500, 
                         target = 'The Orville.', 
                         evol = 'rand', 
                         mutaprob = 0.01)
    strevol.engine()
