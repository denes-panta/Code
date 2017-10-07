import random
import string
import numpy as np
import time

class population: #Population - single gender species
    def __init__(self, n = 100, target = 'Sentence.'):
        # n: number of species in the population
        # target: the target, towards the population tries to evolve
        self.n = n #Population size
        self.fit = [0] * n #Starting Fittness Scores
        self.apex = list([target, len(target)]) #Target for the evolution
        self.cells = list(string.ascii_letters) + list(string.digits) + list(' ') + list('!@#$%&*()[]{}=_-/-+.\",') #Building blocks of the String Species
        self.population = [] #Empty list for the population
        
        for speciment in range(n): #Creating the initial population of "String creatures"
            self.population.append(''.join(random.sample(self.cells, len(target))))
 
    
    def fitness_function(self, score, max_score): #Calculates the Fitness Score
        #score = fitness score
        #max_score = score of the perfect species / length of the target
        result = score**2/max_score
        return result
    
    
    def fit_scores(self): #Calculates the Fittness Scores for each species
        self.fit = [] # Empty list for Fittness Scores
        
        for i, speciment in enumerate(self.population): #Evaluate each species of the population
            score = 0 # Initial score for the Speciment
            for cell_t, cell_c in zip(list(self.apex[0]), speciment): #Compare each cell of each Species to their respective target cells
                if cell_t == cell_c: score += 1
            self.fit.append(self.fitness_function(score, self.apex[1])) #Call the fitness function to calculate the Fitness Score
    
    
    def evolution(self, evol = 'split', k = 2, mutaprob = 0.01):
        # evol: the type of evolution
        # k: number of species needed for procreation
        # mutaprob: chance of mutation
        new_generation = [] #List for the Next Generation
        self.mutaprob = mutaprob #mutation probability
        self.k_parent = k #Number of parents needed for reproduction
        self.evol_type = evol # evolution type
        self.rel_prob = np.nan_to_num(self.fit / np.nansum(self.fit)).tolist() #Calculating the relative probability based on Fitness Scores
        candidates = [] #Candidate pool for procreation
        
        for speciments in range(self.n): #randomly chooses two speciments based on the relative probabilities
            candidates = np.random.choice(self.population, self.k_parent, p = self.rel_prob, replace = False) #Selects k number of candidates based on relative probability
            
            if self.evol_type == 'split': #Evolving using the split method
                splits = int(np.round(self.apex[1]/self.k_parent)) #mid point of the String Speciment
                child = candidates[0][0:splits] + candidates[1][splits:self.apex[1]] #Takes the first and last half of the parents, respectively
            
            if self.evol_type == 'rand': #Evolve using random genes of parents
                child = candidates[0]
                parent = np.random.binomial(1, 0.5, self.apex[1]) #randomly selecting which cell comes from which parent
                for cell in range(self.apex[1]):  #Takes random genes from each parent
                    list(child)[cell] = list(candidates[parent[cell]])[cell]
                child = ''.join(child)    
                
            for i, cell in enumerate(child): #Random mutation based on the mutaprob
                if np.random.binomial(1, self.mutaprob, 1) == 1:
                    lab = list(child)
                    lab[i] = ''.join(random.sample(self.cells, 1))
                    child = ''.join(lab)
                    
            new_generation.append(child) #Append the new generation with the mutated child
        return new_generation       


    def paragon(self): #Returnes the spl;;;eciment with the highest Fitness Score.
        fit_scores = np.asarray(self.fit)
        index = np.argmax(fit_scores)
        return self.population[index]
      
        
#Script
def letsplaygod(n = 500, target = 'The Orville', evol = 'split', mutaprob = 0.01):
    start_time = time.time()
    experiment = population(n = n, target = target)
    generation = 0
    while experiment.paragon() != experiment.apex[0]:
        experiment.fit_scores()
        experiment.population = experiment.evolution(evol = evol, k = 2, mutaprob = mutaprob) #This version only works with k = 2
        generation += 1
        if (generation+1) % 10 == 0: print('Generation: %d: ' % (generation+1) + str(experiment.paragon()))
    duration = (time.time() - start_time)/60
    print('')
    print('Final Species: %s' % (experiment.paragon()))
    print('Number of Generations: %d' % (generation))
    print('Population size: %d' % (experiment.n))
    print('Mutation chance: %d' % (experiment.mutaprob*100))
    print('Number of parents: %d' % (experiment.k_parent))
    print('Evolution type: %s' % (experiment.evol_type))
    print('Time of evolution: %0.2f' % (duration) + ' minutes.')


letsplaygod(n = 1000, target = 'The Orville.', evol = 'rand', mutaprob = 0.01)
