import numpy as np
import parts as p
import math
from operator import itemgetter

class evolution(object):
    def spawn(pop, specDict, specList, sameSpecies):
        #Threshold to assign genomes into the same species
        threshold = sameSpecies
        
        #Link of the new generation of genomes
        new_gen = []
        
        #
        for ind, species in enumerate(specList):
            new_gen.append(pop[specDict[species].leader])

        for ind, species in enumerate(specList):        
            p1 = None
            p2 = None
            
            to_spawn = math.ceil(specDict[species])
            
            if specDict[species].number > 1:
                while to_spawn > 0:
                    while p1 == p2:
                        p1 = \
                        np.random.random_sample(specDict[species].adjScore)[0]
                        p2 = \
                        np.random.random_sample(specDict[species].adjScore)[0]
                    
                    new_genome = evolution.crossover(pop[p1], pop[p2])
                    
                    
                    speciation(g, l)
        while len(new_gen) > len(pop):
            darwin = np.random.randint((len(specList) + 1), len(new_gen))
            new_gen[darwin].__del__()
            
        return new_gen
    
    #Calculate adjusted fitness scores for the population
    def adjust_fitness(pop, y_th, y_mod, o_th, o_mod):     
        #Define adjuster parameter at 1
        adjuster = 1
        
        for ind, gnome in enumerate(pop):                
            #If the raw score is below zero, set the raw score to 0
            if gnome.r_fitness < 0:
                gnome.r_fitness = 0
            
            #Apply the youth / old age adjustments
            if gnome.age <= y_th:
                adjuster = y_mod
            elif gnome.age >= o_th:
                adjuster = o_mod
                
            #Get the adjusted scores
            gnome.a_fitness = gnome.r_fitness * adjuster
            
        return pop
    
    #Get the relevant data from the past generation
    def get_cards(pop, specDict, specList):
        #Sum of the fitness score for the population
        fit_sum = 0
        
        #Dictionary for the species data
        for ind, species in enumerate(specList):
            specDict[species].adjScore.clear()
        
        #Clear the species list
        specList.clear()

        for ind, genome in enumerate(pop):
            #Append the active species list with the species number
            if genome.species not in specList:
                specList.append(genome.species)
                
            #If the species doesn't exist, create a card for it
            if specDict.get(genome.species, None) == None:
                specDict[genome.species] = p.Species()
            
            #Fill in the values
            #Species number
            specDict[genome.species].species = genome.species
            
            #Get leader and leader score
            if specDict[genome.species].leader == None:
                specDict[genome.species].leader = ind
                specDict[genome.species].lead_score = genome.a_fitness
            elif specDict[genome.species].lead_score < genome.a_fitness:
                specDict[genome.species].leader = ind
                specDict[genome.species].lead_score = genome.a_fitness

            #Get the adjusted and shared scores with the genome number
            l = [ind, genome.a_fitness]
            specDict[genome.species].adjScore.append(l)

        #Check to see if there has been any improvement in the species'
        for ind, species in enumerate(specList):
            if specDict[species].lead_score > specDict[species].histBest:
                specDict[species].histBest = specDict[species].lead_score
                specDict[species].impEpoch = 0
            else:
                specDict[species].impEpoch += 1
                if specDict[species].impEpoch == 15:
                    specDict[species].alive = False
        
        #Sort the scores
        for ind, species in enumerate(specList):
            #Sort the scores
            specDict[species].adjScore = sorted(specDict[species].adjScore, 
                                                key = itemgetter(1), 
                                                reverse = True
                                                )
            #Get the number of genomes
            n_genome = len(specDict[species].adjScore)
            specDict[species].number = n_genome
            
            #Iterate through the fitness scores of each species and share it
            for i, score in enumerate(specDict[species].adjScore):
                #Share the fitness scores
                    score[1] /= n_genome
                    fit_sum += score[1]
            
        #Calculate the average adj. fitness score for the population
        fit_avg = fit_sum / len(pop)

        for ind, species in enumerate(specList):            
            #Iterate through the fitness scores of each species and share it
            for i, score in enumerate(specDict[species].adjScore):
                score.append(score[1] / fit_avg)
                specDict[species].spawn += score[1] / fit_avg               

            #Get the last element of the top 20% of genomes
            last = math.ceil(len(specDict[species].adjScore) * 0.2)
            
            #Get the top 20%
            specDict[genome.species].adjScore = \
            specDict[genome.species].adjScore[0:last]

        return specDict, specList

    #Get full genome from the link and node dictionary
    def get_genome(self, linkDict, nodeDict):
        #Create a list for the innovation numbers in the genome
        ginnDict = dict()
        
        #Get the innovation numbers for
        for i in range(1, len(linkDict) + 1):
            ginnDict[linkDict[i].innNum] = linkDict[i] 
            
        for i in range(1, len(nodeDict) + 1):
            ginnDict[nodeDict[i].innNum] = nodeDict[i] 
            
        return ginnDict
    
    #Check for the latest innovation in the genome
    def get_max(self, g1, g2):   
        #Compare the largest element in both lists
        if max(g1, key = int) > max(g2, key = int):
            mx = max(g1, key = int)
            mn = max(g2, key = int)
        else:
            mx = max(g2, key = int)
            mn = max(g1, key = int)
        
        return mx, mn
    
    #Crossover function
    def crossover(p1, p2):
        #Get the length of the two genes
        p1_genes = len(p1.linkDict) + len(p1.nodeDict)
        p2_genes = len(p2.linkDict) + len(p2.nodeDict)
        
        #Get the actual genomes
        p1_genome = evolution.get_genome(p1.linkDict, p1.nodeDict)
        p2_genome = evolution.get_genome(p2.linkDict, p2.nodeDict)
        
        #Create the child dictionaries
        ch_linkDict = dict()
        ch_nodeDict = dict()

        mx, g1_max, g2_max = evolution.get_max(p1_genome, p2_genome)
         
        #Compare fitness scores
        if p1.a_fitness == p2.a_fitness:
            #check if the genes are identically long
            if p1_genes == p2_genes:
    
                #Pick one at random
                pick = np.random.randint(0, 2)
                if pick == 0:
                    worst = p2_genome
                    best = p1_genome
                else:
                    worst = p1_genome
                    best = p2_genome
            
            #If the genes are different, get the shorter one
            else:
                if p1_genes > p2_genes:
                    worst = p1_genome
                    best = p2_genome
                elif p1_genes < p2_genes:
                    worst = p2_genome
                    best = p1_genome
        
        #If the fitness scores are different, get the larger one
        else:
            if p1.a_fitness > p2.a_fitness:
                worst = p2_genome
                best = p1_genome
            elif p1.a_fitness < p2.a_fitness:
                worst = p1_genome
                best = p2_genome
        
        #Cycle through all the available innovations
        for key in range(1, mx):
            #If the innovation exists in both genomes
            if best.get(key, None) != None and \
            worst.get(key, None) != None:
                #Pick one at random
                pick = np.random.randint(0, 2)
                if pick == 0:
                    origin = best[key]
                else:
                    origin = worst[key]
        
                #Assign the link or node to the appropriate child dictionary
                if best[key].s_type == "link":
                    ch_linkDict[origin.linkID] = origin
                elif best[key].s_type == "node":
                    ch_nodeDict[origin.nodeID] = origin
                    
            #If the innovation only exists in the better one
            elif best.get(key, None) != None and \
            worst.get(key, None) == None:
                #assign it to the appropriate dictionary
                if best[key].s_type == "link":
                    ch_linkDict[best[key].linkID] = best[key]
                elif best[key].s_type == "node":
                    ch_nodeDict[best[key].nodeID] = best[key]
         
            #Flush the input/output connections and rebuild them with new values
            for link in range(1, (max(ch_linkDict, key = int) + 1)):
                ch_nodeDict[ch_linkDict[link].inp_n].i_links.clear()
                ch_nodeDict[ch_linkDict[link].inp_n].o_links.clear()

                ch_nodeDict[ch_linkDict[link].out_n].i_links.clear()
                ch_nodeDict[ch_linkDict[link].out_n].o_links.clear()

                ch_nodeDict[ch_linkDict[link].inp_n].o_links.append(ch_linkDict[link].linkID)
                ch_nodeDict[ch_linkDict[link].out_n].i_links.append(ch_linkDict[link].linkID)
                                
        return ch_nodeDict, ch_linkDict
    
    #Speciation of two genomes
    def speciation(g, l):
        #Tuning Variable
        cDjoint = 1
        cExcess = 1
        cMatch = 0.4
        
        #Get the length of the geonmes
        g_genes = len(g.linkDict) + len(g.nodeDict)
        l_genes = len(l.linkDict) + len(l.nodeDict)

        #Get the actual genomes
        g_genome = evolution.get_genome(g.linkDict, g.nodeDict)
        l_genome = evolution.get_genome(l.linkDict, l.nodeDict)
        
        #Determine the longer gene
        if g_genes > l_genes:
            long = g_genome
        elif g_genes < l_genes:
            long = l_genome
                    
        #Maximum innovation number
        mx, mn = evolution.get_max(g_genome, l_genome)
        
        #Sub-Scores
        numDisjoint = 0
        numExcess = 0
        numMatch = 0
        diffWeight = 0
        
        #Cycle through all the available innovations
        for key in range(1, mx):
            #If the innovation exists in both genomes
            if g_genome.get(key, None) != None and \
            l_genome.get(key, None) != None:
                #Increased match counter
                numMatch += 1
                
                #If the gene is a link, get weight difference
                if g_genome[key].s_type == "link":
                    diffWeight += abs(g_genome[key].w - l_genome[key].w)
            
            #Check if the gene is missing from either genome
            elif g_genome.get(key, None) != None or \
            l_genome.get(key, None) != None:
                #If the innovation number is only in one, count as excess
                if g_genome.get(key, None) > mn or \
                l_genome.get(key, None) > mn:
                    numExcess += 1
    
                #If the innovation number is in both, count as disjoint
                if g_genome.get(key, None) <= mn or \
                l_genome.get(key, None) <= mn:
                    numDisjoint += 1
        
        #Get the actuall species score
        score = numDisjoint * cDjoint / long + \
                numExcess * cExcess / long + \
                diffWeight * cMatch / numMatch
        
        return score
            