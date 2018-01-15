import numpy as np
import parts as p
import math
import neuralnet
from operator import itemgetter

class evolution(object):
    def spawn(pop, specDict, specList, sameSpecies, specCount):
        #Threshold to assign genomes into the same species
        threshold = sameSpecies
        inp_out = [pop[0].i, pop[0].o]
        
        #Link of the new generation of genomes
        new_gen = []
        
        #Number of leaders
        n_leaders = 0
        
        #Get the leaders into the new generation
        for ind, species in enumerate(specList):
            #If there are genomes to spawn
            if specDict[species].spawn >= 1:
                #Add leader
                new_gen.append(pop[specDict[species].leader])
                
                #Increase leader count
                n_leaders += 1
                
                #Reduce the to be spawned number by the added leader
                specDict[species].spawn -= 1
                
        #Iterate throught the species list
        for ind, species in enumerate(specList):     
            #Get number of scores
            sc_len = len(specDict[species].adjScore)

            #Get number of children to be spawned minus the leader that has been
            #added before the loop
            to_spawn = math.ceil(specDict[species].spawn)

            #While there are children to be spawned
            while to_spawn > 0:
                to_spawn -= 1
                
                #If there are multiple genomes                
                if sc_len > 1:
                    #Set initial parents to None
                    p1 = None
                    p2 = None

                    #Get parents that are different
                    while p1 == p2:
                        p1 = \
                        np.random.randint(low = 0, 
                                          high = sc_len
                                          )
                        p2 = \
                        np.random.randint(low = 0, 
                                          high = sc_len
                                          )
                        
                #If there is only one genome, the leader
                elif sc_len == 1:
                    #Select the same species
                    p1 = 0
                    p2 = 0

                g1 = specDict[species].adjScore[p1][0]
                g2 = specDict[species].adjScore[p2][0]
                
                #Assigned to a species of not
                assigned = False
                
                #Spawn new genome from selected parents
                new_genome = neuralnet.Neuralnet(inp_out[0], inp_out[1])
                new_genome.nodeDict, new_genome.linkDict = \
                evolution.crossover(pop[g1], pop[g2])

                #Compare the new genome to the best genomes of each species
                for leader in range(0, n_leaders):
                    sc = evolution.speciation(new_genome, new_gen[leader])
                    
                    #If the score is within the tolerance, assign it to the
                    #species
                    if sc < threshold:
                        new_genome.species = new_gen[leader].species
                        assigned = True
                
                #Check whether the genome has been assigned to a species
                #If not, create a new species
                if assigned == False:
                    specCount += 1
                    new_genome.species = specCount

                #Append the new genome to the new generation
                new_gen.append(new_genome)

        #Reduce the new generation to the size of the previous generation
        #By removing random elements or adding new ones

        while len(new_gen) != len(pop):
            if len(new_gen) > len(pop):
                darwin = np.random.randint(low = 0,
                                           high = len(new_gen)
                                           )
                new_gen.pop(darwin).__del__()
            elif len(new_gen) < len(pop):
                new_gen, specCount = evolution.add_genomes(pop,
                                                           new_gen,
                                                           specCount,
                                                           inp_out[0],
                                                           inp_out[1],
                                                           threshold,
                                                           n_leaders
                                                           )
        
        return new_gen, specCount
    
    #In case there aren't wnough genomes produced
    def add_genomes(pop, new_gen, specCount, i, o, threshold, n_leaders):
        #Set initial parents to None
        g1 = np.random.randint(low = 0, high = len(pop))
        g2 = np.random.randint(low = 0, high = len(pop))

        #Get parents that are different
        while g1 == g2 and pop[g1].species == pop[g2].species:
            g1 = np.random.randint(low = 0, high = len(pop))
            g2 = np.random.randint(low = 0, high = len(pop))
            
            #Assigned to a species of not
            assigned = False
            
            #Spawn new genome from selected parents
            new_genome = neuralnet.Neuralnet(i, o)
            new_genome.nodeDict, new_genome.linkDict = \
            evolution.crossover(pop[g1], pop[g2])
            
            #Compare the new genome to the best genomes of each species
            for leader in range(0, n_leaders):
                sc = evolution.speciation(new_genome, new_gen[leader])
                
                #If the score is within the tolerance, assign it to the
                #species
                if sc < threshold:
                    new_genome.species = new_gen[leader].species
                    assigned = True
                if assigned == False:
                    specCount += 1
                    new_genome.species = specCount

                #Append the new genome to the new generation
                new_gen.append(new_genome)
        
        return new_gen, specCount
        
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
            specDict[species].spawn = 0
            specDict[species].number = 0
            
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
                if specDict[species].impEpoch == 15 and len(specList) > 0:
                    specDict.pop(species).__del__()
                    
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
            
            #See if there are species with 0 memebers
            if specDict[species].number == 0:
                del specDict[species]
                del specList[ind]
                
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
    def get_genome(linkDict, nodeDict):
        #Create a list for the innovation numbers in the genome
        ginnDict = dict()
        mx, mn = evolution.get_max(linkDict.keys(), nodeDict.keys())
        
        for i in range(1, (mx + 1)):
            if nodeDict.get(i, None) != None:
                ginnDict[nodeDict[i].innID] = nodeDict[i] 
            if linkDict.get(i, None) != None:
                ginnDict[linkDict[i].innID] = linkDict[i] 
            
        return ginnDict
    
    #Check for the latest innovation in the genome
    def get_max(g1, g2):   
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

        mx, mn = evolution.get_max(p1_genome, p2_genome)
         
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
        for key in range(1, (mx + 1)):
            #If the innovation exists in the best gene
            if best.get(key, None) != None:
                #If the structure is a link
                if best[key].s_type == "link":
                    #If the link exists in both genomes
                    if best.get(key, None) != None and \
                    worst.get(key, None) != None:
                        #If the link is enabled in one, disabled in the other
                        if (best[key].enabled == True and \
                        worst[key].enabled == False) or \
                        (best[key].enabled == False and \
                        worst[key].enabled == True):
                            
                            #Pick one at random
                            pick = np.random.randint(0, 2)
                            
                            if pick == 0:
                                origin = best[key]
                            elif pick == 1:
                                origin = worst[key]
                                
                        #if not, pick the best genome
                        else:
                            #Pick the best                
                            origin = best[key]

                        #Assign the link or node to the appropriate child dictionary         
                        ch_linkDict[origin.innID] = p.Link(origin.innID,
                                                           origin.inp_n,
                                                           origin.out_n,
                                                           origin.recurr
                                                           )
                        ch_linkDict[origin.innID].set_vars(origin.w,
                                                           origin.enabled
                                                           )
 
                elif best[key].s_type == "node":
                    #Pick the best                
                    origin = best[key]

                    ch_nodeDict[origin.innID] = p.Node(origin.innID,
                                                       origin.nodeID,
                                                       origin.n_type,
                                                       origin.recurr
                                                       )
                    ch_nodeDict[origin.innID].set_vars(origin.splitY,
                                                       origin.value,
                                                       )
                    
                    for ind, link in enumerate(list(origin.i_links.keys())):
                        ch_nodeDict[origin.innID].i_links[link] = link

                    for ind, link in enumerate(list(origin.o_links.keys())):
                        ch_nodeDict[origin.innID].o_links[link] = link
        
        ch_linkDict, ch_nodeDict = evolution.flush(ch_linkDict, ch_nodeDict)
        
        return ch_nodeDict, ch_linkDict

    #Flush the node connections
    def flush(linkDict, nodeDict):
        #Flush the input/output connections
        for ind, node in enumerate(list(nodeDict.keys())):
            nodeDict[node].i_links.clear()
            nodeDict[node].o_links.clear()

            nodeDict[node].i_links.clear()
            nodeDict[node].o_links.clear()
        #Rebuild them with new values
        for ind, link in enumerate(list(linkDict.keys())):
            nodeDict[linkDict[link].inp_n].o_links[linkDict[link].innID] = linkDict[link].innID
            nodeDict[linkDict[link].out_n].i_links[linkDict[link].innID] = linkDict[link].innID
        
        return linkDict, nodeDict
    
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
        if g_genes >= l_genes:
            long = len(g_genome)
        elif g_genes < l_genes:
            long = len(l_genome)
                    
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
               
                if key > mn or \
                key > mn:
                    numExcess += 1
    
                #If the innovation number is in both, count as disjoint
                if key <= mn or \
                key <= mn:
                    numDisjoint += 1
        
        #Get the actuall species score
        score = numDisjoint * cDjoint / long + \
                numExcess * cExcess / long + \
                diffWeight * cMatch / numMatch
        
        return score
            