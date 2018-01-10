import numpy as np
import parts as p

class mcu(object):
    def __init__(self):
        pass
    
    def epoch(self):
        pass        
    
    def get_cards(self, pop):
        #Dictionary for the species data
        species = dict()
        species[specNum] = p.Species()
    
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
    def crossover(self, p1, p2):
        #Get the length of the two genes
        p1_genes = len(p1.linkDict) + len(p1.nodeDict)
        p2_genes = len(p2.linkDict) + len(p2.nodeDict)
        
        #Get the actual genomes
        p1_genome = self.get_genome(p1.linkDict, p1.nodeDict)
        p2_genome = self.get_genome(p2.linkDict, p2.nodeDict)
        
        #Create the child dictionaries
        ch_linkDict = dict()
        ch_nodeDict = dict()

        mx, g1_max, g2_max = self.get_max(p1_genome, p2_genome)
         
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
            if best.get(key, default = None) != None and \
            worst.get(key, default = None) != None:
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
            elif best.get(key, default = None) != None and \
            worst.get(key, default = None) == None:
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
    
    def speciation(self, g, l, cDjoint, cExcess, cMatch):
        #Get the length of the geonmes
        g_genes = len(g.linkDict) + len(g.nodeDict)
        l_genes = len(l.linkDict) + len(l.nodeDict)

        #Get the actual genomes
        g_genome = self.get_genome(g.linkDict, g.nodeDict)
        l_genome = self.get_genome(l.linkDict, l.nodeDict)
        
        #Determine the longer gene
        if g_genes > l_genes:
            long = g_genome
        elif g_genes < l_genes:
            long = l_genome
                    
        #Maximum innovation number
        mx, mn = self.get_max(g_genome, l_genome)
        
        #Sub-Scores
        numDisjoint = 0
        numExcess = 0
        numMatch = 0
        diffWeight = 0
        
        #Cycle through all the available innovations
        for key in range(1, mx):
            #If the innovation exists in both genomes
            if g_genome.get(key, default = None) != None and \
            l_genome.get(key, default = None) != None:
                #Increased match counter
                numMatch += 1
                
                #If the gene is a link, get weight difference
                if g_genome[key].s_type == "link":
                    diffWeight += abs(g_genome[key].w - l_genome[key].w)
            
            #Check if the gene is missing from either genome
            elif g_genome.get(key, default = None) != None or \
            l_genome.get(key, default = None) != None:
                #If the innovation number is only in one, count as excess
                if g_genome.get(key, default = None) > mn or \
                l_genome.get(key, default = None) > mn:
                    numExcess += 1
    
                #If the innovation number is in both, count as disjoint
                if g_genome.get(key, default = None) <= mn or \
                l_genome.get(key, default = None) <= mn:
                    numDisjoint += 1
        
        #Get the actuall species score
        score = numDisjoint * cDjoint / long + \
                numExcess * cExcess / long + \
                numMatch * diffWeight / numMatch
        
        return score
        
        
    