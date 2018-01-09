import numpy as np

class mcu(object):
    def __init__(self):
        pass
    
    def get_genome(self, linkDict, nodeDict):
        #Create a list for the innovation numbers in the genome
        ginnDict = dict()
        
        #Get the innovation numbers for
        for i in range(1, len(linkDict) + 1):
            ginnDict[linkDict[i].innNum] = linkDict[i] 
            
        for i in range(1, len(nodeDict) + 1):
            ginnDict[nodeDict[i].innNum] = nodeDict[i] 
            
        return ginnDict
    
    def crossover(self, p1, p2):
        #Get the length of the two genes
        p1_genes = len(p1.linkDict) + len(p1.nodeDict)
        p2_genes = len(p2.linkDict) + len(p2.nodeDict)
        
        #Get the actual genomes
        p1_genome = self.get_genome(p1.linkDict, p1.nodeDict)
        p2_genome = self.get_genome(p2.linkDict, p2.nodeDict)
        
        #Create the child dictionaries
        child_linkDict = dict()
        child_nodeDict = dict()

        #Check for the latest innovation in the genome
        if max(p1_genome, key = int) > max(p2_genome, key = int):
            mx = max(p1_genome, key = int)
        else:
            mx = max(p2_genome, key = int)
         
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
                    child_linkDict[origin.linkID] = origin
                elif best[key].s_type == "node":
                    child_nodeDict[origin.nodeID] = origin

            #If the innovation only exists in the better one
            elif best.get(key, default = None) != None and \
            worst.get(key, default = None) == None:
                #assign it to the appropriate dictionary
                if best[key].s_type == "link":
                    child_linkDict[best[key].linkID] = best[key]
                elif best[key].s_type == "node":
                    child_nodeDict[best[key].nodeID] = best[key]
                
        return child_nodeDict, child_linkDict
    
    def
    