import os
import datetime
import pickle as pl
import math
import neuralnet as nn
import parts as p
import numpy as np
import mcu

class IO(object):
    def __init__(self):
        pass
      
    def create_dir(path):
        dir_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        mydir = os.path.join(path, dir_name)
        os.makedirs(mydir)

        return mydir
    
    #Import the saved generation
    def imp(path, pop_length, name, i, o):
        #New population for the imported genomes
        population = []
        
        #Get full path        
        imp_path = os.path.join(path, str(name))

        #Innovation dictionary and innovation number
        innovDict = pl.load(open(imp_path + "/innov_dict", "rb" ))
        innovNum = max(innovDict.keys(), key = int) 
        
        #Get nodeID
        nID = 0

        for ind, innov in enumerate(innovDict):
            if innovDict[innov].s_type == "node":
                if innovDict[innov].nodeID > nID:
                    nID = innovDict[innov].nodeID

        #Number of genome leaders saved
        specNum = len(os.listdir(imp_path))
        specList = []
        n = int((specNum - 1) / 2)

        #Number of genomes to be created by
        n_gen = int(math.floor(pop_length / n))

        #Creat the population from the saved files
        for lead in range(n):
            #Get nodeDict and linkDict
            i_linkDict = pl.load(open(imp_path + "/link_Dict" + str(lead), "rb" ))
            i_nodeDict = pl.load(open(imp_path + "/node_Dict" + str(lead), "rb" ))

            #Append speclist
            specList.append(lead + 1)
                        
            for gen in range(n_gen):
                #Create new genome
                genome = nn.Neuralnet(i, o)

                #Assign species number
                genome.species = lead + 1
                
                #Create link dictionary
                for ind, l_ID in enumerate(i_linkDict):
                    link = i_linkDict[l_ID]

                    genome.linkDict[l_ID] = p.Link(link.innID,
                                                   link.inp_n,
                                                   link.out_n,
                                                   link.recurr
                                                   )
                    genome.linkDict[l_ID].set_vars(link.w,
                                                   link.enabled
                                                   )
                
                #Create node dictionary
                for ind, n_ID in enumerate(i_nodeDict):
                    node = i_nodeDict[n_ID]
                    genome.nodeDict[n_ID] = p.Node(node.innID,
                                                   node.nodeID,
                                                   node.n_type,
                                                   node.recurr
                                                   )
                    genome.nodeDict[n_ID].set_vars(node.splitY,
                                                   node.value,
                                                   )
                
                genome.linkDict, genome.nodeDict = \
                mcu.evolution.flush(genome.linkDict, genome.nodeDict)
                
                population.append(genome)
                
        #If the new population is smaller then the required one,
        #fill it up with random ones
        while len(population) < pop_length:
            #Get random leader
            lead = np.random.randint(low = 0, high = n)
            
            #Create new genome
            genome = nn.Neuralnet(i, o)

            #Assign species number
            genome.species = lead

            #Get linkDict and nodeDict
            i_linkDict = pl.load(open(imp_path + "/link_Dict" + str(lead), "rb" ))
            i_nodeDict = pl.load(open(imp_path + "/node_Dict" + str(lead), "rb" ))

            #Create link dictionary
            for ind, l_ID in enumerate(i_linkDict):
                link = i_linkDict[l_ID]

                genome.linkDict[l_ID] = p.Link(link.innID,
                                               link.inp_n,
                                               link.out_n,
                                               link.recurr
                                               )
                genome.linkDict[l_ID].set_vars(link.w,
                                               link.enabled
                                               )
                
            #Create node dictionary
            for ind, n_ID in enumerate(i_nodeDict):
                node = i_nodeDict[n_ID]
                genome.nodeDict[n_ID] = p.Node(node.innID,
                                               node.nodeID,
                                               node.n_type,
                                               node.recurr
                                               )
                genome.nodeDict[n_ID].set_vars(node.splitY,
                                               node.value,
                                               )

            genome.linkDict, genome.nodeDict = \
            mcu.evolution.flush(genome.linkDict, genome.nodeDict)
            
            population.append(genome)

        return population, innovDict, innovNum, nID, specList, specNum
    
    #Export the generations
    def exp(path, name, generation, innDict, pop, specList):
        gen_dir = os.path.join(path, generation)
        if os.path.exists(gen_dir) == False:
            os.makedirs(gen_dir)
        
        f_dir = os.path.join(gen_dir, name)
        os.makedirs(f_dir)

        output = open(f_dir + "/innov_dict", "wb")        
        pl.dump(innDict, output)
        output.close()

        for l in range(len(specList)):
            output = open(f_dir + "/link_dict" + str(l), "wb")        
            pl.dump(pop[l].linkDict, output)
            output.close()

            output = open(f_dir + "/node_dict" + str(l), "wb")        
            pl.dump(pop[l].nodeDict, output)
            output.close()
            
        