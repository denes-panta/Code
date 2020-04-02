import numpy as np

class Node(object):
    def __init__(self, innID, nodeID, typ, rec):
        #Innovation ID
        self.innID = innID
        
        #Node ID
        self.nodeID = nodeID
        
        #Structure type
        self.s_type = "node"
        
        #Node Type
        self.n_type = typ
        
        #Value
        if typ == "bias":
            self.value = 1
        else:
            self.value = 0.0
                
        #Recurrent or not
        self.recurr = rec
        
        #Position in the nework grid
        self.splitX = None
        
        if typ == "input" or typ == "bias":
            self.splitY = 0
        elif typ == "output":
            self.splitY = 1
        
        #Links into the neuron
        self.i_links = dict()
        
        #Links out of the neuron
        self.o_links = dict()
    
    def set_vars(self, spY, val):
        # Y Coordinate
        self.splitY = spY

        # Value
        self.value = val

    def get_varaibles(self):
        variables = list()
        variables.append(self.innID)
        variables.append(self.nodeID)
        variables.append(self.s_type)
        variables.append(self.n_type)
        variables.append(self.value)
        variables.append(self.rec)
        variables.append(self.splitY)
        variables.append(self.i_links)
        variables.append(self.o_links)
        
        return variables
        
class Link(object):
    def __init__(self, innID, inp, out, rec):
        #Innovation ID
        self.innID = innID
        
        #Structure type
        self.s_type = "link"
        
        #Input
        self.inp_n = inp
        
        #Output
        self.out_n = out
        
        #Weight
        self.w = np.random.randn()
        
        #Enabled or not
        self.enabled = True
        
        #Recurrent or not
        self.recurr = rec

    def set_vars(self, wght, enab):
        # Y Coordinate
        self.w = wght

        # Value
        self.enabled = enab
        
    def get_varaibles(self):
        variables = list()
        variables.append(self.innID)
        variables.append(None)
        variables.append(self.s_type)
        variables.append(self.n_type)
        variables.append(self.value)
        variables.append(self.rec)
        variables.append(self.splitY)
        variables.append(self.i_links)
        variables.append(self.o_links)
        
        return variables

class Species(object):
    def __init__(self):
        #Species number
        self.specNum = None
        
        #Paragon of the species
        self.leader = None
        
        #Leader score
        self.lead_sore = None
        
        #Epoch since improvement
        self.impEpoch = 0
        
        #Historical best score
        self.histBest = 0
        
        #Number of speciements to spawn
        self.spawn = 0
        
        #Number of species
        self.number = 0
        
        #List of genomes and scores of the species
        #Structure: [[genes, adj fitness score]]
        self.adjScore = []
    
    def __del__(self):
        pass
    