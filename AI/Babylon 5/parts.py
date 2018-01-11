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
        
        #Activation curvature
        self.act = np.random.rand(0, 1)
        
        #Position in the nework grid
        self.splitX = None
        
        if typ == "input" or typ == "bias":
            self.splitY = 0
        elif typ == "output":
            self.splitY = 1
        
        #Links into the neuron
        self.i_links = []
        
        #Links out of the neuron
        self.o_links = []
      
    
class Link(object):
    def __init__(self, innID, linkID, inp, out, rec):
        #Innovation ID
        self.innID = innID
        
        #Link ID
        self.linkID = linkID
        
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
        
        
class Species(object):
    def __init__(self):
        #Species number
        self.specNum = None
        
        #Alive or Dead
        self.alive = True
        
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
