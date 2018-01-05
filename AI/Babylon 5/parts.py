import numpy as np

class Node(object):
    def __init__(self, innID, nodeID, typ):
        #Innovation ID
        self.innID = innID
        
        #Node ID
        self.nodeID = nodeID
        
        #Structure type
        self.s_type = "node"
        
        #Node Type
        self.n_type = typ
        
        #Input
        self.value = 0.0
                
        #Recurrent or not
        self.recurr = False
        
        #Activation curvature
        self.act = np.random.rand(0, 1)
        
        #Position in the nework grid
        self.splitX = None
        if typ == "input":
            self.splitY = 0
        elif typ == "output":
            self.splitY = 1
        
        #Links into the neuron
        self.links = []
      
    
class Link(object):
    def __init__(self, innID, linkID, inp, out):
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
        self.recurr = False
