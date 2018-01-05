import parts as p
import numpy as np
class Neuralnet(object):
    
    def __init__(self, i, o, innDict, innNum):
        #Output vector
        self.output = []
        
        #Number of input/output nodes
        self.i = i
        self.o = o
        
        #Innovation number
        self.innNum = innNum
        self.innDict = innDict
        
        #Node ID counter
        self.nodeID = 0
        self.linkID = 0
        
        #Dictionaries for the nodes and links
        self.nodeDict = dict()
        self.linkDict = dict()
        
        #Fitness scores
        self.r_fitness = None
        self.a_fitness = None
        
        #Number of offsprings
        self.n_spawn = None
        
        #Create initial neural network
        self.create_net()
    
    def get_innNum(self):
        return self.innNum
    
    def get_innDict(self):
        return self.innDict    
    
    def sigmoid(self, x):
        y = 1/(1+np.exp(-x))    
        return y
    
    def duplicate_link(self):
        result = False
        
        return result
    
    def duplicate_node(self):
        result = False
        
        return result
    
    def get_node_pos(self):
        pass
    
    def create_net(self):
        #Check if any innovations exists
        if self.innNum == 0:
            cr = True
        else:
            cr = False
            
        #Create the input nodes
        for ind in range(self.i):
            self.add_node("input", cr)
        
        #Create the output nodes
        for ind in range(self.o):
            self.add_node("output", cr)
        
        inp_node = 0
        
        #Craete the connections between them
        while self.nodeDict[inp_node].n_type == "input":
            for out_node in range(inp_node, len(self.nodeDict)):
                if self.nodeDict[out_node].n_type == "output":
                    self.add_link(inp_node, out_node, cr)
                    self.nodeDict[out_node].links.append(self.linkID)
            inp_node += 1
        
    def add_node(self, typ, innov = True):
        if typ == "input" or typ == "output":
            #Add node to node dictionary
            self.nodeDict[self.nodeID] = p.Node(self.innNum,
                                                self.nodeID,
                                                typ
                                                )
            #Increase node counter
            self.nodeID +=1
        elif typ == "hidden":
            pass
        elif typ == "bias":
            pass

        #If the link is an innovation
        if innov == True:
            #Add node to innovation dictionary
            self.innDict[self.innNum] = p.Node(self.innNum,
                                               self.nodeID,
                                               typ
                                               )
            #Increase innovation counter
            self.innNum += 1

    def add_link(self, inp, out, innov = True):
        #Add link to node dictionary
        self.linkDict[self.linkID] = p.Link(self.innNum, self.linkID, inp, out)
        #Increase the link counter
        self.linkID += 1
        
        #If the link is an innovation
        if innov == True:
            #Add link to innovation dictionary
            self.innDict[self.innNum] = p.Link(self.innNum, self.linkID, inp, out)
            #Increase innovation number
            self.innNum += 1

    def get_depth(self):
        pass

    def update(self, input_data, method):

        if method == "snapshot":
            flushCount = self.get_depth()
        elif method == "active":
            flushCount = 1
        
        for i in range(flushCount):
            #Clear the output vector
            self.output.clear()
            node = 0
            
            #Set the input node's data with the input_data
            while self.nodeDict[node].n_type == "input":
                self.nodeDict[node].value = input_data[node]
                node += 1
            
            #Iterate through the rest of the nodes
            while node < len(self.nodeDict):
                n_sum = 0

                #Get the links leading into the node
                for link, ind in enumerate(self.nodeDict[node].links):

                    #Take the weight of each link and multiply it with the
                    #connected node's value
                    n_sum += self.linkDict[link].w * \
                             self.nodeDict[self.linkDict[link].inp_n].value                

                #Set the value of the node with the activated n_sum value
                self.nodeDict[node].value = self.sigmoid(n_sum)
                
                #If it is an output node, put the value into the output list
                if self.nodeDict[node].n_type == "output":
                    self.output.append(int(self.nodeDict[node].value))            
                node += 1
        
        if method == "snapshot":
            for node in range(len(self.nodeDict)):
                self.nodeDict[node].value = 0
        
        return self.output