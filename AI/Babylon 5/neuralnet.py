import parts as p
import scipy as sc
import numpy as np
import math

class Neuralnet(object):
    
    def __init__(self, i, o):
        #Output vector
        self.output = []
        
        #Number of input/output nodes
        self.i = i
        self.o = o
        
        #Number of shots omitted out of 5
        self.so = 4
                
        #Node ID counter
        self.nodeID = 0
        self.linkID = 0
        
        #Dictionaries for the nodes and links
        self.nodeDict = dict()
        self.linkDict = dict()
        
        #Fitness scores
        self.r_fitness = None
        self.a_fitness = None
        
        #Speciation score
        self.spec_score = None
        
        #Species it belongs to
        self.species = None
        
    def create_net(self, innDict, innNum):
        #Check if any innovations exists
        if innNum == 0:
            cr = True
        else:
            cr = False
        
        #Nodes, links are not recurrent
        rec = False
        
        #Create the input nodes
        for ind in range(self.i):
            innDict, innNum = self.create_node("input", innDict, innNum, rec, cr)
            
        #Create the output nodes
        for ind in range(self.o):
            innDict, innNum = self.create_node("output", innDict, innNum, rec, cr)
        
        #Create the bias node
            innDict, innNum = self.create_node("bias", innDict, innNum, rec, cr)
                
        inp_node = 1
        
        #Create the connections from the input nodes
        while self.nodeDict[inp_node].n_type == "input" and \
        inp_node <= (self.i - self.so):
            #To each output nodes
            for out_node in range(inp_node, (len(self.nodeDict) + 1)):
                if self.nodeDict[out_node].n_type == "output":
                    innDict, innNum = self.create_link(inp_node, 
                                                       out_node, 
                                                       innDict, 
                                                       innNum,
                                                       rec,
                                                       cr 
                                                       )
                    
                    #Append the LinkId to the node for futre reference
                    self.nodeDict[inp_node].o_links.append(self.linkID)
                    self.nodeDict[out_node].i_links.append(self.linkID)
            inp_node += 1

        return innDict, innNum
    
    def create_node(self, typ, innDict, innNum, rec = False, innov = True):
        #Increase node counter
        self.nodeID +=1
        #Add node to node dictionary
        self.nodeDict[self.nodeID] = p.Node(innNum,
                                            self.nodeID,
                                            typ,
                                            rec
                                            )

        #If the link is an innovation
        if innov == True:
            #Increase innovation counter
            innNum += 1
            #Add node to innovation dictionary
            innDict[innNum] = p.Node(innNum,
                                     self.nodeID,
                                     typ,
                                     rec
                                     )

        return innDict, innNum

    def create_link(self, inp, out, innDict, innNum, rec = False, innov = True):
        #Increase the link counter
        self.linkID += 1
        #Add link to node dictionary
        self.linkDict[self.linkID] = p.Link(innNum, 
                                            self.linkID, 
                                            inp, 
                                            out,
                                            rec
                                            )
        
        #If the link is an innovation
        if innov == True:
            #Increase innovation number
            innNum += 1
            #Add link to innovation dictionary
            innDict[innNum] = p.Link(innNum, 
                                     self.linkID, 
                                     inp, 
                                     out,
                                     rec
                                     )
        
        return innDict, innNum
    
    #Get innovationID for links and nodeID for Nodes
    def get_innovation(self, structure, innDict, nl1 = None, nl2 = None):
        #Check links
        if structure == "link":
            for entry in range(1, (len(innDict) + 1)):
                #Check only the defined structures
                if innDict[entry].s_type == structure:

                    #Check for origin and destination node
                    if innDict[entry].inp_n == nl1 and \
                    innDict[entry].out_n == nl2:
                        innDict[entry].innID
                    else:
                        return None
        #Check nodes
        elif structure == "node":
            for entry in range(1, (len(innDict) + 1)):
                #Check only the defined structures
                if innDict[entry].s_type == structure:

                    #Check for input and output link
                    if nl1 in innDict[entry].i_links == True and \
                    nl2 in innDict[entry].o_links == True:
                        return innDict[entry].nodeID
                    else:
                        return None
    
    #Check if the link already exists
    def duplicate_link(self, node_1, node_2):
        result = False

        for link in range(1, (len(self.linkDict) + 1)):
            if self.linkDict[link].inp_n == node_1 and \
            self.linkDict[link].out_n == node_2:
                result = True

        return result

    def add_link(self, mutaProb, loopProb, numLoopTry, 
                 numLinkTry, innDict, innNum):
        #Check for mutation probability
        if np.random.random() <= mutaProb:        
            #Initiate the node variables
            n_1 = None
            n_2 = None
            
            #Set recursive variable to False
            rec = False
            
            #Set max iterations for finding a node without Loop or existing Link
            findLoop = numLoopTry
            findLink = numLinkTry

            #Check if a looped link is to be created
            if np.random.random() <= loopProb:
                while findLoop >= 0:

                    findLoop -= 1
                    #Get random node
                    n = np.random.randint(1, (len(self.nodeDict) + 1))
                    
                    #Check if it is not recurrent, bias or input
                    if self.nodeDict[n].recurr == False and \
                    self.nodeDict[n].n_type != "input" and \
                    self.nodeDict[n].n_type != "bias":
                        n_1 = n_2 = n
                        self.nodeDict[n].recurr = True
                        rec = True
                        #Close the loop
                        findLoop = 0
                    else:      
                        return innDict, innNum

            #Check if a new link has to be created
            else:
                while findLink >= 0:

                    findLink -= 1
                    #Get destination node
                    n_2 = np.random.randint(1, (len(self.nodeDict) + 1))
                    
                    #Check if it is bias or input node
                    if self.nodeDict[n_2].n_type != "bias" or \
                    self.nodeDict[n_2].n_type != "input":
                        #If not, get origin node
                        n_1 = np.random.randint(1, (len(self.nodeDict) + 1))

                        #Check if the connection is duplicate, or if it loop
                        if n_1 != n_2 and \
                        self.duplicate_link(n_1, n_2) == False:
                            #Get direction of the connection

                            if self.nodeDict[n_1].splitY > self.nodeDict[n_2].splitY:
                                rec = True

                            #Close the loop
                            findLink = 0
                            
            #If no valid link has been found, return
            if n_1 == None:                
                return innDict, innNum

            #Check to see if the innovation has already been discovered
            if self.get_innovation("link", innDict, n_1, n_2) == None:
                innov = False
            else:
                innov = True
                
            #Create new link
            innDict, innNum = \
            self.create_link(n_1, n_2, innDict, innNum, rec, innov)
            
            #Append the linkID to the origin and destination nodes
            self.nodeDict[n_1].o_links.append(self.linkID)
            self.nodeDict[n_2].i_links.append(self.linkID)

        return innDict, innNum
            
    def add_node(self, mutaProb, numOldLink, innDict, innNum):    
        #Check for mutation probability    
        if np.random.random() <= mutaProb:
            #New link found
            done = False
            
            #New node
            l = None
            
            #Tries to find link
            findLink = numOldLink
            
            #Node Threshold
            node_th = self.o + self.i + 5
            
            #If the genome is small
            if (len(self.nodeDict) + 1) < node_th:
                while findLink > 0:
                    findLink -= 1
                    
                    #Choose an older link randomly
                    l = np.random.randint(0, \
                                          len(self.linkDict) - \
                                          1 - \
                                          int(math.sqrt((len(self.linkDict) + 1)))
                                          )
                    
                    #Make sure that it is not a bias link, ecurrent or disabled
                    if self.linkDict[l].enabled == True and \
                    self.linkDict[l].recurr == False and \
                    self.nodeDict[self.linkDict[l].inp_n].n_type != "bias":
                        #Flag it if found
                        done = True
                        
                        #Close the loop
                        findLink = 0
            
                if done == False:
                    return innDict, innNum
                
            #If the genome is large enough
            else:
                while done == False:
                    #Choose any link randomly
                    l = np.random.randint(1, (len(self.linkDict) + 1))

                    #Make sure that it is not a bias link, recurrent or disabled
                    if self.linkDict[l].enabled == True and \
                    self.linkDict[l].recurr == False and \
                    self.nodeDict[self.linkDict[l].inp_n].n_type != "bias":
                        #Flag it if found
                        done = True
                        
                        #Close the loop
                        findLink = 0  
            
            #Disable the link
            self.linkDict[l].enabled = False
            
            #Get it's weight, Y position, origin and destination
            weight = self.linkDict[l].w
            n_from = self.linkDict[l].inp_n
            n_to = self.linkDict[l].out_n
            depthY = (self.nodeDict[n_from].splitY + self.nodeDict[n_to].splitY)
            depthY = depthY / 2
            
            #See if the innovation exists in the innovation table
            nodeID = self.get_innovation("node", innDict, n_from, n_to)
            
            #Check if the innovation exists globally and in the genome
            if nodeID != None and nodeID in self.nodeDict.keys():
                nodeID = None
            
            #If the innovation doesn't exist
            if nodeID == None:
                #Create the new node
                innDict, innNum = \
                self.create_node("hidden", innDict, innNum, False, True)
                self.nodeDict[self.nodeID].splitY = depthY
                
                #Create the first link
                innDict, innNum = \
                self.create_link(n_from, self.nodeID, innDict, innNum, False, True)
                #Append the LinkId to the node for futre reference
                self.nodeDict[n_from].o_links.append(self.linkID)
                self.nodeDict[self.nodeID].i_links.append(self.linkID)
                
                #Create the second link
                innDict, innNum = \
                self.create_link(self.nodeID, n_to, innDict, innNum, False, True)
                
                #Append the LinkId to the node for futre reference
                self.nodeDict[(self.nodeID)].o_links.append(self.linkID)
                self.nodeDict[n_to].i_links.append(self.linkID)
                self.linkDict[self.linkID].w = weight
            
            #If the innovation already exists
            else:
                #Create the new node
                innDict, innNum = \
                self.create_node("hidden", innDict, innNum, False, False)
                
                #Create the first link
                innID = self.get_innovation("link", innDict, n_from, self.nodeID)
                innDict, innNum = \
                self.create_link(n_from, self.nodeID, innDict, innID, False, False)
                
                #Append the LinkId to the node for futre reference
                self.nodeDict[n_from].o_links.append(self.linkID)
                self.nodeDict[self.nodeID].i_links.append(self.linkID)

                #Create the second link
                innID = self.get_innovation("link", innDict, self.nodeID, n_to)
                innDict, innNum = \
                self.create_link(self.nodeID, n_to, innDict, innID, False, False)
                
                #Append the LinkId to the node for future reference
                self.nodeDict[self.nodeID].o_links.append(self.linkID)
                self.nodeDict[n_to].i_links.append(self.linkID)
                self.linkDict[self.linkID].w = weight

        return innDict, innNum
    
    def mutate_weight(self, mutaProb, replcProb, muta_type):
        #Do severe mutation
        if np.random.random() > 0.5:
            severe = True
        else:
            severe = False
            
        #Get number of links
        num_links = len(self.linkDict) + 1
        new_links = int(num_links * 0.8)
        
        #Iterate through the links
        for link in range(1, num_links):
            if severe == True:
                gausspoint = 0.7
                coldgausspoint = 0.9
            elif num_links >= 10 and link > new_links:
                gausspoint = 0.5
                coldgausspoint = 0.7
            else:
                if np.random.random() > 0.5:
                    gausspoint = mutaProb
                    coldgausspoint = replcProb + 0.1
                else:
                    gausspoint = mutaProb
                    coldgausspoint = replcProb
            
            if muta_type == "GAUSS":
                if np.random.random() < gausspoint:
                    self.linkDict[link].w += np.random.randn()
                elif np.random.random() < coldgausspoint:
                    self.linkDict[link].w += np.random.randn()
                    
            elif muta_type == "COLD":
               self.linkDict[link].w = np.random.randn() 
            

    def get_depth(self):
        return 1

    def update(self, input_data, method):
        #Check the method and define the repetition cycle number
        if method == "snapshot":
            flushCount = self.get_depth()
        elif method == "active":
            flushCount = 1
        
        for i in range(flushCount):
            #Clear the output vector
            self.output.clear()
            node = 1
            
            #Set the input node's data with the input_data
            while self.nodeDict[node].n_type == "input":
                self.nodeDict[node].value = input_data[node - 1]
                node += 1
            
            #Iterate through the rest of the nodes
            while node < (len(self.nodeDict) + 1):
                n_sum = 0

                #Get the links leading into the node
                for ind, link in enumerate(self.nodeDict[node].i_links):
                    #Check if the link is enabled
                    if self.linkDict[link].enabled == True:
                        #Take the weight of each link and multiply it with the
                        #connected node's value
                        n_sum += self.linkDict[link].w * \
                                 self.nodeDict[self.linkDict[link].inp_n].value                

                #Set the value of the node with the activated n_sum value
                self.nodeDict[node].value = sc.special.expit(n_sum)

                #If it is an output node, put the value into the output list
                if self.nodeDict[node].n_type == "output":
                    self.output.append(int(self.nodeDict[node].value))            
                node += 1
        
        if method == "snapshot":
            for node in range(1, (len(self.nodeDict) + 1)):
                self.nodeDict[node].value = 0
        
        return self.output