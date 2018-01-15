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
                        
        #Dictionaries for the nodes and links
        self.nodeDict = dict()
        self.linkDict = dict()
        
        #Fitness scores
        self.r_fitness = None
        self.a_fitness = None
        self.s_fitness = None
        
        #Speciation score
        self.spec_score = None
        
        #Species it belongs to
        self.species = 1
        
        #Age
        self.age = 1
    
    #Delete instance
    def __del__(self):
        pass
    
    #Create neural net
    def create_net(self, innDict, innNum, nodeID):
        #Zero the innovation and node counters
        innNum = 0
        nodeID = 0
        
        #Set innovation variable to True        
        cr = True
            
        #Nodes, links are not recurrent
        rec = False
        
        #Create the bias node
        innDict, innNum, nodeID = self.create_node("bias", 
                                                   innDict, 
                                                   innNum, 
                                                   nodeID, 
                                                   rec, 
                                                   cr
                                                   )
        
        #Create the input nodes
        for ind in range(self.i):
            innDict, innNum, nodeID = self.create_node("input", 
                                                       innDict, 
                                                       innNum, 
                                                       nodeID, 
                                                       rec, 
                                                       cr
                                                       )
            
        #Create the output nodes
        for ind in range(self.o):
            innDict, innNum, nodeID = self.create_node("output", 
                                                       innDict, 
                                                       innNum, 
                                                       nodeID, 
                                                       rec, 
                                                       cr
                                                       )
                        
        inp_node = 2
        
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
                    self.nodeDict[inp_node].o_links[innNum] = innNum
                    self.nodeDict[out_node].i_links[innNum] = innNum
                    
            inp_node += 1

        return innDict, innNum, nodeID

    #Create node
    def create_node(self, typ, innDict, innNum, 
                    nodeID, rec = False, innov = True):
        #Increase innNum during the creation of the populations
        if innov == True:
            innNum += 1
        
            #Increase node counter
            nodeID += 1

            #Add node to innovation dictionary
            innDict[innNum] = p.Node(innNum,
                                     nodeID,
                                     typ,
                                     rec
                                     )

        #Add node to node dictionary
        self.nodeDict[innNum] = p.Node(innNum,
                                       nodeID,
                                       typ,
                                       rec
                                       )

        return innDict, innNum, nodeID
    
    #Create link
    def create_link(self, inp, out, innDict, innNum, rec = False, innov = True):
        #Increase innNum during the creation of the populations
        if innov == True:
            innNum += 1

            #Add link to innovation dictionary
            innDict[innNum] = p.Link(innNum, 
                                     inp, 
                                     out,
                                     rec
                                     )

        #Add link to node dictionary
        self.linkDict[innNum] = p.Link(innNum, 
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
                        return innDict[entry].innID
                    else:
                        return None
        #Check nodes
        elif structure == "node":
            for entry in range(1, (len(innDict) + 1)):
                #Check only the defined structures
                if innDict[entry].s_type == structure:

                    #Check for input and output link
                    if nl1 in list(innDict[entry].i_links.keys()) == True and \
                    nl2 in list(innDict[entry].o_links.keys()) == True:
                        return innDict[entry].innID
                    else:
                        return None
    
    #Check if the link already exists
    def duplicate_link(self, node_1, node_2):
        result = False
        #Get max Link innovation
        mx = max(self.linkDict, key = int)
        #Loop through the innovation
        for link in range(1, (mx + 1)):
            #Check to see if the innovation exists
            if self.linkDict.get(link, None) != None:
                if self.linkDict[link].inp_n == node_1 and \
                self.linkDict[link].out_n == node_2:
                    result = True
                    return result

        return result

    #Add link
    def add_link(self, mutaProb, loopProb, numLoopTry,
                 numLinkTry, innDict, innNum):
        #Check for mutation probability
        if np.random.random() <= mutaProb:        

            #Initiate the node variables
            n_1 = None
            n_2 = None
            
            #Set recursive variable to False
            rec = False

            #Get max node innovation
            lst = list(self.nodeDict.keys())
            
            #Found a valid path
            found = False
            
            #Set max iterations for finding a node without Loop or existing Link
            findLoop = numLoopTry
            findLink = numLinkTry

            #Check if a looped link is to be created
            if np.random.random() <= loopProb:
                while findLoop > 0 and found == False:
                    
                    findLoop -= 1
                    
                    #Get random node                    
                    n_id = np.random.randint(0, len(lst))
                    n = lst[n_id]
                    
                    #Check if it is not recurrent, bias or input
                    if self.nodeDict[n].recurr == False and \
                    self.nodeDict[n].n_type != "input" and \
                    self.nodeDict[n].n_type != "bias":
                        n_1 = n_2 = n
                        self.nodeDict[n].recurr = True
                        rec = True
                        
                        #Close the loop
                        findLoop = 0
                        
                        #Set the found variable to True
                        found = True
                        
            #Check if a new link has to be created
            else:
                while findLink > 0 and found == False:
                    
                    findLink -= 1
                    #Get origin and destination nodes
                    n1_id = np.random.randint(0, len(lst))
                    n_1 = lst[n1_id]
                    
                    n2_id = np.random.randint(0, len(lst))
                    n_2 = lst[n2_id]
                    
                    #Do node checks and link check
                    if n_1 != n_2 and \
                    self.nodeDict[n_1].n_type != "output" and \
                    self.nodeDict[n_2].n_type != "bias" and \
                    self.nodeDict[n_2].n_type != "input" and \
                    self.duplicate_link(n_1, n_2) == False:
                        #check if it is recurrent                                
                        if self.nodeDict[n_1].splitY > self.nodeDict[n_2].splitY:
                            rec = True

                        #Close the loop
                        findLink = 0
                        
                        #Set the found variable to True
                        found = True
            
            #If no valid link has been found, return
            if found == False:                
                return innDict, innNum
            elif found == True:         
                
                #Check to see if the innovation has already been discovered
                innID = self.get_innovation("link", innDict, n_1, n_2)
                
                #If there is no such innovation, set the innID to the latest innovation
                if innID == None:
                    innov = True
                    innID = innNum
                else:
                    innov = False
                
                #Create new link
                innDict, innNum = \
                self.create_link(n_1, n_2, innDict, innID, rec, innov)
                self.nodeDict[n_1].o_links[innNum] = innNum
                self.nodeDict[n_2].i_links[innNum] = innNum
            
            return innDict, innNum
    
    #Add node
    def add_node(self, mutaProb, numOldLink, innDict, innNum, nodeID):    
        #Check for mutation probability    
        if np.random.random() <= mutaProb:
            #New link found
            done = False

            #Selected link
            l = None
            
            #Get max node innovation
            lst = list(self.linkDict.keys())
            
            #Tries to find link
            findLink = numOldLink
            
            #Node Threshold
            node_th = self.o + self.i + 5
            
            #If the genome is small
            if (len(self.nodeDict) + 1) < node_th:
                while findLink > 0:
                    findLink -= 1

                    #Choose an older link randomly
                    l_id = np.random.randint(0, \
                                             len(lst) - \
                                             1 - \
                                             int(math.sqrt(len(lst)))
                                             )
                    l = lst[l_id]

                    #Make sure that it is not a bias link, recurrent or disabled
                    if self.linkDict[l].enabled == True and \
                    self.linkDict[l].recurr == False and \
                    self.nodeDict[self.linkDict[l].inp_n].n_type != "bias":
                        #Flag it if found
                        done = True

                        #Close the loop
                        findLink = 0
            
                if done == False:
                    return innDict, innNum, nodeID
                
            #If the genome is large enough
            else:
                while done == False:
                    #Choose any link randomly
                    l_id = np.random.randint(0, len(lst))
                    l = lst[l_id]
                    
                    #Make sure that it is not a bias link, recurrent or disabled
                    if self.linkDict[l].enabled == True and \
                    self.linkDict[l].recurr == False and \
                    self.nodeDict[self.linkDict[l].inp_n].n_type != "bias" and \
                    self.nodeDict[self.linkDict[l].inp_n].n_type != "input":
                        #Flag it if found
                        done = True
            
            #Disable the link
            self.linkDict[l].enabled = False
            
            #Get it's weight, Y position, origin and destination
            weight = self.linkDict[l].w
            n_from = self.linkDict[l].inp_n
            n_to = self.linkDict[l].out_n
            depthY = (self.nodeDict[n_from].splitY + self.nodeDict[n_to].splitY)
            depthY = depthY / 2
            
            #See if the innovation exists in the innovation table
            innID = self.get_innovation("node", innDict, n_from, n_to)
            
            #Check if the node id of the innovation and the genome are the same
            if innID != None: 
                for ind, node in enumerate(self.nodeDict):
                    if node.nodeID == innDict[innID].nodeID:
                        innID = None

            #If there needs to be an innovation created
            if innID == None:
                #Create the new node
                innDict, innNum, nodeID = \
                self.create_node("hidden", innDict, innNum, nodeID, False, True)
                self.nodeDict[innNum].splitY = depthY
                n_innN = innNum
                
                #Create the first link
                innDict, innNum = \
                self.create_link(n_from, n_innN, innDict, innNum, False, True)
                
                #Append the LinkId to the node for futre reference
                self.nodeDict[n_from].o_links[innNum] = innNum
                self.nodeDict[n_innN].i_links[innNum] = innNum
                
                #Create the second link
                innDict, innNum = \
                self.create_link(n_innN, n_to, innDict, innNum, False, True)
                
                #Append the LinkId to the node for futre reference
                self.nodeDict[n_innN].o_links[innNum] = innNum
                self.nodeDict[n_to].i_links[innNum] = innNum
                self.linkDict[innNum].w = weight

            #If the innovation already exists
            else:
                #Create the new node
                innDict, innNum = \
                self.create_node("hidden", innDict, innID, False, False)
                n_innN = innID
                
                #Create the first link
                innID = self.get_innovation("link", innDict, n_from, n_innN)
                innDict, innNum = \
                self.create_link(n_from, n_innN, innDict, innID, False, False)
                
                #Append the LinkId to the node for futre reference
                self.nodeDict[n_from].o_links[innID] = innID
                self.nodeDict[n_innN].i_links[innID] = innID

                #Create the second link
                innID = self.get_innovation("link", innDict, n_innN, n_to)
                innDict, innNum = \
                self.create_link(n_innN, n_to, innDict, innID, False, False)
                
                #Append the LinkId to the node for future reference
                self.nodeDict[n_innN].o_links[innID] = innID
                self.nodeDict[n_to].i_links[innID] = innID
                self.linkDict[n_innN].w = weight

        return innDict, innNum, nodeID
    
    #Mutate the wiegths of the nodes
    def mutate_weight(self, mutaProb, replcProb, muta_type):
        #Do severe mutation
        if np.random.random() > 0.5:
            severe = True
        else:
            severe = False
            
        #Get number of links
        lst_links = list(self.linkDict.keys())
        num_links = len(lst_links)
        new_links = int(num_links * 0.8)
        
        #Iterate through the links
        for ind, link in enumerate(lst_links):
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
            
    #Get depth of the neural network
    def get_depth(self):
        return 1
    
    #Update neurak network
    def update(self, input_data, method):
        #Check the method and define the repetition cycle number
        if method == "snapshot":
            flushCount = self.get_depth()
        elif method == "active":
            flushCount = 1
        
        for i in range(flushCount):
            #Clear the output vector
            self.output.clear()
            
            for ind, node in enumerate(list(self.nodeDict.keys())):
            
                #Set bias value to 1
                if self.nodeDict[node].n_type == "bias":
                    self.nodeDict[node].value = 1
                #Set the input node's data with the input_data
                elif self.nodeDict[node].n_type == "input":
                    self.nodeDict[node].value = input_data[node - 2]
                    
                else:                    
                    n_sum = 0
                    #Get the links leading into the node
                    for ind, link in enumerate(list(self.nodeDict[node].i_links.keys())):
                        #Check if the link is enabled
#                        try:
#                            self.linkDict[link].enabled
#                        except:
#                            print("----------Error------------")
#                            print(self.nodeDict)
#                            print()
#                            print(self.linkDict)
#                            print()
#                            for ind, node in enumerate(self.nodeDict):
#                                print(self.nodeDict[node].innID, self.nodeDict[node].n_type, "inp: ", self.nodeDict[node].i_links.keys())
#                                print(self.nodeDict[node].innID, self.nodeDict[node].n_type, "out: ", self.nodeDict[node].o_links.keys())
#                            print("----------Error------------")

                        if self.linkDict[link].enabled == True:
                            #Take the weight of each link and multiply it with the
                            #connected node's value
                            n_sum += self.linkDict[link].w * \
                                     self.nodeDict[self.linkDict[link].inp_n].value                
    
                    #Set the value of the node with the activated n_sum value
                    self.nodeDict[node].value = int(sc.special.expit(n_sum))
    
                    #If it is an output node, put the value into the output list
                    if self.nodeDict[node].n_type == "output":
                        self.output.append(int(self.nodeDict[node].value))            
                    
        #If used in snapshot mode, set all variables
        if method == "snapshot":
            for node in range(1, (len(self.nodeDict) + 1)):
                self.nodeDict[node].value = 0
        
        return self.output