import numpy as np

class node(object):
    def __init__(self, innov, typ):
        self.inn_id = innov
        self.s_type = "neuron"
        self.n_type = typ
        self.fr = -1
        self.to = -1
    
    def sigmoid(self, x):
        y = 1/(1+np.exp(-x))    
        return y
    
class link(object):
    def __init__(self, innov, inp, out):    
        self.inn_id = innov
        self.s_type = "link"
        self.n_type = "none"
        self.fr = inp
        self.to = out
        self.w = np.random.randn(0, 1)
        self.enabled = True
        if inp == out:
            self.recurrent = True
        else:
            self.recurrent = False