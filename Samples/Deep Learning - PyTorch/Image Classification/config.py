class Config(object):

    def __init__(self):
        self.randomSeed = 117
        
        self.trainBatchSize = 50
        self.testBatchSize = 50
        
        self.epoch = 100
        
        self.lr = 0.01
        self.mom = 0.9