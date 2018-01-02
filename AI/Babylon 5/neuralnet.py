class neuralnet(object):
    
    def __init__(self, input_data):
        self.data = input_data
        
        self.input_layer = []
        self.output_layer = []
        self.hidden_layer = []
        self.innovTable = []
