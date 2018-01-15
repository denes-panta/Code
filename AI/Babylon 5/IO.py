import pandas as pd
import os, datetime

class IO(object):
    def __init__(self):
        pass
     
    def create_dir(self, path):
        mydir = os.path.join(path, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(mydir)
        
    def imp(self):
        pass
    
    def exp(self):
        pass