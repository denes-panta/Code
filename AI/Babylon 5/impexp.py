import os, datetime
import pickle as pl

class IO(object):
    def __init__(self):
        pass
      
    def create_dir(p):
        dir_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        mydir = os.path.join(p, dir_name)
        os.makedirs(mydir)

        return mydir
    
    def imp():
        pass
    
    def exp(p, name, generation, innDict, pop, specList):
        gen_dir = os.path.join(p, generation)
        if os.path.exists(gen_dir) == False:
            os.makedirs(gen_dir)
        
        f_dir = os.path.join(gen_dir, name)
        os.makedirs(f_dir)

        output = open(f_dir + "/innov_dict", "wb")        
        pl.dump(innDict, output)
        output.close()

        for l in range(len(specList)):
            output = open(f_dir + "/link_dict" + str(l), "wb")        
            pl.dump(pop[l].linkDict, output)
            output.close()

            output = open(f_dir + "/node_dict" + str(l), "wb")        
            pl.dump(pop[l].nodeDict, output)
            output.close()
            
        