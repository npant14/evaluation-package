from rdm import RDM
from bs import BrainScore
from korean import Korean

class Evaluation():
    def __init__(self, model, input_size, outdir, device='cpu'):
        self.model = model
        self.outdir = outdir
        self.device = device
        self.input_size = input_size

    def setup_RDM():
        ## input file locations as args
        pass
    
    def setup_Brainscore():
        ## input file locations as args
        pass

    def setup_Korean(self, data_dir):
        ## input file locations as args
       self.korean = Korean(self.model, self.outdir, self.device, data_dir, self.input_size) 

    def setup_Pasupathy():
        ## input file locations as args
        pass