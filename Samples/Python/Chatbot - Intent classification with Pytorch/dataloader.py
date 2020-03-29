import torch
from torch.utils import data

class SampleLoader(data.Dataset):

  def __init__(self, IDs, labels, data):
        # Initialization
        self.lData = data
        self.lIDs = IDs        
        self.lLabels = labels

  def __len__(self):
        return len(self.lIDs)

  def __getitem__(self, index):
        iID = self.lIDs[index]
        
        # Get Sample and Label
        x = self.lData[iID]
        y = self.lLabels[iID]

        return x, y