import torch
from torch import nn
import torch.nn.functional as F

class ConvNetwork(nn.Module):
    def __init__(self, num_classes):
        super(ConvNetwork, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels = 300, 
            out_channels = 64, 
            kernel_size = 2,
            padding = 0
            )

        self.fc1 = nn.Linear(
            in_features = 64 * 19,
            out_features = 256
            )

        self.fc2 = nn.Linear(
            in_features = 256,
            out_features = 128
            )
 
        self.out = nn.Linear(
            in_features = 128,
            out_features = num_classes)
                
    def forward(self, x):
        x = x.float()
 
        x = self.conv1(x)
        x = F.relu(x)

        x = x.reshape(-1, 64 * 19)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.out(x)
        x = F.softmax(x, dim = 1)

        return x
    


    
