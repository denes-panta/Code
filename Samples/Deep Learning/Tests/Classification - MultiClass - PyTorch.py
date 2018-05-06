import torch
from torch import nn, autograd
from torch.nn import functional as F
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split as split

#Import the data
dfDataX = pd.DataFrame(datasets.load_wine()["data"])
dfDataX.columns = datasets.load_wine()["feature_names"]
dfDataY = pd.DataFrame(datasets.load_wine()["target"])
dfDataY.columns = ["class"]

#Data munging
dfDataXY = dfDataX.join(dfDataY)

dfTraVal, dfTest = split(dfDataXY, train_size = 0.8, stratify = dfDataXY["class"])
dfTrain, dfValid = split(dfTraVal, train_size = 0.75, stratify = dfTraVal["class"])
vDim = [dfTrain.iloc[:, 0:-1].shape[1], dfTrain["class"].nunique()]

del dfDataX, dfDataY, dfTraVal

#Create input data
tTrainX = torch.from_numpy(dfTrain.iloc[:, 0:-1].as_matrix())
tTrainY = torch.from_numpy(dfTrain.iloc[:, -1].as_matrix())
tValidX = torch.from_numpy(dfValid.iloc[:, 0:-1].as_matrix())
tValidY = torch.from_numpy(dfValid.iloc[:, -1].as_matrix())
tTestX = torch.from_numpy(dfTest.iloc[:, 0:-1].as_matrix())
tTestY = torch.from_numpy(dfTest.iloc[:, -1].as_matrix())

#Putting tensors on the GPU
if torch.cuda.is_available():
    tTrainX = tTrainX.cuda().type(torch.FloatTensor)
    tTrainY = tTrainY.cuda().type(torch.LongTensor)
    tValidX = tValidX.cuda().type(torch.FloatTensor)
    tValidY = tValidY.cuda().type(torch.LongTensor)

vTrainX = autograd.Variable(tTrainX)
vValidX = autograd.Variable(tValidX)
vTrainY = autograd.Variable(tTrainY)
vValidY = autograd.Variable(tValidY)

#Parameters
i_size = vDim[0]
h1_size = 100
o_size = vDim[1]

#Create the Neural Network
class Model(nn.Module):
    def __init__(self, i_size, h1_size, o_size):
        super().__init__()
        self.h1 = nn.Linear(i_size, h1_size)
        self.h2 = nn.Linear(h1_size, o_size)
        
    def forward(self, x):
        x = self.h1(x)
        x = F.relu(x)
        x = self.h2(x)
        x = F.softmax(x)
        return x
    
net = Model(i_size = i_size, h1_size = h1_size, o_size = o_size)
net.zero_grad()
net.parameters()
out = net(vTrainX)
print("out", out)
