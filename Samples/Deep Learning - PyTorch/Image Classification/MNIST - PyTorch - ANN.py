#Data: https://www.kaggle.com/c/titanic/data
#Libraries
import pandas as pd
import numpy as np
#Deep Learning with pytorch
import torch
from torch import optim
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

#Neural Network algorithm
def NeuralNet(trainX, trainY, testX, testY = None, e = 100, mode = "train"):
    epochs = e
    
    def validate(testX, testY, mode):
        correct = 0
        if mode == "train":
            predsY = []
        elif mode == "validate":
            total = len(testY)
        
        for i, record in enumerate(testX):
            preds = net(record)
            _, predicted = torch.max(preds.data, 0)
            
            if mode == "validate":
                correct += (predicted[0] == testY[i])
            elif mode == "train":
                predsY.append(predicted)

        if mode == "validate":
            return (100 * correct / total)
        elif mode == "train":
            return pd.DataFrame(predsY)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.c1 = nn.Conv2D(1, 64, kernel_size = 3)
            self.d1 = nn.Dropout2d(p = 0.625)

            self.c2 = nn.Conv2D(64, 64, kernel_size = 3)
            self.d2 = nn.Dropout2d(p = 0.625)

            self.c3 = nn.Conv2D(64, 64, kernel_size = 3)
            self.d3 = nn.Dropout2d(p = 0.625)

            self.m = nn.MaxPool2D(2)

            self.fc1 = nn.Linear(64, 128)
            self.d4 = nn.Dropout2d(p = 0.625)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)
            
        def forward(self, x):
            x = F.relu(self.c1(x))
            x = self.d1(x)

            x = F.relu(self.c2(x))
            x = self.d2(x)

            x = F.relu(self.c3(x))
            x = self.d3(x)

            x = self.fc1(x)
            x = self.d4(x)

            x = self.fc2(x)
            x = self.fc3(x)

            y_hat = F.logsigmoid(self.fc3(x))
            
            return y_hat
        
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9)
    loss_func = torch.nn.CrossEntropyLoss()

    if mode == "train":
        print("\n" + "Training:")
    elif mode == "validate":
        print("\n" + "Validating:")
    
    for epoch in range(epochs):
        preds = net(trainX)
        loss = loss_func(preds, trainY)

        if epoch % 100 == 0:
            if mode == "validate":
                val = validate(testX, testY, mode)
                print("Epoch: %d - Train: %.3f - Val: %.3f" % ((epoch + 1), loss, val))
            elif mode == "train":
                print("Epoch: %d - Train: %.3f" % ((epoch + 1), loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if mode == "train":
        predsY = validate(testX, testY, mode)
        return predsY

#Script
batch_size = 50

trainLoader = torch.utils.data.DataLoader(
    datasets.MNIST('data', 
                   train = True, 
                   download = True, 
                   transform = transforms.ToTensor()
                   ), 
    batch_size=batch_size, 
    shuffle = True
    )

testLoader = torch.utils.data.DataLoader(
    datasets.MNIST('data', 
                   train = False, 
                   transform = transforms.ToTensor()
                   ),
    batch_size = 1000
    )


#Validate
NeuralNet(trainX = tValTrainX, 
          trainY = tValTrainY, 
          testX = tValTestX, 
          testY = tValTestY,
          e = 4000, 
          mode = "validate"
          )

#Train
tTestY = NeuralNet(trainX = tTrainX, 
                   trainY = tTrainY, 
                   testX = tTestX, 
                   e = 4000, 
                   mode = "train"
                   )
