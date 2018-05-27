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


if use_cuda:
    model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    # trainning
    ave_loss = 0
    for batch_idx, (x, target) in enumerate(train_loader):
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, target)
        ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train_loader):
            print '==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                epoch, batch_idx+1, ave_loss)
    # testing
    correct_cnt, ave_loss = 0, 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x, volatile=True), Variable(target, volatile=True)
        out = model(x)
        loss = criterion(out, target)
        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()
        # smooth average
        ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
        
        if(batch_idx+1) % 100 == 0 or (batch_idx+1) == len(test_loader):
            print '==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
epoch, batch_idx+1, ave_loss, correct_cnt * 1.0 / total_cnt)
