#Libraries
import pandas as pd
#Deep Learning with pytorch
import torch
from torch import optim
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os

#Loads the data
def loadData(cfg):
    transform = transforms.ToTensor()
    
    #Set seed from Config file
    torch.manual_seed(cfg.randomSeed)
    
    #Set batch size
    batchSize = cfg.trainBatchSize
    
    #Create train loader
    trainData = datasets.MNIST('data', 
                               train = True, 
                               download = True, 
                               transform = transform
                               ) 
    
    trainLoader = DataLoader(trainData,
                             batch_size = batchSize, 
                             shuffle = True
                             )
    
    #Create test loader
    testData = datasets.MNIST('data', 
                              train = False, 
                              download = True, 
                              transform = transform
                              ) 


    testLoader = DataLoader(testData,
                            batch_size = batchSize
                            )

    return trainLoader, testLoader
    

#Neural Network algorithm
def Network(cfg, trainLoader, testLoader = None, mode = "train"):
    epochs = cfg.epoch
    
    #Evaluation function
    def evaluate(testLoader, mode):
        #Mode of the algorithm
        if mode == "train":
            predsY = []
        elif mode == "validate":
            lossSum = 0
            accSum = 0
        else:
            raise ValueError('Not supported mode')    
        
        #Run the model on the test set
        for i, (x, label) in enumerate(testLoader):
            if torch.cuda.is_available():
                x, label = x.cuda(), label.cuda()
            
            x, label = Variable(x), Variable(label)

            preds = net(x)
            
            predicted = preds.data.max(1)[1]
            
            #Modifiy the variable based on the mode
            if mode == "validate":
                #Calculate the loss
                loss = F.cross_entropy(preds, label)
                lossSum += loss.data[0]
                #Calulate the accuracy
                acc = predicted.eq(label.data).cpu().sum()
                accSum += acc
            elif mode == "train":
                predsY.append(predicted)
                
        #Return the KPI or the prediction
        if mode == "validate":
            return lossSum / len(testLoader), accSum / len(testLoader)
        elif mode == "train":
            return pd.DataFrame(predsY)
    
    #Neural Network Class
    class Net(nn.Module):
        
        #Neural network structure
        def __init__(self):
            super(Net, self).__init__()
            self.m = nn.MaxPool2d(kernel_size = 2, stride = 1, padding = 0)

            self.c1 = nn.Conv2d(1, 64, kernel_size = 3, stride = 1, padding = 1)
            self.d1 = nn.Dropout2d(p = 0.625)

            self.c2 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
            self.d2 = nn.Dropout2d(p = 0.625)

            self.c3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
            self.d3 = nn.Dropout2d(p = 0.625)

            self.c4 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
            self.d4 = nn.Dropout2d(p = 0.625)

            self.fc1 = nn.Linear(64 * 24 * 24, 128)
            self.d5 = nn.Dropout2d(p = 0.625)

            self.fc2 = nn.Linear(128, 64)
            self.d6 = nn.Dropout2d(p = 0.625)

            self.fc3 = nn.Linear(64, 10)
        
        #Forward propagation
        def forward(self, x):
            x = F.relu(self.m(self.c1(x)))
            x = self.d1(x)

            x = F.relu(self.m(self.c2(x)))
            x = self.d2(x)

            x = F.relu(self.m(self.c3(x)))
            x = self.d3(x)

            x = F.relu(self.m(self.c4(x)))
            x = self.d4(x)

            x = x.view(-1, 64 * 24 * 24)

            x = F.relu(self.fc1(x))
            x = self.d5(x)

            x = F.relu(self.fc2(x))
            x = self.d6(x)

            y_hat = self.fc3(x)
            
            return y_hat

    #Define the object and try to put it on the GPU
    net = Net()

    if torch.cuda.is_available():
        net.cuda()

    #Define the optimizer and the loss function
    optimizer = optim.SGD(net.parameters(), lr = cfg.lr, momentum = cfg.mom)
    funcLoss = torch.nn.CrossEntropyLoss()
    
    #Print the header based on the mode
    if mode == "train":
        print("\n" + "Training:")
    elif mode == "validate":
        print("\n" + "Validating:")
    
    #Training Epochs
    for epoch in range(epochs):
        lossSum = 0
        accSum = 0
        #Run the model on the train set
        for i, (x, label) in enumerate(trainLoader):

            if torch.cuda.is_available():
                x, label = x.cuda(), label.cuda()
                
            x, label = Variable(x), Variable(label)
            
            #Predict
            preds = net(x)
            #Calculate loss
            loss = funcLoss(preds, label)
            lossSum += loss.data[0]
            #Zero the gradient
            optimizer.zero_grad()
            #Backprop
            loss.backward()
            #Step with the optimizer
            optimizer.step()
        
            #Calulate accuracy
            predicted = preds.data.max(1)[1]
            acc = predicted.eq(label.data).cpu().sum()
            accSum += acc

        #Print every n epochs
        accSum /= len(trainLoader)
        lossSum /= len(trainLoader)
        
        #Print the epoch information
        if epoch % 1 == 0:
            if mode == "validate":
                valLoss, valAcc = evaluate(testLoader, mode)
                print(("Epoch: %d >> Train: Acc - %.3f - Loss: %.3f ||" \
                      % ((epoch + 1), accSum, lossSum)) + \
                       ("Test: Acc - %.3f - Loss: %.3f" \
                      % (valAcc, valLoss)))
            elif mode == "train":
                print("Epoch: %d >> Train: Acc - %.3f - Loss: %.3f" \
                      % ((epoch + 1), accSum, lossSum))

    #If it was training, return the predictions
    if mode == "train":
        predsY = evaluate(testLoader, mode)
        return predsY

#Main
if __name__ == '__main__':
    os.chdir("F:\Code\Code\Samples\Deep Learning - PyTorch\Image Classification")

    from config import Config
    config = Config()
    trainLoader, testLoader = loadData(config)
    
    Network(config, trainLoader, testLoader, mode = "validate")
    preds = Network(config, trainLoader, testLoader, mode = "train")
