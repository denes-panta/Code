# Import libraries
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords 

import gensim.downloader as api

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

import os
import pickle
import re
import csv
import random
import datetime as dt

import json
import numpy as np
from sklearn.model_selection import train_test_split as split

# Change directory
os.chdir(os.path.dirname(os.path.realpath('__file__')))

# Import custom classes
from network import ConvNetwork
from dataloader import SampleLoader

# Chatbot class
class brobot(object):
    
    # Initialization
    def __init__(self, forceReTrain = False):
        # Set seed
        random.seed(101)
        
        # Download the necessary nltk libraries
        # nltk.download('stopwords')
        nltk.download('wordnet')
        
        # Create tokenizer and lemmatizer objects
        self.tokenizeRegEx = RegexpTokenizer(r'\w+')
        self.lemmaWNet = WordNetLemmatizer()
        
        # define the word2vec model for query embedding
        self.model_w2v = api.load('word2vec-google-news-300')
        
        # Load the stopwords
        # self.lSW = set(stopwords.words('english')) 

        # If the pre-processed data exists, load them
        if os.path.exists('data\\raw.pickle') and forceReTrain == False:
            self.loadRawData()
            self.loadEncodedData()
        
        # If not, load the json file, pre-process and save the data
        elif not os.path.exists('data\\enc.pickle') or forceReTrain == True:
            self.loadNewData()
            self.preProcessData()
            self.saveRawData()         
            self.saveEncodedData()        
        
        # definte the model
        self.defineModel()
        
        # If a trained model exists, load it
        if os.path.exists('model\\brain.model') and forceReTrain == False:
            self.loadModel()
        
        # If not, create one
        elif not os.path.exists('model\\brain.model') or forceReTrain == True:
            self.createModel()
        
        # Run the chatbot engine
        self.engine()
        
    # Load new data from a json file
    def loadNewData(self):
        with open('data\\intents.json', 'r') as file:
            self.dIntents = json.load(file)
        
        file.close()
    
    # Save the processed raw data via pickle
    def saveRawData(self):
        with open('data\\raw.pickle', 'wb') as file:
            pickle.dump((
                self.lLabels,
                self.lDocs_x, 
                self.lDocs_y,
                self.dResponses,
                self.iMaxIntentLen),
                file
                )
            
        file.close()
    
    # Save the encoded data via pickle
    def saveEncodedData(self):
        with open('data\\enc.pickle', 'wb') as file:
            pickle.dump((self.lInput, self.lOutput), file)
            
        file.close()
    
    # Load pre-processed raw data
    def loadRawData(self):
        with open('data\\raw.pickle', 'rb') as file:
            self.lLabels, \
            self.lDocs_x, self.lDocs_y, \
            self.dResponses, self.iMaxIntentLen = pickle.load(file)

        file.close()
    
    # Load encoded data
    def loadEncodedData(self):
        with open('data\\enc.pickle', 'rb') as file:
            self.lInput, self.lOutput = pickle.load(file)

        file.close()
    
    # Encode a document / query
    def encodeDoc(self, doc, predict = False):
        lWords = doc

        # If the document will be used for prediction, run pre-processing
        if predict == True:
            lWords = self.tokenizeRegEx.tokenize(doc)
            lWords = map(str.casefold, lWords)
            lWords = map(self.lemmaWNet.lemmatize, lWords)
            #lWords = filter(lambda w: w not in self.lSW, lWords)
            lWords = list(lWords)

        lEncoding = []
        
        # if the word is in the dictionary, encoded it
        for w in lWords:
            if w in self.model_w2v:
                lEncoding.append(self.model_w2v[w])
        
        # if the encoded query is smaller than the largest query, pad it
        iPadding = self.iMaxIntentLen - len(lEncoding)
        
        if iPadding < self.iMaxIntentLen:
            for i in range(iPadding):
                lEncoding.append([0] * len(lEncoding[0]))

        lEncoding = np.asarray(lEncoding).transpose()
        
        return lEncoding
    
    # Pre-process the data
    def preProcessData(self):
        # unique labels 
        self.lLabels = []
        
        # the queries
        self.lDocs_x = []
        
        # the intents
        self.lDocs_y = []
        
        # dictionary for the intents and responses
        self.dResponses = {}
        
        # max default length of an intent
        self.iMaxIntentLen = 20
        
        # iterate through the intents in the json file
        for intent in self.dIntents['intents']:
            
            # tokenize each entry in the patterns tag
            lWords = map(self.tokenizeRegEx.tokenize, intent['patterns'])
            
            # lowercase each word
            lWords = [list(map(str.casefold, w)) for w in lWords]
            
            # lemmatize each words
            lWords = [list(map(self.lemmaWNet.lemmatize, w)) for w in lWords]
            
            # filter out the stop words
            #lWords = [list(filter(lambda w: w not in self.lSW, l)) for l in lWords]
            
            # filter out the empty queries
            lWords = [l for l in lWords if l != []]
            
            # get the maximum intent length in words
            iMaxIntentLen = max(map(len, lWords))
            
            # if it is larger than the default, update it
            if self.iMaxIntentLen < iMaxIntentLen:
                self.iMaxIntentLen = iMaxIntentLen
            
            # add the intents to the queries            
            self.lDocs_x.extend(lWords)
            
            # add the corresponding intent tags to the predicted list
            self.lDocs_y.extend([intent['tag']] * len(lWords))
            
            # create and entry for the intent and response
            self.dResponses[intent['tag']] = intent['responses']
        
        # get unique list of intents
        self.lLabels = list(set(self.lDocs_y))
        self.lLabels = sorted(self.lLabels)

        # embed the queries
        self.lInput = [self.encodeDoc(d, self.model_w2v) for d in self.lDocs_x]
        
        # create mapping for intent and category numbers and convert the labels
        dLabelMap = {n : i for i, n in enumerate(self.lLabels)}                
        self.lOutput = [dLabelMap[l] for l in self.lDocs_y]
        
        # Filter out empty inputs
        lNEmp = [False if len(l) == 0 else True for l in self.lInput]
        self.lInput = [l[1] for l in zip(lNEmp, self.lInput) if l[0] == True]
        self.lOutput = [l[1] for l in zip(lNEmp, self.lOutput) if l[0] == True]
        
        # Convert the inputs and outputs to numpy arrays
        self.lInput = np.array(self.lInput)
        self.lOutput = np.array(self.lOutput)

    # Train the model
    def trainModel(self, model, criterion, optimizer, train_gen, **kwargs):
        # Devide to run the training on
        sDevice = kwargs['device']
        
        # number of epoch
        iEpoch = kwargs['epoch']
        
        # training loss
        fTrainLoss = 0.0
        
        # put the model into training mode
        model.train()
        
        # Run the model on each batch
        for iBatchId, (queries, labels) in enumerate(train_gen):
            queries, labels = queries.to(sDevice), labels.to(sDevice)

            # Run the forward pass
            outputs = model(queries)
            loss = criterion(outputs, labels.type(torch.LongTensor))
    
            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # add the losses from each batch together
            fTrainLoss += loss.item()

        # calculate average loss for the epoch
        fTrainLoss /= len(train_gen)
        
        # print out the loss and the epoch
        if iEpoch % 5 == 4:
            return('Epoch: %d, Train - loss: %.3f' %
                  (iEpoch + 1, fTrainLoss)
                  )

    # Evaluate the model
    def evalModel(self, model, criterion, test_gen, **kwargs):
        # Test loss
        fTestLoss = 0.0
        
        # Number of correctly predicted intents
        fCorrect = 0.0
        
        # device to run the model on
        sDevice = kwargs['device']
        
        # number of epoch
        iEpoch = kwargs['epoch']
        
        # put the model into validation mode
        model.eval()
        
        # deactivate the autograd engine. 
        with torch.no_grad():
            
            # Run the model on each batch
            for queries, labels in test_gen:
                queries, labels = queries.to(sDevice), labels.to(sDevice)
                
                # Make predictions
                outputs = model(queries)
                
                # Calculate the loss
                loss = criterion(outputs, labels.type(torch.LongTensor))
                fTestLoss += loss.item()
                
                # Compare the prediction to the label
                pred = outputs.argmax(dim = 1, keepdim = False)
                fCorrect += pred.eq(labels.view_as(pred)).sum().item()
        
        # Calculate the loss and prediction accuracy
        fTestLoss /= len(test_gen.dataset)
        fCorrect /= len(test_gen.dataset)

        # print out the status
        if iEpoch % 5 == 4:
            return('Val - loss: %.3f, Correct: %.2f' %
                  (fTestLoss, fCorrect)
                  )
    
    # Define the model
    def defineModel(self):
        # number of classes in the data
        self.iNumClasses = len(self.lLabels)
        self.model = ConvNetwork(self.iNumClasses)
        
    # create the model
    def createModel(self):
        # Define config parameters
        # Number of epoch
        iNumEpochs = 50
        
        #Shuffle the data or not
        bShuffle = True
        
        # batch size
        iBatchSize = 5
        
        # initial learning rate
        iLR = 0.001
        
        # convert the inputs and outputs to tensors
        self.lInput = torch.from_numpy(self.lInput)
        self.lOtuput = torch.from_numpy(self.lOutput)
        
        # create ids for the training and validation sets
        lTrain_ids, lTest_ids, _, _ = split(
            range(len(self.lOutput)), 
            self.lOutput,
            random_state = 101,
            stratify = self.lOutput
            )
        
        # Create the train loader/generator
        train_loader = SampleLoader(lTrain_ids, self.lOutput, self.lInput)
        train_generator = torch.utils.data.DataLoader(
            train_loader,
            batch_size = iBatchSize,
            shuffle = bShuffle
            )
        
        # Create the test loader/generator
        test_loader = SampleLoader(lTest_ids, self.lOutput, self.lInput)
        test_generator = torch.utils.data.DataLoader(test_loader)        
        
        # Check for cuda/gpu availablility
        bUseCuda = torch.cuda.is_available()
        sDevice = torch.device('cuda:0' if bUseCuda else 'cpu')
        
        # Loss, optimizer and scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = iLR)
        scheduler = StepLR(optimizer, step_size = 10, gamma = 0.1)
        
        # train and validate the model for each epoch
        for iEpoch in range(iNumEpochs):
            trainResult = self.trainModel(
                self.model, 
                criterion, 
                optimizer,
                train_generator, 
                device = sDevice,
                epoch = iEpoch
                )
            
            testResult = self.evalModel(
                self.model, 
                criterion, 
                test_generator, 
                device = sDevice,
                epoch = iEpoch
                )
            
            # print the status
            if iEpoch % 5 == 4:
                print(trainResult + '   ' + testResult)
            
            scheduler.step()
        
        # Save the model at the end
        torch.save(self.model.state_dict(), 'model\\brain.model')
        
        print('Finished Training')
            
    # Load he model
    def loadModel(self):
        self.model.load_state_dict(torch.load('model\\brain.model'))

    # Chatbot engine        
    def engine(self):
        # Variable to hold the truck information
        lTruck = None
        
        # Create a file for the conversation
        sDateTime = dt.datetime.utcnow().strftime('%Y-%m-%d_%H_%M_%S.%f')[:-3]
        sName = 'brobot_conv_' + sDateTime + '.txt'
        fileRecord = open('conversations\\' + sName, 'w')
        
        # Print out a welcome message
        sWelcome_1 = 'Bro-bot: Welcome to Truck Help-desk, I\'m Bro-bot.' + '\n'
        fileRecord.write(sWelcome_1)
        print(sWelcome_1)

        sWelcome_2 = '(Type \"quit\" to end the conversation).' + '\n' + '\n'
        fileRecord.write(sWelcome_2)        
        print(sWelcome_2)
        
        # Run the chatbot engine
        while True:
            # Get the input from the user
            inp = input('You: ')
            
            # Write it to the file
            fileRecord.write('You: ' + inp + '\n')
            
            # If the user types quit, close the program
            if inp.lower() == 'quit':
                # Close the conversation file and break the loop
                fileRecord.close()
                break
            
            # Check if there is a licence plate number in the input
            rgxMatch = re.search('\(([^\)]+)\)', inp)

            # if there is a plate number in the input, 
            # look up the plate number in the csv file
            if rgxMatch is not None:
                # Clean the match
                sMatch = rgxMatch.group(0)
                sMatch = sMatch.replace('(', '')
                sMatch = sMatch.replace(')', '')

                # Set the found truck to None
                lTruck = None
                
                with open('data\\truck_data.csv', 'r') as file_truck:
                    reader = csv.reader(file_truck, delimiter = ";")
                    
                    # Try to find the queried truck                    
                    for i, line in enumerate(reader):
                        if line[0] == sMatch:
                            lTruck = line
                            break
                    
                    # If found, comminicate the user's options
                    if lTruck != None:                            
                        sFoundTruck = 'Bro-bot: I can provide you with the location, fleet number or cargo of the truck.'
                        sFoundTruck += '\n'
                        sFoundTruck += '\n'
                        fileRecord.write(sFoundTruck)
                        print(sFoundTruck)
                         
                    # If there is no match, communicate the bad news
                    else:
                        sNotFound = 'Bro-bot: The licence plate is unkwnown, please try another.'          
                        sNotFound += '\n' + '\n'
                        fileRecord.write(sNotFound)
                        print(sNotFound)
            
            # if no plate number found, encode the query, and make intent prediction
            else:           
                # Encode the user-input for prediction
                lUserInput = self.encodeDoc(inp, predict = True)
                
                # If the encoded user-input is empty, or larger than 20, 
                # communicate that the request was not understood
                if lUserInput.size == 0 or lUserInput.shape[1] > self.iMaxIntentLen:
                    sError = 'Bro-bot: Sorry, I did not understand your request.'
                    sError += '\n'
                    fileRecord.write(sError)
                    print(sError)
                
                else:
                    # Otherwise encode the query
                    lUserInput = lUserInput[np.newaxis, :, :]
                    lUserInput = torch.from_numpy(lUserInput)
                    lUserOutput = self.model(lUserInput)
                    
                    # Identify the intent
                    lUserOutput = lUserOutput.detach().numpy()
                    iResult = np.argmax(lUserOutput)
                    sTag = self.lLabels[iResult]
                    
                    # If there is no licence plate input yet
                    if lTruck == None:
                        if sTag == 'general':
                            lResponses = self.dResponses[sTag]
                            sResponse = 'Bro-bot: ' + random.choice(lResponses) 
                            sResponse += '\n'
                            
                        elif sTag == 'greeting':
                            lResponses = self.dResponses[sTag]
                            sResponse = 'Bro-bot: ' + random.choice(lResponses)
                            sResponse += '\n'
                            
                        else:
                            sResponse = 'Bro-bot: I need a licence plate number for that.' 
                            sResponse += '\n'
                            
                    # if there is alread a plate number:
                    else:
                        # Answer based on the intent
                        if sTag == 'location':
                            lResponses = self.dResponses[sTag]
                            sResponse = 'Bro-bot: ' + random.choice(lResponses) 
                            sResponse += lTruck[5] + '\n'
                            
                        elif sTag == 'cargo':
                            lResponses = self.dResponses[sTag]
                            sResponse = 'Bro-bot: ' + random.choice(lResponses)
                            sResponse += lTruck[4] + '\n'
                            
                        elif sTag == 'specs':
                            sResponse = 'Bro-bot: The specs of ' + lTruck[0] + ' are: '
                            sResponse += 'Height: ' + lTruck[1] + '; '
                            sResponse += 'Width: ' + lTruck[2] + '; '
                            sResponse += 'Length: ' + lTruck[3] + '; '
                            sResponse += 'Fleet number: ' + lTruck[6] + '.'
                            sResponse += '\n' 
                        else:
                            lResponses = self.dResponses[sTag]
                            sResponse = random.choice(lResponses) + '\n'
                    
                    # write the response to the screen and into the file
                    fileRecord.write(sResponse + '\n')
                    print(sResponse)

if __name__ == '__main__':
    truckbot = brobot(forceReTrain = False)
