# Input files: https://www.kaggle.com/c/invasive-species-monitoring/data

# Import libraries
import cv2
from scipy import ndimage

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import optimizers
from keras import regularizers

from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D

from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

# Change path to the location of the file
os.chdir(os.path.dirname(os.path.realpath('__file__')))

# Image classifier object
class ImageClassifier(object):
    
    def __init__(self):
        # Set seed
        np.random.seed(101)    

    # Get training labels
    def getLabels(self): #Read in the classes   
        # Read in the labels
        sLabPath = 'raw/train_labels/'
        dfLabels = pd.read_csv(sLabPath + 'train_labels.csv', sep = ',')
        
        # Extract the names
        self.lNames = dfLabels['name']
        # Extract the labels
        self.lLabels = dfLabels['invasive']
        
    def processTrain(self):
        # Create lists for the the processed names and labels
        self.lProcNames = []
        self.lProcLabels = []
        
        # Create threshold for the training-validation split
        iSplit = int(len(self.lNames) * 0.8)
        
        # Counter for progress repots
        iCounter = 0
        
        # Iterate through the files
        for n, c in zip(self.lNames, self.lLabels):
            
            img = cv2.imread('raw/train/' + str(n) + ".jpg", 1)
            
            # If the images are not rotated, rotate them back
            if img.shape[0] != 866 and img.shape[1] != 1154:
                 img = ndimage.interpolation.rotate(img, 90)
            
            # Crop the image into rwo images and resize
            imgCrL = cv2.resize(img[0:866, 0:866], (224, 224))            
            imgCrR = cv2.resize(img[0:866, 288:1154], (224, 224))
            
            # If the necessary folder don't exist, create them
            if not os.path.isdir(self.sPathTrain):
                os.mkdir(self.sPathTrain)

            if not os.path.isdir(self.sPathTrain + '1'):
                os.mkdir(self.sPathTrain + '1')

            if not os.path.isdir(self.sPathTrain + '0'):
                os.mkdir(self.sPathTrain + '0')

            if not os.path.isdir(self.sPathValid):
                os.mkdir(self.sPathValid)

            if not os.path.isdir(self.sPathValid + '1'):
                os.mkdir(self.sPathValid + '1')

            if not os.path.isdir(self.sPathValid + '0'):
                os.mkdir(self.sPathValid + '0')
            
            # Seprate the training samples into train/validation sets and categories
            if iCounter <= iSplit:
                if c == 1:
                    cv2.imwrite(self.sPathTrain + '1/' + str(n) + 'a.jpg', imgCrL)
                    cv2.imwrite(self.sPathTrain + '1/' + str(n) + 'b.jpg', imgCrR)
                else:
                    cv2.imwrite(self.sPathTrain + '0/' + str(n) + 'a.jpg', imgCrL)
                    cv2.imwrite(self.sPathTrain + '0/' + str(n) + 'b.jpg', imgCrR)
                    
            else:
                if c == 1: 
                    cv2.imwrite(self.sPathValid + '1/' + str(n) + 'a.jpg', imgCrL)
                    cv2.imwrite(self.sPathValid + '1/' + str(n) + 'b.jpg', imgCrR)   
                else:
                    cv2.imwrite(self.sPathValid + '0/' + str(n) + 'a.jpg', imgCrL)
                    cv2.imwrite(self.sPathValid + '0/' + str(n) + 'b.jpg', imgCrR)   
            
            # Create the new label and name lists
            self.lProcNames.append(str(n) + 'a')
            self.lProcNames.append(str(n) + 'b')
            
            self.lProcLabels.extend([c, c])     
            
            # Print out progress report
            iCounter += 1
            
            if (iCounter % 100 == 0): 
                print ('Train image number: ' + str(iCounter))
    
    # Pre-process the testing data
    def processTest(self):
        iTestN = len(os.listdir('raw/test'))
        
        if not os.path.isdir(self.sPathTest):
            os.mkdir(self.sPathTest)
        
        # Iterate through the samples, crop the middle part and resize
        for n in range(1, iTestN + 1):
            
            img = cv2.imread('raw/test/' + str(n) + ".jpg", 1)
            imgCropCenter = cv2.resize(img[0:866, 144:1010], (224, 224))
            cv2.imwrite(self.sPathTest + str(n) + '.jpg', imgCropCenter)
            
            if (n % 100 == 0): 
                print ('Test image number: ' + str(n))

    # Create the dataframe for the processed data        
    def processLabels(self):
        self.dfLabels = pd.DataFrame(
            {'names': self.lProcNames,
             'labels': self.lProcLabels})
        
        if not os.path.isdir(self.sPathLab):
            os.mkdir(self.sPathLab)
        
        self.dfLabels.to_csv(
            self.sPathLab + 'train_labels.csv', 
            sep = ',',
            index = False)
    
    # Run the pre-processing functions
    def preprocess(self):
        # Define the pathes
        self.sPathTrain = 'processed/train/'
        self.sPathValid = 'processed/valid/'
        self.sPathLab = 'processed/train_labels/'
        self.sPathTest = 'processed/test/'
        self.sPathModel = 'model/'
        self.sPathCheckPoints = 'checkpoints/'
        
        if not os.path.isdir('processed'):

            os.mkdir('processed')
            self.getLabels()
            self.processTrain()
            self.processLabels()
            self.processTest()            

    # Train the model - VGG (modified)  
    def defineModel(self):
        # Add some noise to the model
        iNoise = 0.1 * (1.0/255)
        
        # Model based on a modified VGG
        model = Sequential()
        model.add(GaussianNoise(iNoise, input_shape = (224, 224, 3)))
        
        model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same'))
        model.add(MaxPooling2D((2, 2), strides = (2 ,2)))
        model.add(BatchNormalization())
        
        model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
        model.add(MaxPooling2D((2, 2), strides = (2, 2)))
        model.add(BatchNormalization())
    
        model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
        model.add(MaxPooling2D((2, 2), strides = (2, 2)))
        model.add(BatchNormalization())
    
        model.add(Conv2D(256, (3, 3), activation = 'relu', padding = 'same'))
        model.add(MaxPooling2D((2, 2), strides = (2, 2)))
        model.add(BatchNormalization())
    
        model.add(Conv2D(512, (3, 3), activation = 'relu', padding = 'same'))
        model.add(MaxPooling2D((2, 2), strides = (2, 2)))
        model.add(BatchNormalization())

        model.add(Conv2D(1024, (3, 3), activation = 'relu', padding = 'same'))
        model.add(MaxPooling2D((2, 2), strides = (2, 2)))
        model.add(BatchNormalization())
    
        model.add(Flatten())
        
        model.add(Dense((2048), 
            activation = 'relu', 
            kernel_regularizer = regularizers.l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense((512), 
            activation = 'relu', 
            kernel_regularizer = regularizers.l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(1))
        model.add(BatchNormalization())
        
        model.add(Activation('sigmoid'))
        
        self.model = model
        
        # Print the model summary
        print(model.summary())
 
    # Load the model weights
    def loadModel(self):
        self.model.load_weights(self.sPathModel + 'final_model.hdf5')
        print('Model weights loaded.')
    
    # train the model
    def trainModel(self):
        # Image pre-processor
        train_img_gen = ImageDataGenerator(
            #featurewise_std_normalization = True,
            rotation_range = 7.5,
            width_shift_range = 0.05,
            height_shift_range = 0.05,
            shear_range = 0.05,
            zoom_range = 0.05,
            horizontal_flip = True)
        
        valid_img_gen = ImageDataGenerator()
        
        # Train batch generator
        train_gen = train_img_gen.flow_from_directory(
            self.sPathTrain,
            target_size = (224, 224),
            batch_size = 16,
            class_mode = 'binary')

        # Validation batch generator
        valid_gen = valid_img_gen.flow_from_directory(
            self.sPathValid,
            target_size = (224, 224),
            batch_size = 16,
            class_mode = 'binary')    
    
        # Compile the model
        self.model.compile(
            loss = 'binary_crossentropy',
            optimizer = optimizers.adam(),
            metrics = ['accuracy'])
        
        # Save the weigth of each iteration
        sCheckPoint = '{epoch:02d}-{loss:.4f}-{val_loss:.4f}-{acc:.4f}-{val_acc:.4f}.hdf5'
        checkpoint = ModelCheckpoint(
            filepath = self.sPathCheckPoints + sCheckPoint,
            verbose = 0)
    
        # Early stop, when improvement stops        
        early_stop = EarlyStopping(
            monitor = 'val_loss', 
            min_delta = 0,
            patience = 80)
            
        # Train the model
        self.model_trained = self.model.fit_generator(
            generator = train_gen,
            validation_data = valid_gen, 
            steps_per_epoch = len(train_gen.filenames) / 16,
            validation_steps = len(valid_gen.filenames) / 16,            
            epochs = 300, 
            callbacks = [early_stop, checkpoint],
            verbose = 1)
    
        # Summarize history for accuracy check
        plt.plot(self.model_trained.history['acc'],'r')
        plt.plot(self.model_trained.history['val_acc'],'b')
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc = 'upper left')
        plt.show()
        
        plt.plot(self.model_trained.history['loss'],'r')
        plt.plot(self.model_trained.history['val_loss'],'b')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc=  'upper left')
        plt.show()

    # Create the model by importing weights or running the training
    def createModel(self):
        self.defineModel()
        
        if not os.path.isdir(self.sPathModel):
            os.mkdir(self.sPathModel)
        
        if len(os.listdir(self.sPathModel)) != 0:
            self.loadModel()
        else:
            self.trainModel()
    
    # Make the predictions
    # For some reason using predict_generation did not work,
    # and that's the reason for the work around
    def makePredictions(self):        
        self.lTestNames = []
        self.lTestPreds = []
        
        for i in range(1, len(os.listdir(self.sPathTest)) + 1):
            img = cv2.imread(self.sPathTest + str(i) + '.jpg')
            img = np.asarray([img])
            pred = round(self.model.predict(img).item())
            
            self.lTestNames.append(i)
            self.lTestPreds.append(pred)

            if (i % 100 == 0): 
                print ('Test image number: ' + str(i))
    
    # Write the predictions to a file
    def writeToFile(self):
        self.dfPreds = pd.DataFrame(
            {'names': self.lTestNames,
             'labels': self.lTestPreds})
        
        self.dfPreds.to_csv('test_preds.csv', sep = ',', index = False)

# Run the script if not imported
if __name__ == '__main__':
    invasive = ImageClassifier()
    invasive.preprocess()
    invasive.createModel()
    invasive.makePredictions()
    invasive.writeToFile()
