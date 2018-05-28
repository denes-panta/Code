#Data : https://www.kaggle.com/c/digit-recognizer/data

#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gc
import glob

from scipy.stats import mode
from sklearn import model_selection

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization

#Functions
def bulk_predict(test_xy): #Reads in the weights from the specified folder and returnes the predictions for each one ina a matrix 
    weight_list = glob.glob("F:/Code/Python/1 Digits/Models/Final/*.hdf5") #get the list of weights from the specified folder
    end_test_p = np.zeros((len(test_xy), 0)) #create the prediction matrix
    
    for i in range(len(weight_list)): #Load each weigth and make the predictions
        print ("Weight list number: %d" % (i+1))
        model.load_weights(weight_list[i]) #Load Weigth
        v_pred = model.predict(test_xy, 
                               batch_size = 64, 
                               verbose = 1
                               )
        print ("\n") #New line
        v_pred = np.argmax(v_pred, axis = 1)
        v_pred =  np.reshape(v_pred, (v_pred.shape[0], 1))
        end_test_p = np.concatenate((end_test_p, v_pred), axis = 1)

    return(end_test_p) #Return the prediction matrix


def pred_vote(test_pred): #Vote based on the predictions
    m_y = np.zeros((int(len(test_pred)), 1), dtype = np.uint)
    m_y = mode(test_pred, axis = 1)[0]

    return(m_y)


def write_to_file(pred_array): #Write the results to a file
    m_temp = np.zeros((len(pred_array), 2))
    
    for i in range(len(m_temp)):
        m_temp[i, 0] = i + 1
        m_temp[i, 1] = pred_array[i,0]
        
    df_sample = pd.DataFrame(m_temp,columns = ['ImageId', 'Label']).astype(int)
    del m_temp, i
    df_sample.to_csv("F:/Code/Python/1 Digits/submission.csv", index = False)
    
    return print("Submission saved to csv file.")


#Set seed
np.random.seed(117)

#Reading in the Data
df_train = pd.read_csv("F:/Code/Python/1 Digits/train.csv").as_matrix()
df_test = pd.read_csv("F:/Code/Python/1 Digits/test.csv").as_matrix()

end_train_x = np.reshape(df_train[:, 1:],(len(df_train), 28, 28, 1)).astype(dtype = "float32")
end_train_y = np_utils.to_categorical(df_train[:, 0], 10).astype(dtype = "int16")
end_test_x = np.reshape(df_test,(len(df_test), 28, 28, 1)).astype(dtype = "float32")

del df_train, df_test

#Build the model
def neuralnet(): #Custom neural network
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation = "relu", input_shape = (28, 28, 1)))
    model.add(Dropout(0.625))
    
    model.add(Conv2D(64, (3, 3), activation = "relu"))
    model.add(Dropout(0.625))
    
    model.add(Conv2D(64, (6, 6), activation = "relu"))
    model.add(Dropout(0.625))
    
    model.add(Conv2D(64, (9, 9), activation = "relu"))
    model.add(Dropout(0.625))
    
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.625))
    model.add(Dense(64, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation = 'softmax'))

    print(model.summary())

    return model

#Bagging
rounds = 10
splits = 5

for i in range(rounds):
    gc.collect() #Garbage collection because of memory constraints
    
    kf = model_selection.KFold(n_splits = splits, #K-fold split the data
                               shuffle = True
                               )

    for train_index, test_index in kf.split(end_train_x): #Validation, training set creation
        X_train, X_test = end_train_x[train_index], end_train_x[test_index] 
        y_train, y_test = end_train_y[train_index], end_train_y[test_index]

    gc.collect() #Garbage collection because of memory constraints

    y_train = np.uint8(y_train) #Data type modification
    y_test = np.uint8(y_test)
    train_index = np.int16(train_index)
    test_index = np.int16(test_index)

    datagen = ImageDataGenerator(featurewise_center = True, #Preprocess the images
                                 featurewise_std_normalization = True,
                                 rotation_range = 5,
                                 width_shift_range = 0.025,
                                 height_shift_range = 0.025,
                                 shear_range = 0.025,
                                 zoom_range = 0.025,
                                 horizontal_flip = False
                                 )

    datagen.fit(X_train)
        
    model = neuralnet()

    model.compile(loss = 'categorical_crossentropy', #compile the model
                  optimizer = optimizers.adam(lr = 0.001, decay = 1e-06),
                  metrics = ['accuracy']
                  )

    checkpointer = ModelCheckpoint(filepath = "F:/Code/Python/1 Digits/Models/" + str(i) + "/{epoch:02d}-{loss:.4f}-{acc:.4f}---{val_loss:.4f}-{val_acc:.4f}.hdf5", 
                                   verbose = 0) #Save the weigth of each iteration
    
    early_stop = EarlyStopping(monitor='val_loss', #Early stop, when improvement stops
                               min_delta = 0,
                               patience = 20
                               )

    gc.collect() #Garbage collection because of memory constraints

    model_vgg = model.fit_generator(datagen.flow(X_train, #Fit the model
                                                 y_train, 
                                                 batch_size = 64,
                                                 shuffle = True
                                                 ),
                                    validation_data = (X_test, y_test), 
                                    steps_per_epoch = len(X_train) / 64,
                                    epochs = 100, 
                                    callbacks = [early_stop, checkpointer],
                                    verbose = 1
                                    )

    #Summarize history for accuracy
    plt.plot(model_vgg.history['acc'],'r')
    plt.plot(model_vgg.history['val_acc'],'b')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    plt.plot(model_vgg.history['loss'],'r')
    plt.plot(model_vgg.history['val_loss'],'b')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


#Predict
end_test_p = bulk_predict(end_test_x)
end_test_r = pred_vote(end_test_p)

#Write
write_to_file(end_test_p)