#Source:
#https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/

#Word Embedding Vector:
#https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip

#General libraries
import random
import pandas as pd
import numpy as np
import os
import re

#Sklearn
from sklearn.feature_extraction.text import TfidfVectorizer as tfidfV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split as split

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report as cr

#Keras
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

#Functions
#Import data
def impData(sPath, sCat, fP):
    reviews = []

    for i, filename in enumerate(os.listdir(sPath)):
        fSpam = open(sPath + filename, 'rb')
        reviews.insert(i, [fSpam.read(), sCat])
    
    #Create training & test sets
    lTrain, lTest = split(reviews, train_size = fP)
    
    return lTrain, lTest

#Extract email parts
def extractParts(emails):
    body = []

    for email in emails:
        #Extract bodies
        body.append([(str(re.search(b'(?m)^Subject: (.+)$', email[0], re.DOTALL).group(1)) + 
                      str(re.search(b'(?m)^From: (.*)', email[0]).group(1))), email[1]
                    ]) 
        
    return body

#Load the pre-trained word-embedding vectors 
def loadEmbVec(sVecFilePath):
    dEmbInd = {}

    for i, line in enumerate(open(sVecPath, "r", encoding = "utf-8")):
        lValues = line.split()
        dEmbInd[lValues[0]] = np.asarray(lValues[1:], dtype = 'float32')

    return dEmbInd

#Get Topic Summaries
def summTopics(countVect, cvTrainX):
    #Train the model
    modelLDA = LatentDirichletAllocation(n_components = 20, 
                                         learning_method = 'online', 
                                         max_iter = 20
                                         )
    
    LDAtopics = modelLDA.fit_transform(cvTrainX)
    LDAwords = modelLDA.components_ 
    LDAvocab = countVect.get_feature_names()

    #Get the topic models
    iW = 10
    lTopicSum = []
    
    for i, topicDist in enumerate(LDAwords):
        mTopicW = np.array(LDAvocab)[np.argsort(topicDist)][:-(iW + 1):-1]
        lTopicSum.append(' '.join(mTopicW))
        
    return mTopicW

#General trainer function
def trainModel(classifier, lTrX, lTrY, lTeX, lTeY, is_neural_net = False):
    #Fit the training dataset on the classifier
    classifier.fit(lTrX, lTrY)
    
    #Predict the labels on validation dataset
    lPreds = classifier.predict(lTeX)
    
    if is_neural_net:
        lPreds = lPreds.argmax(axis = -1)
    else:
        lNames = list(map(str, dictIdToLab.values()))
        print(cr(lTeY, lPreds, target_names = lNames))
    
    return accuracy_score(lPreds, lTeY)


#Neural Network Architecures:
def createShallowNN(input_size):
    input_layer = layers.Input((input_size, ), sparse = True)

    hidden_layer = layers.Dense(100, activation = "relu")(input_layer)

    output_layer = layers.Dense(1, activation = "sigmoid")(hidden_layer)

    classifier = models.Model(inputs = input_layer, 
                              outputs = output_layer)
    classifier.compile(optimizer = optimizers.Adam(), 
                       loss = 'binary_crossentropy')

    return classifier 

def createCNN(word_index, embedding_matrix):
    input_layer = layers.Input((1000, ))

    embedding_layer = layers.Embedding(len(word_index) + 1, 300, 
                                       weights = [embedding_matrix], 
                                       trainable = False)(input_layer)
    
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    conv_layer = layers.Convolution1D(100, 3, activation = "relu")(embedding_layer)

    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    output_layer1 = layers.Dense(50, activation = "relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation = "sigmoid")(output_layer1)

    model = models.Model(inputs = input_layer, outputs = output_layer2)
    model.compile(optimizer = optimizers.Adam(), loss = 'binary_crossentropy')
    
    return model

def createLSTM(word_index, embedding_matrix):
    input_layer = layers.Input((1000, ))

    embedding_layer = layers.Embedding(len(word_index) + 1, 300, 
                                       weights = [embedding_matrix], 
                                       trainable = False)(input_layer)
    
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    lstm_layer = layers.LSTM(100)(embedding_layer)

    output_layer1 = layers.Dense(50, activation = "relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation = "sigmoid")(output_layer1)

    model = models.Model(inputs = input_layer, outputs=output_layer2)
    model.compile(optimizer = optimizers.Adam(), loss = 'binary_crossentropy')
    
    return model

def createGRU(word_index, embedding_matrix):
    input_layer = layers.Input((1000, ))

    embedding_layer = layers.Embedding(len(word_index) + 1, 300, 
                                       weights = [embedding_matrix], 
                                       trainable = False)(input_layer)
    
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    lstm_layer = layers.GRU(100)(embedding_layer)

    output_layer1 = layers.Dense(50, activation = "relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation = "sigmoid")(output_layer1)

    model = models.Model(inputs = input_layer, outputs = output_layer2)
    model.compile(optimizer = optimizers.Adam(), loss = 'binary_crossentropy')
    
    return model

def createBiDiRNN(word_index, embedding_matrix):
    input_layer = layers.Input((1000, ))

    embedding_layer = layers.Embedding(len(word_index) + 1, 300, 
                                       weights = [embedding_matrix], 
                                       trainable = False)(input_layer)
    
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    lstm_layer = layers.Bidirectional(layers.GRU(100))(embedding_layer)

    output_layer1 = layers.Dense(50, activation = "relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation = "sigmoid")(output_layer1)

    model = models.Model(inputs = input_layer, outputs = output_layer2)
    model.compile(optimizer = optimizers.Adam(), loss = 'binary_crossentropy')
    
    return model

def createRCNN(word_index, embedding_matrix):
    input_layer = layers.Input((1000, ))

    embedding_layer = layers.Embedding(len(word_index) + 1, 300, 
                                       weights = [embedding_matrix], 
                                       trainable = False)(input_layer)
    
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)
    
    rnn_layer = layers.Bidirectional(layers.GRU(50, return_sequences = True))(embedding_layer)
    
    conv_layer = layers.Convolution1D(100, 3, activation = "relu")(rnn_layer)

    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    output_layer1 = layers.Dense(50, activation = "relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation = "sigmoid")(output_layer1)

    model = models.Model(inputs=input_layer, outputs = output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss = 'binary_crossentropy')
    
    return model


#Script
if __name__ == "__main__":
    random.seed(117)
    
    #Import
    sPath = "F:\\Code\\Code\\Natural Language Processing\\Text Classification\\Spam Filter\\BinaryClass\\"
    sVecPath = "F:\\Code\\NLP\\wiki-news-300d-1M.vec"

    lTrainH, lTestH = impData(sPath + "ham\\", "H", 0.8)
    lTrainS, lTestS = impData(sPath + "spam\\", "S", 0.8)

    lTrain = lTrainH + lTrainS
    lTest = lTestH + lTestS

    lTrain = extractParts(lTrain)
    lTest = extractParts(lTest)
    
    dfTrain = pd.DataFrame(lTrain)
    dfTrain.columns = ["email", "cat"]
    dfTest = pd.DataFrame(lTest)
    dfTest.columns = ["email", "cat"]
    dfData = pd.concat([dfTrain, dfTest])


    #Encoding Labels/Categories
    dfData['category_id'] = dfData['cat'].factorize()[0]
    dfLabelToId = \
    dfData[['cat', 'category_id']].drop_duplicates().sort_values('category_id')
    dictLabToId = dict(dfLabelToId.values)
    dictIdToLab = dict(dfLabelToId[['category_id', 'cat']].values)

    lTrainY = dfTrain["cat"].apply(lambda x: dictLabToId[x])
    lTestY = dfTest["cat"].apply(lambda x: dictLabToId[x])

    del lTrainH, lTrainS, lTestH, lTestS, lTest, lTrain

    #Create Count Vectorizer
    countVect = CountVectorizer(analyzer = 'word', 
                                token_pattern=r'\w{1,}'
                                )
    countVect.fit(dfTrain["email"])

    cvTrainX = countVect.transform(dfTrain["email"])
    cvTestX = countVect.transform(dfTest["email"])
    
    #Word level tf-idf
    tfidfVect = tfidfV(analyzer = 'word', 
                       token_pattern = r'\w{1,}', 
                       max_features = 5000)
    tfidfVect.fit(dfData["email"])
    XtrV = tfidfVect.transform(dfTrain["email"])
    XteV = tfidfVect.transform(dfTest["email"])
    
    #Ngram level tf-idf 
    tfidfVectNgram = tfidfV(analyzer = 'word', 
                            token_pattern = r'\w{1,}', 
                            ngram_range = (2, 3), 
                            max_features = 5000)
    tfidfVectNgram.fit(dfData["email"])
    XtrVN = tfidfVectNgram.transform(dfTrain["email"])
    XteVN = tfidfVectNgram.transform(dfTest["email"])
    
    #Characters level tf-idf
    tfidfVectNgramChars = tfidfV(analyzer = 'char', 
                                 token_pattern = r'\w{1,}', 
                                 ngram_range = (2, 3), 
                                 max_features = 5000)
    tfidfVectNgramChars.fit(dfData["email"])
    XtrVNC = tfidfVectNgramChars.transform(dfTrain["email"])
    XteVNC = tfidfVectNgramChars.transform(dfTest["email"])
    
    #Load embediing vector
    dEmbInd = loadEmbVec(sVecPath)

    #Create a tokenizer 
    token = text.Tokenizer()
    token.fit_on_texts(dfTrain['email'])
    wordInd = token.word_index
    
    #Convert text to sequence of tokens and pad them to ensure equal length vectors 
    seqTrainX = sequence.pad_sequences(token.texts_to_sequences(dfTrain["email"]), 
                                       maxlen = 1000
                                       )
    seqTestX = sequence.pad_sequences(token.texts_to_sequences(dfTest["email"]), 
                                      maxlen = 1000
                                      )
    
    #Create token-embedding mapping
    mEmbed = np.zeros((len(wordInd) + 1, 300))
    for word, i in wordInd.items():
        lVecEmbed = wordInd.get(word)
        if lVecEmbed is not None:
            mEmbed[i] = lVecEmbed
    
    lTopicSummaries = summTopics(countVect, cvTrainX)
    
    #Testing of various Neural Network architectures
    #NN on NGram Level TF IDF Vectors
    classifier = createShallowNN(XtrVN.shape[1])
    accuracy = trainModel(classifier,
                          XtrVN, lTrainY, 
                          XteVN, lTestY, 
                          is_neural_net = True)
    print("NN, Ngram Level TF IDF Vectors: %0.4f" % (accuracy))

    #CNN on Word Embeddings
    classifier = createCNN(wordInd, mEmbed)
    accuracy = trainModel(classifier,
                          seqTrainX, lTrainY, 
                          seqTestX, lTestY, 
                          is_neural_net = True)
    print("CNN, Word Embeddings: %0.4f" % (accuracy))
    
    #LSTM on Word Embeddings
    classifier = createLSTM(wordInd, mEmbed)
    accuracy = trainModel(classifier,
                          seqTrainX, lTrainY, 
                          seqTestX, lTestY, 
                          is_neural_net = True)
    print("LSTM, Word Embeddings: %0.4f" % (accuracy))

    #GRU on Word Embeddings
    classifier = createGRU(wordInd, mEmbed)
    accuracy = trainModel(classifier,
                          seqTrainX, lTrainY, 
                          seqTestX, lTestY, 
                          is_neural_net = True)
    print("GRU, Word Embeddings: %0.4f" % (accuracy))

    #BiDiRNN on Word Embeddings
    classifier = createBiDiRNN(wordInd, mEmbed)
    accuracy = trainModel(classifier,
                          seqTrainX, lTrainY, 
                          seqTestX, lTestY, 
                          is_neural_net = True)
    print("BiDiRNN, Word Embeddings: %0.4f" % (accuracy))
    
    #CRNN on Word Embeddings
    classifier = createRCNN(wordInd, mEmbed)
    accuracy = trainModel(classifier,
                          seqTrainX, lTrainY, 
                          seqTestX, lTestY, 
                          is_neural_net = True)
    print("RCNN, Word Embeddings: %0.4f" % (accuracy))