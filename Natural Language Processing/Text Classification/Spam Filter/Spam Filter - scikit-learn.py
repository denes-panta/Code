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

#XgBoost
from xgboost import XGBClassifier

#Sklearn
from sklearn.feature_extraction.text import TfidfVectorizer as tfidfV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split as split

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report as cr
from sklearn.naive_bayes import BernoulliNB as bNB
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC

#Keras
from keras.preprocessing import text, sequence

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
        
    lNames = list(map(str, dictIdToLab.values()))
    print(cr(lTeY, lPreds, target_names = lNames))
    
    return accuracy_score(lPreds, lTeY)


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
    
    #Create token-embedding mapping
    mEmbed = np.zeros((len(wordInd) + 1, 300))
    for word, i in wordInd.items():
        lVecEmbed = wordInd.get(word)
        if lVecEmbed is not None:
            mEmbed[i] = lVecEmbed
    
    lTopicSummaries = summTopics(countVect, cvTrainX)
    
    #Testing of various models
    
    #Naive Bayes
    #Naive Bayes on Count Vectors
    accuracy = trainModel(bNB(), cvTrainX, lTrainY, cvTestX, lTestY)
    print("NB, Count Vectors: %0.4f" % (accuracy))
    
    #Naive Bayes on Word Level TF IDF Vectors
    accuracy = trainModel(bNB(), XtrV, lTrainY, XteV, lTestY)
    print("NB, Word Level TF-IDF: %0.4f" % (accuracy))
    
    #Naive Bayes on Ngram Level TF IDF Vectors
    accuracy = trainModel(bNB(), XtrVN, lTrainY, XteVN, lTestY)
    print("NB, N-Gram Level TF-IDF: %0.4f" % (accuracy))
    
    #Naive Bayes on Character Level TF IDF Vectors
    accuracy = trainModel(bNB(), XtrVNC, lTrainY, XteVNC, lTestY)
    print("NB, Char Level TF-IDF:: %0.4f" % (accuracy))


    #Logistic Regression
    # Linear Classifier on Count Vectors
    accuracy = trainModel(LogReg(), cvTrainX, lTrainY, cvTestX, lTestY)
    print("LR, Count Vectors: %0.4f" % (accuracy))
    
    # Linear Classifier on Word Level TF IDF Vectors
    accuracy = trainModel(LogReg(), XtrV, lTrainY, XteV, lTestY)
    print("LR, Word Level TF-IDF: %0.4f" % (accuracy))
    
    # Linear Classifier on Ngram Level TF IDF Vectors
    accuracy = trainModel(LogReg(), XtrVN, lTrainY, XteVN, lTestY)
    print("LR, N-Gram Level TF-IDF: %0.4f" % (accuracy))
    
    # Linear Classifier on Character Level TF IDF Vectors
    accuracy = trainModel(LogReg(), XtrVNC, lTrainY, XteVNC, lTestY)
    print("LR, Char Level TF-IDF:: %0.4f" % (accuracy))
 
    #SVM
    #SVM on Ngram Level TF IDF Vectors
    accuracy = trainModel(SVC(), XtrVN, lTrainY, XteVN, lTestY)
    print("SVM, N-Gram Level TF-IDF: %0.4f" % (accuracy))


    #Random Forest    
    #RF on Count Vectors
    accuracy = trainModel(RFC(), cvTrainX, lTrainY, cvTestX, lTestY)
    print("RF, Count Vectors: %0.4f" % (accuracy))
    
    #RF on Word Level TF IDF Vectors
    accuracy = trainModel(RFC(), XtrV, lTrainY, XteV, lTestY)
    print("RF, Word Level TF-IDF: %0.4f" % (accuracy))

    #RF on Ngram Level TF IDF Vectors
    accuracy = trainModel(RFC(), XtrVN, lTrainY, XteVN, lTestY)
    print("RF, N-Gram Level TF-IDF: %0.4f" % (accuracy))

    #RF on Character Level TF IDF Vectors
    accuracy = trainModel(RFC(), XtrVNC, lTrainY, XteVNC, lTestY)
    print("RF, Char Level TF-IDF:: %0.4f" % (accuracy))


    #XGBoost
    #Extereme Gradient Boosting on Count Vectors
    accuracy = trainModel(XGBClassifier(), 
                          cvTrainX.tocsc(), lTrainY, 
                          cvTestX.tocsc(), lTestY)
    print("Xgb, Count Vectors: %0.4f" % (accuracy))
    
    #Extereme Gradient Boosting on Word Level TF IDF Vectors
    accuracy = trainModel(XGBClassifier(), 
                          XtrV.tocsc(), lTrainY, 
                          XteV.tocsc(), lTestY)
    print("Xgb, Word Level TF-IDF: %0.4f" % (accuracy))
    
    #Extereme Gradient Boosting on NGram Level TF IDF Vectors
    accuracy = trainModel(XGBClassifier(), 
                          XtrVN.tocsc(), lTrainY, 
                          XteVN.tocsc(), lTestY)
    print("Xgb, N-Gram Level TF-IDF: %0.4f" % (accuracy))
    
    #Extereme Gradient Boosting on Character Level TF IDF Vectors
    accuracy = trainModel(XGBClassifier(), 
                          XtrVNC.tocsc(), lTrainY, 
                          XteVNC.tocsc(), lTestY)
    print("Xgb, Char Level TF-IDF: %0.4f" % (accuracy))
    