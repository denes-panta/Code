#General libraries
import string
import os
import nltk
import random
import re
import math
from collections import Counter

#Sklearn and NLTK libraries
from sklearn.model_selection import train_test_split as split
from nltk.tokenize import RegexpTokenizer
from nltk import WordNetLemmatizer as Lemm
from nltk.stem.porter import PorterStemmer as Stem

#Set seed
random.seed(117)

#Import data
def impData(sPath, sCat, fP):
    reviews = []

    for i, filename in enumerate(os.listdir(sPath)): 
        reviews.insert(i, [open(sPath + filename).read(), sCat])
    
    #Create training & test sets
    lTrain, lTest = split(reviews, train_size = fP)
    
    return lTrain, lTest

#Pre-process the data
def preProcess(lData, norm = "l"):
    reviews = []
    iL = len(lData)
    
    #Define the tokenizer
    tokenizer = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', 
                                gaps = True
                                ) 

    #Define Stopwords
    stopWords = set(nltk.corpus.stopwords.words('english') + \
                    list((' ', '\n', 'and', 'a', 'all')))

    #Translate punctuations to ' '
    ttab = str.maketrans(string.punctuation, ' ' * len(string.punctuation)) 
    
    for i in range(0, iL): 
        #tokenize
        text = tokenizer.tokenize(lData[i][0])
        #lower case the words and translate punctuations
        text = [w.lower().translate(ttab) for w in text]
        #filter out digits
        text = [re.sub(r'\d+', ' ', w) for w in text if w] 
        #filter out the stopwords
        text = [w for w in text if w not in stopWords] 
        #Stem/Lemmatize the remaining words
        if norm == "s":
            #Stem the remaining words
            text = [Stem().stem(w) for w in text]
        elif norm == "l":
            #Lemmatize the remaining words
            text = [Lemm().lemmatize(w) for w in text]
        
        reviews.append([text, lData[i][1]])
        
    return reviews

#Calculate the score of each features by tf, idf or tf-idf
def calcScore(lData, method = "tf-idf"):
    #Calculate the word frequencies
    iL = len(lData)
    #Total number of documents
    iTnD = iL
    
    lCounterTf = []

    #Calculate the Frequencies of each word by document    
    for i in range(0, iL):
        cFr = Counter(lData[i][0])
        cTf = {k: v/len(cFr) for k, v in cFr.items()}
        lCounterTf.append([cTf, lData[i][1]])        
    
    if method == "tf":
        return lCounterTf
    
    #Calculate the tf-idf or idf
    for i in range(0, iL):
        lDocWords = list(lCounterTf[i][0].keys())

        for w in lDocWords:
            d = sum(list(map(lambda x: math.ceil(x[0].get(w, 0)), lCounterTf)))                
            
            if method == "idf":
                lCounterTf[i][0][w] = math.log(iTnD/d)
            elif method == "tf-idf":
                lCounterTf[i][0][w] *= math.log(iTnD/d)

        return lCounterTf

#Format the test data into dictionary format
def dictTest(lTest):
    lNew = []
    iL = len(lTest)
    
    for i in range(0, iL):
        lNew.append([dict([(j, False) for j in lTest[i][0]]), lTest[i][1]])
    
    return lNew

#Get the features into the proper format for the NLTK bayesian algorithm
def dimRed(lTrain, lTest, th = 0.0):
    lFeat = []
    lAllWords = []
    
    #Sets each feature either True or False based on its presence
    def featurize(lData, lWords):
        for dDict in lData:
            for word in lAllWords:
                if dDict[0].get(word, None) == None:
                    dDict[0][word] = False
                else:
                    dDict[0][word] = True

        return lData
    
    #Filter the dictionary based on the (th)reshold
    for dFull in lTrain:
        dFilt = dict((k, v) for (k, v) in dFull[0].items() if v > th)
        if dFilt:
            lFeat.append([dFilt, dFull[1]])
            lAllWords += list(dFilt.keys())
    
    lAllWords = list(set(lAllWords))
    
    lFeatTr = featurize(lFeat, lAllWords)
    lFeatTe = featurize(lTest, lAllWords)

    return lFeatTr, lFeatTe

#Run the classification algorithm
def classify(lTrain, lTest):
    #Define the classifier
    classifier = nltk.NaiveBayesClassifier.train(lTrain) 

    #Print the test accuracy
    print("Testing Accuracy (NBC): ", 
          (nltk.classify.accuracy(classifier, lTest)) * 100) 
    
    #return the most informative features
    return classifier.most_informative_features(25)


#Script
if __name__ == "__main__":
    sPath = "F:\\Code\\Code\\Natural Language Processing\\Text Classification\\Sentiment Analysis\\BinaryClass\\"
    lTrainN, lTestN = impData(sPath + "neg\\", "N", 0.8)
    lTrainP, lTestP = impData(sPath + "pos\\", "P", 0.8)
    
    lTrain = lTrainN + lTrainP
    lTest = lTestN + lTestP
    
    random.shuffle(lTest)
    random.shuffle(lTrain)
    
    del lTrainN, lTrainP, lTestN, lTestP
    
    lTrain = preProcess(lTrain, "l")
    lTest = preProcess(lTest, "l")
    
    lTrain = calcScore(lTrain, "tf")
    lTest = dictTest(lTest)
    
    lTrFeat, lTeFeat = dimRed(lTrain, lTest, th = 0.005)
    classify(lTrFeat, lTeFeat)