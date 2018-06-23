#Source:
#https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f

#Libraries
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import random   

#Seaborn
import seaborn as sns

#Sk-learn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split as split
from sklearn.model_selection import cross_val_score as cvs
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report as cr

from sklearn.feature_extraction.text import TfidfVectorizer as tfidfV
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.svm import LinearSVC as lsvc

#Import the reviews
def impFiles(sPath):
    lData = []

    for i in range(1, 11):
        sFilePath = sPath + "\\" + str(i) + "\\"
    
        for file in os.listdir(sFilePath):
            text = open(sFilePath + file, "r", encoding = "utf-8")
            lData.append([text.read(), i])

    dfData = pd.DataFrame(lData)
    dfData.columns = ["review", "rating"]
    
    return dfData

#Plot the number of datapoints to each class
def visualiseClasses(df):
    df.groupby('rating').review.count().plot.bar(ylim = 0)
    plt.show()

#Subsample the data to get equal number of datapoints in each category
def subSampleData(df):
    iSize = min(df.groupby('rating').review.count())
    replace = False
 
    fn = lambda x: x.loc[np.random.choice(x.index, iSize, replace), :]
    dfSampled = df.groupby("rating", as_index = False).apply(fn)

    return dfSampled

#Get number the most correlated ngrams for the labels
def getNGrams(mFeatures, SLabels, dictIdToLab):
    N = 2

    for rating, catid in sorted(dictIdToLab.items()):
        mFeatChi2 = chi2(mFeatures, SLabels == catid)
        indices = np.argsort(mFeatChi2[0])
        mFeatNames = np.array(model_tfidf.get_feature_names())[indices]

        lUnigrams = [v for v in mFeatNames if len(v.split(' ')) == 1]
        lBigrams = [v for v in mFeatNames if len(v.split(' ')) == 2]
        lTrigrams = [v for v in mFeatNames if len(v.split(' ')) == 3]

        print("# '{}':".format(rating))
        print("  . Most cor. unigrams:\n. {}".format('\n. '.join(lUnigrams[-N:])))
        print("  . Most cor. bigrams:\n. {}".format('\n. '.join(lBigrams[-N:])))
        print("  . Most cor. trigrams:\n. {}".format('\n. '.join(lTrigrams[-N:])))

    return(lUnigrams, lBigrams, lTrigrams)    

#Create the train, test datasets
def createPartitions(dfData, p =  0.3):
    dfTrain, dfTest = split(dfData, 
                            test_size = p, 
                            stratify = dfData["rating"]
                            )
    
    #Shuffle the data
    dfTrain = shuffle(dfTrain)
    dfTest = shuffle(dfTest)
    
    #Split the parts to X and Y
    dfTrainX = dfTrain["review"]
    dfTrainY = dfTrain["category_id"]
    dfTestX = dfTest["review"]
    dfTestY = dfTest["category_id"]

    return dfTrainX, dfTrainY, dfTestX, dfTestY

#Train the selected models to see the best one
def trainModels(cv, TrainX, TrainY):
    #Specify the models to be trained
    lModels = [rf(n_estimators = 200, max_depth = 3), 
               lsvc(), 
               LogReg()
               ]
    #Number of runs for cv
    iCv = cv
    
    #Create DataFrame for results
    dfCv = pd.DataFrame(index = range(iCv * len(lModels)))
    entries = []
    
    #Trainy the models    
    for model in lModels:
        model_name = model.__class__.__name__
        accuracies = cvs(model, TrainX, TrainY, scoring = 'accuracy', cv = iCv)

        #Get scores for cV
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))

        dfCv = pd.DataFrame(entries, 
                            columns = ['model_name', 'fold_idx', 'accuracy']
                            )
    #Visualise the results
    sns.boxplot(x = 'model_name', 
                y = 'accuracy', 
                data = dfCv
                )
    sns.stripplot(x = 'model_name', 
                  y = 'accuracy', 
                  data = dfCv, 
                  size = 8, 
                  jitter = True, 
                  edgecolor = "gray", 
                  linewidth = 2
                  )
    plt.show()

    return dfCv

#Evalute the predictions of the best model
def evalBest(model, mDataX, dfDataY, dictIdToLab):
    model = model

    #Split the data into train and test sets
    trainX, testX, trainY, testY = split(mDataX, 
                                         dfDataY, 
                                         test_size = 0.2
                                         )
    
    #Fit the model and predict
    model.fit(trainX, trainY)
    predsY = model.predict(testX)
    
    #Confusion matrix and visualisation
    conf_mat = cm(testY, predsY)
    sns.heatmap(conf_mat, 
                annot = True, 
                fmt = 'd',
                xticklabels = dictIdToLab.values(), 
                yticklabels = dictIdToLab.values()
                )
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    #Print the model evaluation
    lNames = list(map(str, dictIdToLab.values()))
    print(cr(testY, predsY, target_names = lNames))   
    
    return None

#Script
if __name__ == "__main__":
    random.seed(117)
    
    sDirPath = "F:\\Code\\Code\\Natural Language Processing\\Text Classification\\Sentiment Analysis\\MultiClass"
    dfData = impFiles(sDirPath)
    visualiseClasses(dfData)
    
    dfData['category_id'] = dfData['rating'].factorize()[0]
    #dfData = subSampleData(dfData)
    
    dfLabelToId = \
    dfData[['rating', 'category_id']].drop_duplicates().sort_values('category_id')
    dictLabToId = dict(dfLabelToId.values)
    dictIdToLab = dict(dfLabelToId[['category_id', 'rating']].values)
    
    model_tfidf = tfidfV(sublinear_tf = True, 
                         min_df = 5, 
                         norm = 'l2', 
                         encoding = 'latin-1', 
                         ngram_range = (1, 3),
                         stop_words = 'english'
                         )
    
    dfTrainX = dfData["review"] 
    dfTrainY = dfData["category_id"]
    
    mTrainX = model_tfidf.fit_transform(dfTrainX).toarray()
    
    lUni, lBi, lTri = getNGrams(mTrainX, dfTrainY, dictIdToLab)
    dfModInfo = trainModels(5, mTrainX, dfTrainY)

    evalBest(LogReg(solver = "newton-cg", 
                    multi_class = "multinomial",
                    class_weight = "balanced",
                    C = 0.5
                    ),
                    mTrainX, dfTrainY, dictIdToLab
                    )

