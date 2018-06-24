import gensim
from nltk.tokenize import word_tokenize
import pandas as pd
import random
import numpy as np

#Functions
#Import the data
def impData(sPath):
    dfX = pd.read_csv(sPath + "complains_data.csv", header = 0)
    dfX = dfX.loc[:, ("Complaint ID", "Consumer complaint narrative")]

    return dfX

#Sample the data
def getSample(dfX, p):
    i = int(len(dfX) * p)
    dfX = dfX[:i]

    return dfX

#Script
if __name__ == "__main__":
    random.seed(117)
    sPath = "F:\\Code\\Interviews\\Closer\\raw\\"
    sCol = "Consumer complaint narrative"

    dfFullData = impData(sPath)
    dfData = getSample(dfFullData, 0.01)

    iL = len(dfFullData[sCol])

    print("Number of documents: ", iL)

    lDocs = [[w.lower() for w in word_tokenize(text)] for text in dfData[sCol]]
    dictionary = gensim.corpora.Dictionary(lDocs)    
    print("Number of words in dictionary:", len(dictionary))
    
    Corpus = [dictionary.doc2bow(doc) for doc in lDocs]
    modelTfIdf = gensim.models.TfidfModel(Corpus)
    print(modelTfIdf)

    s = 0
    for i in Corpus:
        s += len(i)
        print(s)

    sims = gensim.similarities.Similarity(sPath, 
                                          modelTfIdf[Corpus],
                                          num_features = len(dictionary))

    lQDoc = [w.lower() for w in word_tokenize(dfData.loc[np.random.randint(0, iL), sCol])]
    lQB = dictionary.doc2bow(lQDoc)
    QDocTfIdf = modelTfIdf[lQB]

    dfData['sim'] = pd.Series(sims[QDocTfIdf]).astype("double")
    dfData.sort_values(by = ['sim'], ascending = False, inplace = True)
    print(dfData.loc[dfData.loc[:, "sim"] > 0.25, "Consumer complaint narrative"])
    
    