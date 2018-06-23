#General libraries
import pandas as pd

#Gensim libraries
from gensim.summarization import summarize
from gensim.summarization import keywords

#Import articles
filepath = "F:\\Code\\Code\\Natural Language Processing\\Summarization & KWE\\Extractive\\"
dfData = pd.read_csv(filepath + "Articles.csv", header = 0)
dfData = dfData.iloc[:100, :]

#Data Munging
dfData = pd.DataFrame(dfData.loc[:, "content"])

#Summarizer - gensim
def summary(df, sumRatio, keyCount):
    
    lSum = []
    
    for i in range(0, len(df)):
        summary = summarize(df.iloc[i, 0], ratio = sumRatio)
        kwords = keywords(dfData.iloc[i, 0], words = keyCount)
        lSum.append([summary, kwords])
        
    dfSummaries = pd.DataFrame(lSum)
    dfSummaries.columns = ["summaries", "keywords"]

    return dfSummaries

dfSumKey = summary(dfData, 0.1, 5)
