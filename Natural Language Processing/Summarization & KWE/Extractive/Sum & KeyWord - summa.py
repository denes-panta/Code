#General libraries
import pandas as pd

#TextRank libraries
from summa.summarizer import summarize
from summa.keywords import keywords

#Import articles
filepath = "F:\\Code\\Code\\Natural Language Processing\\Summarization & KWE\\Extractive\\"
dfData = pd.read_csv(filepath + "Articles.csv", header = 0)
dfData = dfData.iloc[:100, :]

#Data Munging
dfData = pd.DataFrame(dfData.loc[:, "content"])

#Summarizer - TextRank
def summary(df, r, w):
    
    lSum = []
    
    print('\nSummarising & Extracting: \n', end='')
    
    for i in range(0, len(df)):
        print('\r', end='')
        print("Completed: " + str(round((i + 1)/len(df) * 100, 1)) + "%",
              end = "", 
              flush = True
              )
        summary = summarize(df.iloc[i, 0], ratio = r)
        kwords = keywords(dfData.iloc[i, 0], words = w)
        lSum.append([summary, kwords])
        
    dfSummaries = pd.DataFrame(lSum)
    dfSummaries.columns = ["summaries", "keywords"]

    return dfSummaries

dfSumKeySu = summary(dfData, 0.1, 5)
