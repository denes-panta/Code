#General libraries
import pandas as pd

#TextRank libraries
import textrank

#Import articles
filepath = "F:\\Code\\Code\\Natural Language Processing\\Summarization & KWE\\Extractive\\"
dfData = pd.read_csv(filepath + "Articles.csv", header = 0)
dfData = dfData.iloc[:100, :]

#Data Munging
dfData = pd.DataFrame(dfData.loc[:, "content"])

#Summarizer - TextRank
def summary(df):
    
    lSum = []
    
    print('\nSummarising & Extracting: \n', end='')
    
    for i in range(0, len(df)):
        print('\r', end='')
        print("Completed: " + str(round((i + 1)/len(df) * 100, 1)) + "%",
              end = "", 
              flush = True
              )
        summary = textrank.extract_sentences(df.iloc[i, 0])
        kwords = textrank.extract_key_phrases(dfData.iloc[i, 0])
        lSum.append([summary, kwords])
        
    dfSummaries = pd.DataFrame(lSum)
    dfSummaries.columns = ["summaries", "keywords"]

    return dfSummaries

dfSumKeyTr = summary(dfData)
