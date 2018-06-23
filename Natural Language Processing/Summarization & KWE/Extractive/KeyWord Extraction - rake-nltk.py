#General libraries
import pandas as pd

#rake_nltklibraries
from rake_nltk import Metric, Rake

#Import articles
filepath = "F:\\Code\\Code\\Natural Language Processing\\Summarization & KWE\\Extractive\\"
dfData = pd.read_csv(filepath + "Articles.csv", header = 0)
dfData = dfData.iloc[:100, :]

#Data Munging
dfData = pd.DataFrame(dfData.loc[:, "content"])

#Keyword Extraction - rake-nltk
def extractKW(df, minL, maxL):
    
    lWords = []
    
    print('\nExtracting keywords: \n', end='')
    
    for i in range(0, len(df)):
        print('\r', end='')
        print("Completed: " + str(round((i + 1)/len(df) * 100, 1)) + "%",
              end = "", 
              flush = True
              )
        r = Rake(min_length = minL, 
                 max_length = maxL,
                 ranking_metric = Metric.DEGREE_TO_FREQUENCY_RATIO,
                 language = "english",
                 punctuations="“”’!?.,"
                 ) 
        r.extract_keywords_from_text(dfData.iloc[i, 0])
        
        lWords.append([r.get_ranked_phrases()])

    dfKeyWords = pd.DataFrame(lWords)
    dfKeyWords.columns = ["keywords"]

    return dfKeyWords

dfKeyWords = extractKW(dfData, 1, 5)
