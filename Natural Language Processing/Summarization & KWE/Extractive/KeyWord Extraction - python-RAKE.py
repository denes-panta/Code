#General libraries
import pandas as pd

#Rake libraries
import RAKE

#Import articles
filepath = "F:\\Code\\Code\\Natural Language Processing\\Summarization & KWE\\Extractive\\"
dfData = pd.read_csv(filepath + "Articles.csv", header = 0)
dfData = dfData.iloc[:100, :]

#Data Munging
dfData = pd.DataFrame(dfData.loc[:, "content"])

#Keyword Extraction - RAKE
def extractKW(df, minCh, maxW, minF):
    
    lWords = []
    
    print('\nExtracting keywords: \n', end='')

    lStopWords = list(set(RAKE.SmartStopList() + \
                          RAKE.FoxStopList() + \
                          RAKE.NLTKStopList() + \
                          RAKE.MySQLStopList()
                          ))
    
    for i in range(0, len(df)):
        print('\r', end='')
        print("Completed: " + str(round((i + 1)/len(df) * 100, 1)) + "%",
              end = "", 
              flush = True
              )

        Rake = RAKE.Rake(lStopWords)
        kwords = Rake.run(dfData.iloc[i, 0], 
                          minCharacters = minCh, 
                          maxWords = maxW, 
                          minFrequency = minF
                          )

        lWords.append([kwords])
        
    dfKeyWords = pd.DataFrame(lWords)
    dfKeyWords.columns = ["keywords"]

    return dfKeyWords

dfKeyWords = extractKW(dfData, 1, 5, 1)
