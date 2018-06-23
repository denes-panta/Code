#General libraries
import pandas as pd
import string
import os

#NLTK
from nltk.corpus import stopwords

#Pke
from pke import load_document_frequency_file as ldff
from pke import compute_document_frequency as cdf
from pke.supervised import Kea
from pke.supervised import WINGNUS

#Functions
#Write the texts from a dataframe to single txt files
def writeText(dfData, sPath):
    for i, text in enumerate(dfData.values.tolist()):
        fText = open(sPath + "input_txt\\" + (str(i) + ".txt"), "w", encoding = "utf-8")
        fText.write(text[0])
        fText.close()

    return None

#Create document frequency counts
def createDF(sPath, sOutFile):
    input_dir = sPath + "KeyWord Extraction - pke\\data\\input_txt\\"
    output_file = sPath + "KeyWord Extraction - pke\\data\\" + sOutFile
    lStopList = list(string.punctuation)
    lStopList += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    lStopList += stopwords.words('english')

    cdf(input_dir = input_dir,
        output_file = output_file,
        format = "raw",
        use_lemmas = True,
        stemmer = "porter",
        stoplist = lStopList,
        delimiter='\t',
        extension = 'txt',
        n = 5)              

#Unsupervised extraction
def extractKW_S(sPath, extr, phrases = 10, stem = False, cdf = None):
    lWords = []
    sDataPath = sPath + "KeyWord Extraction - pke\\data\\"
    iL = len(os.listdir(sDataPath + "\\input_txt"))

    print('\nExtracting keywords: \n', end='')
    
    #Define position agruments and stop word list
    lStopList = list(string.punctuation)
    lStopList += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    lStopList += stopwords.words('english')
    
    for i in range(0, iL):
        #Status update on screen
        print('\r', end='')
        print("Completed: " + str(round((i+1)/iL * 100, 1)) + "%",
              end = "", 
              flush = True
              )
        
        #Create extractor and load document
        extractor = extr(input_file = sDataPath + "input_txt\\" + str(i) + ".txt")
        extractor.read_document(format = "raw", use_lemmas = True)
        
        #Kea
        if extr == Kea:
            extractor.candidate_selection(stoplist = lStopList)
            if cdf != None:
                fDocFreq = ldff(input_file = sDataPath + cdf)
            else:
                fDocFreq = None
            extractor.feature_extraction(df = fDocFreq, training = False)
            
        #WINGNUS
        elif extr == WINGNUS:
            extractor.candidate_selection()
            if cdf != None:
                fDocFreq = ldff(input_file = sDataPath + cdf)
            else:
                fDocFreq = None
            extractor.feature_extraction(df = fDocFreq, training = False)

        lKPhrases = extractor.get_n_best(n = phrases, stemming = stem) 
        lWords.append([lKPhrases])

    dfKeyWords = pd.DataFrame(lWords)
    dfKeyWords.columns = ["keywords"]

    return dfKeyWords

#Import articles
sInputPath = "F:\\Code\\Code\\Natural Language Processing\\Summarization & KWE\\Extractive\\"
dfData = pd.read_csv(sInputPath + "Articles.csv", header = 0)
dfData = dfData.iloc[:100, :]

#Data Munging
dfData = pd.DataFrame(dfData.loc[:, "content"])

#Write articles to file in order to comply with the input data format
writeText(dfData, sInputPath + "KeyWord Extraction - pke\\data\\")

#Keyword Extraction - pke
createDF(sInputPath, "df_tsv.gz")
dfKeyWords = extractKW_S(sInputPath, WINGNUS, 10, False, "df_tsv.gz")
