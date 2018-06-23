#General libraries
import pandas as pd
import string
import os

#NLTK
from nltk.corpus import stopwords

#Pke
from pke import load_document_frequency_file as ldff
from pke import compute_document_frequency as cdf
from pke.unsupervised.graph_based import TopicRank as TR
from pke.unsupervised.graph_based import MultipartiteRank as MR
from pke.unsupervised.graph_based import PositionRank as PR
from pke.unsupervised.graph_based import SingleRank as SR
from pke.unsupervised.graph_based import TopicalPageRank as TPR
from pke.unsupervised.statistical import KPMiner as KPM
from pke.unsupervised.statistical import TfIdf

#Functions
#Write the texts from a dataframe to single txt files
def writeText(dfData, sPath):
    for i, text in enumerate(dfData.values.tolist()):
        fText = open(sPath + "input_txt\\" + (str(i) + ".txt"), 
                     "w", 
                     encoding = "utf-8"
                     )
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
def extractKW_U(sPath, extr, phrases = 10, stem = False, cdf = None):
    lWords = []
    sDataPath = sPath + "KeyWord Extraction - pke\\data\\"
    iL = len(os.listdir(sDataPath + "\\input_txt"))

    print('\nExtracting keywords: \n', end='')
    
    #Define position agruments and stop word list
    pos = set(['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'])
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
        
        #TfIdf
        if extr == TfIdf:
            n = 3
            extractor.candidate_selection(n = n, stoplist = lStopList)
            if cdf != None:
                fDocFreq = ldff(input_file = sDataPath + cdf)
            else:
                fDocFreq = None
            extractor.candidate_weighting(df = fDocFreq)
        
        #KPMiner
        elif extr == KPM:
            iLasf = 1
            iCutOff = 200
            extractor.candidate_selection(lasf = iLasf, cutoff = iCutOff)
            if cdf != None:
                fDocFreq = ldff(input_file = sDataPath + cdf)
            else:
                fDocFreq = None
            iAlpha = 2.3
            iSigma = 3.0
            extractor.candidate_weighting(df = fDocFreq, 
                                          alpha = iAlpha, 
                                          sigma = iSigma
                                          )
            
        #Single Rank    
        elif extr == SR:
            extractor.candidate_selection(pos = pos, stoplist = lStopList)
            extractor.candidate_weighting(window = 10, pos = pos)

        #Topic Rank
        elif extr == TR:
            extractor.candidate_selection(pos = pos, stoplist = lStopList)
            extractor.candidate_weighting(threshold = 0.74, method = 'average')
        
        # !!!!Broken code!!!! - Topical Page Rank
        elif extr == TPR:
            extractor.candidate_selection()
            extractor.candidate_weighting(window = 10)
        
        #Position Rank
        elif extr == PR:            
            extractor.candidate_selection()
            extractor.candidate_weighting(window = 10)
        
        #Multi-partite Rank
        elif extr == MR:
            extractor.candidate_selection(pos = pos, stoplist = lStopList)
            extractor.candidate_weighting(alpha = 1.1,
                                          threshold = 0.74,
                                          method = 'average'
                                          )
        #If non, raise exception
        else:
            raise ValueError('No such model')
            
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
dfKeyWords = extractKW_U(sInputPath, TfIdf, 10, False, cdf = "df_tsv.gz")
