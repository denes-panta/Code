#General libraries
import pandas as pd
import re
from math import sqrt, ceil

#summy libraries
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer 
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer

#Import articles
filepath = "F:\\Code\\Code\\Natural Language Processing\\Summarization & KWE\\Extractive\\Articles.csv"
dfData = pd.read_csv(filepath, header = 0)
dfData = dfData.iloc[:100, :]

#Data Munging
dfData["sentences"] = dfData.loc[:, "content"].apply(lambda x: len(re.split(r'[.!?]+', x)))
dfData = dfData.loc[:, ("content", "sentences")]

#Summarizer - summy
def summary(df, summarizer):
    
    def sentCount(number):
        return ceil(sqrt(number))

    lSum = []
    
    for i in range(0, len(df)):
        parser = PlaintextParser.from_string(df.iloc[i, 0], Tokenizer("english"))
    
        summary = summarizer(parser.document, sentCount(df.iloc[i, 1]))
        
        lSum.append("".join(map(str, summary)))
    
    dfSummaries = pd.DataFrame(lSum)
    dfSummaries.columns = ["summaries"]

    return dfSummaries

#SumBasic
dfSumBasic = summary(dfData, SumBasicSummarizer())
    
#LexRank
dfLexRank = summary(dfData, LexRankSummarizer())

#TextRank
dfTextRank = summary(dfData, TextRankSummarizer())

#Lsa
dfLsa = summary(dfData, LsaSummarizer())

#Luhn
dfLuhn = summary(dfData, LuhnSummarizer())

