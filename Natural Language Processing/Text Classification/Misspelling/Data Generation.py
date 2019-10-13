# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 12:08:33 2018

@author: Denes
"""

#Libraries
import pandas as pd

#Import data
sPath = \
"F:/Code/Code/Natural Language Processing/Text Classification/Misspelling/Cities.csv"
dfData = pd.read_csv(sPath, encoding = 'UTF-8', sep = '|')
