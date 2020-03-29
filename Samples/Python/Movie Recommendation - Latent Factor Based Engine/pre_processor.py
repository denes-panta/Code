# Dataset used: MovieLens: 20M dataset - https://grouplens.org/datasets/movielens/
# Sources used in the implementation:
#    https://www.youtube.com/watch?v=E8aMcwmqsTg
#    https://www.youtube.com/watch?v=E8aMcwmqsTg
#    https://www.youtube.com/watch?v=GGWBMg0i9d4
#    https://www.youtube.com/watch?v=HY3Csl52PfE

# Library import
import os
import random
import csv
from collections import defaultdict
from bidict import bidict

from statistics import mean 
import numpy as np
from scipy import sparse

# Set path to current folder
os.chdir(os.path.dirname(os.path.realpath('__file__')))

# Data Processor Class
class data_processor(object):
    
    # Read-in and save the user inputs
    def __init__(self, ratings, movies, n_records):

        # Set seed
        random.seed(101)
        
        # Filenames for ratings and movies
        self.fileRatings = ratings
        self.fileMovies = movies
        
        # Number of records to use in the model
        self.iNrecords = n_records

        # Record to split between train and test datasets          
        self.iNtrain = round(0.8 * n_records)
        
        # List to hold the ratings
        self.lRatings = []
        
        # Dictionary for movie name data
        self.dictNames = dict()
        
        # Dictionary to hold the movie_id - column_index pairs
        # Key: Movie_id - Value: Column_Index
        self.bidictMovies = bidict()

        # Dictionary to hold the user_id - row_index pairs
        # Key: User_id - Value: Row_Index
        self.bidictUsers = bidict()

        # Dictionary to hold the averages for the movies
        self.dictMovieAvgs = defaultdict(list)
        
        # Dictionary to hold the averages for the users
        self.dictUserAvgs = defaultdict(list)
        
        # Matrix to hold the user-item ratings
        self.mUserItem = np.zeros((0, 0))

        # Run the data preprocessing        
        self.runPreprocess()
 
    # Calculate the global average from a dictionary
    def calcDictGlobAvg(self, dct):
        fTotal = 0
        iCount = 0
        
        for k, v in dct.items():
            fTotal += sum(v)
            iCount += len(v)
            
        fMean = round(fTotal / iCount, 1)
    
        return fMean
    
    # Calculate the average rating in a dictionary
    def calcDictAvg(self, dct):
        for k, v in dct.items():
            dct[k] = round(mean(v), 1)
    
        return dct
       
    # Read-in the data line by line and populate the dictionaries
    def extractRatingsData(self):
        
        # Variables to keep track of the number of users_ids
        iUserInd = 0

        # Variables to keep track of the number of movie_ids
        iMovieInd = 0
        
        # Variable to keep track of the rows in the input file
        iRowCount = 0
        
        # Open the file with the ratings
        with open(self.fileRatings, encoding = 'utf8') as csv_file:
            # Define the csv file reader
            fileReader = csv.reader(csv_file, delimiter = ',')
            
            # Iterate through the file line by line
            for row in fileReader:
                # Skip the first line because of the heading

                if iRowCount == 0:
                    iRowCount += 1
                    continue

                # Stop the loop once the specified number of records have been reached
                if iRowCount > self.iNrecords:
                    break
                
                # For the selected record number, create the data dictionaries
                if iRowCount <= self.iNrecords:
                    # If the user_id doesn't exist in the dictionary add it
                    if self.bidictUsers.get(row[0]) == None:
                        
                        self.bidictUsers[row[0]] = str(iUserInd)
                        iUserInd += 1

                    # If the movie_id doesn't exist in the dictionary add it
                    if self.bidictMovies.get(row[1]) == None:
                        
                        self.bidictMovies[row[1]] = str(iMovieInd)
                        iMovieInd += 1
                
                # Collect the ratings into the avg dictionaries for the training data
                if iRowCount <= self.iNtrain:
                    # Append the ratings to the Avg dictionaries and rating list
                    sUser = self.bidictUsers[row[0]]
                    sMovie = self.bidictMovies[row[1]]

                    self.dictUserAvgs[sUser].append(float(row[2]))
                    self.dictMovieAvgs[sMovie].append(float(row[2]))
                
                # Increase the rowcount
                iRowCount += 1
        
        csv_file.close()
        
        # Calculate the averages: movie, user, global
        self.fGlobalAvg = self.calcDictGlobAvg(self.dictMovieAvgs)        
        self.dictMovieAvgs = self.calcDictAvg(self.dictMovieAvgs)
        self.dictUserAvgs = self.calcDictAvg(self.dictUserAvgs)

        
    # Create a sparse matrix for the data
    def createSparseMatrices(self):
        # Define empty sparse matrix for the training data
        self.smTrain = sparse.lil_matrix(
            (len(self.bidictMovies.values()), len(self.bidictUsers.values())),
            dtype = float)

       # Define empty sparse matrix for the test data 
        self.smTest = sparse.lil_matrix(
            (len(self.bidictMovies.values()), len(self.bidictUsers.values())),
            dtype = float)
        
        # Variable to keep track of the rows in the input file
        iRowCount = 0
        
        # Open the file with the ratings
        with open(self.fileRatings, encoding = 'utf8') as csv_file:
            # Define the csv file reader
            fileReader = csv.reader(csv_file, delimiter = ',')    
            
            # Iterate through the file line by line
            for row in fileReader:
                # Skip the first line because of the heading
                if iRowCount == 0:
                    iRowCount += 1
                    continue
                
                # For rows outside of the selected records
                if iRowCount > self.iNrecords:
                    break
                
                # For the train data, append the values to the train matrix
                if iRowCount <= self.iNtrain:
                    iRow = int(self.bidictMovies[row[1]])
                    iCol = int(self.bidictUsers[row[0]])
                    self.smTrain[iRow, iCol] = row[2]
                
                # For the test data, append the values to the test matrix
                else:
                    iRow = int(self.bidictMovies[row[1]])
                    iCol = int(self.bidictUsers[row[0]])
                    self.smTest[iRow, iCol] = row[2]
                    
                iRowCount += 1
                
        # Close file
        csv_file.close()
        
        # Convert lil to csc
        self.smTrain = self.smTrain.tocsc()
        self.smTest = self.smTest.tocsc()
    
    # Extract the movie_id - movie_name data into a dictionary
    def extractMetaData(self):
        with open(self.fileMovies, encoding = 'utf8') as csv_file:
            # Define the csv file reader
            fileReader = csv.reader(csv_file, delimiter = ',')
        
            # Iterate through the file line by line
            for row in fileReader:
                self.dictNames[row[0]] = row[1]
                
        csv_file.close()
    
    # Function to run the pre-processing steps
    def runPreprocess(self):
        self.extractRatingsData()
        self.createSparseMatrices()
        self.extractMetaData()
    
    # Function to get the train matrix
    def getTrain(self):
        return self.smTrain
    
    # Function to get the test matrix
    def getTest(self):
        return self.smTest
    
    # Function to get the movie names dictionary
    def getMeta(self):
        return self.dictNames
    
    # Get global mean
    def getGlobalMean(self):
        return self.fGlobalAvg

    # Get Movie Means
    def getMovieMeans(self):
        return self.dictMovieAvgs

    # Get User Means
    def getUserMeans(self):
        return self.dictUserAvgs

    # Get the translation dictionary for the users
    def getUsers(self):
        return self.bidictUsers

    # Get the translation dictionary for the movies
    def getMovies(self):
        return self.bidictMovies


if __name__ == "__main__":
    preProcess = data_processor('data/ratings.csv', 'data/movies.csv', 10000)
    