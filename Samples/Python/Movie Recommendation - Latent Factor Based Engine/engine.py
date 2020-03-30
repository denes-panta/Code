# Dataset used: MovieLens: 20M dataset - https://grouplens.org/datasets/movielens/
# Sources used in the implementation:
#    https://www.youtube.com/watch?v=E8aMcwmqsTg
#    https://www.youtube.com/watch?v=E8aMcwmqsTg
#    https://www.youtube.com/watch?v=GGWBMg0i9d4
#    https://www.youtube.com/watch?v=HY3Csl52PfE

# Library import
import time
import os
import random
from math import sqrt

import numpy as np

import matplotlib.pyplot as plt

# Set path to current folder
os.chdir(os.path.dirname(os.path.realpath('__file__')))

# Import custom class
from pre_processor import data_processor

# Engine Class
class recommender(object):
    
    # Read-in and save the input parameters
    def __init__(self, train, test, names, movies, users, gm, mms, ums):
        # Set seed
        random.seed(101)
        
        # Train sparse matrix
        self.smTrain = train
        
        # Train COO matrix
        self.cooTrain = self.smTrain.tocoo()

        # Test sparse matrix
        self.smTest = test
        
        # Test COO matrix
        self.cooTest = self.smTest.tocoo()
        
        # Movie names dictionary
        self.dictNames = names
        
        # Movie - index map dictionary
        self.bidictMovies = movies
        
        # User - index map dictionary
        self.bidictUsers = users
        
        # Global Mean
        self.fGlobalMean = gm
        
        # Movie means
        self.dictMovieAvgs = mms
        
        # User means
        self.dictUserAvgs = ums


    # After the prediction, turn the predictions into real ratings
    def postProcessing(self):
        # Set too high ratings to the Max rating
        self.mPreds[self.mPreds > 5] = 5 
        
        # Set too low ratings to the Min rating
        self.mPreds[self.mPreds < 1] = 1 
        
        # Round the predictions to the nearest .5
        self.mPreds = np.round(self.mPreds * 2) / 2    


    # Predict the missing Ratings
    def predict(self, latent, lmbda, lr, epoch, timeit, verbose, plot):
        # latent = Latent factors in the model 
        self.iLatent = latent
                
        # lmbda = regulariation parameter
        self.fLambda = lmbda
        
        # lr = learning rate
        self.fLr = lr
        
        # epoch = number of times to run the optimization
        self.iEpoch = epoch
        
        # timeit = to time the function or not
        # verbose = print iterations and iteration duration
        # plot = to creat a plot of the error rate or not
         
        # Set up the clocks if specified
        print('Training')
        if timeit == True: start_time = time.time()
        if verbose == True: round_time = time.time()
        
        # Create random latent factors for first dimension (movies)
        U = 0.5 * np.random.rand(self.smTrain.shape[0], self.iLatent) 
        
        # Create random latent factors for second dimension (users)
        V = 0.5 * np.random.rand(self.smTrain.shape[1], self.iLatent) 
        
        # Create error table
        self.mError = np.zeros((epoch, 3))
        
        # Total Training Error
        self.fTrainingError = None
        
        # Train the model for the number of epochs
        for e in range(self.iEpoch):
            # Specify the total variables for the error and the gradiant
            fTotalError = 0
            fTotalGradU = 0
            fTotalGradV = 0

            # Iterate throught the sparse training matrix            
            for r, c, fRating in zip(
                    self.cooTrain.row, self.cooTrain.col, self.cooTrain.data):

                # Calculate the rating bias for the movie based on the mean rating
                fRavg = self.dictMovieAvgs[str(r)] - self.fGlobalMean
                
                # Calculate the rating bias for the user based on the mean rating
                fCavg = self.dictUserAvgs[str(c)] - self.fGlobalMean
                
                # Calculate the rating from the user-movie interractions
                fRatePred = np.dot(U[r, :], V[c, :].T)
                
                # Add the biases to the prediction
                fRatePred += fRavg + fCavg
                
                # Calculate the error between the real rating and the prediction
                fErr = sqrt((fRating - fRatePred) ** 2)
                
                # Create a regularization from the latent factors and the biases
                fReg = np.sum(U[r, :] ** 2) + np.sum(V[c, :] ** 2)
                fReg += fRavg ** 2 + fCavg ** 2
                
                # Add the regularization to the error term
                fErr += self.fLambda * sqrt(fReg)
                
                # Compute the gradient of the matrix slices for the rating
                fUgrad = np.sum(-2 * fErr * V[c, :] + 2 * self.fLambda * U[r, :])
                fVgrad = np.sum(-2 * fErr * U[r, :] + 2 * self.fLambda * V[c, :])
                
                # Add the gradient to the total gradients
                fTotalGradU += fUgrad
                fTotalGradV += fVgrad
                
                # Add the error to the total error
                fTotalError += fErr
            
            # If it is not the first epoch, set the recorded train error to double
            if self.fTrainingError == None:
                self.fTrainingError = fTotalError * 2
            
            # If the training error is worse than the previous epoch's
            # set the learning rate to 0  
            if self.fTrainingError <= fTotalError:

                self.fLr = 0
            
            # Othewise update the latent factors
            else:
                # SGD update latent factors 
                U -= self.fLr * fTotalGradU
                V -= self.fLr * fTotalGradV
            
                # Create the prediction matrix
                self.mPreds = np.dot(U, V.T)
            
                # Update the training error
                self.fTrainingError = fTotalError
                
            # Calculate the Train Error
            iRatingCount = 0
            fTrainRMSE = 0
            
            # Iterate through the training data
            for r, c, fOrigRating in zip(
                    self.cooTrain.row, self.cooTrain.col, self.cooTrain.data):
                
                # Sum the errors
                fPredRating = self.mPreds[r, c]                        
                fTrainRMSE += (fPredRating - fOrigRating) ** 2
                iRatingCount += 1
            
            # Calculate the RMSE
            fTrainRMSE /= iRatingCount
            fTrainRMSE = round(sqrt(fTrainRMSE), 4)

            # Calculate the Test Error
            iRatingCount = 0
            fTestRMSE = 0
            
            # Iterate through the test data
            for r, c, fOrigRating in zip(
                    self.cooTest.row, self.cooTest.col, self.cooTest.data):
                
                # Sum the errors
                fPredRating = self.mPreds[r, c]
                fTestRMSE += (fPredRating - fOrigRating) ** 2
                iRatingCount += 1
            
            # Calculate the RMSE
            fTestRMSE /= iRatingCount
            fTestRMSE = round(sqrt(fTestRMSE), 4)
            
            # Update the error matrix
            self.mError[e, 0] = e                        
            self.mError[e, 1] = fTrainRMSE
            self.mError[e, 2] = fTestRMSE
            
            # Print error after each epoch
            if (verbose == True): 
                print(
                    'Completed: ' + \
                    str(round((e + 1) / epoch * 100, 2)) + \
                    ' % - Duration: ' + \
                    str(np.round((time.time() - round_time) / 60, 3)) + \
                    ' minutes | ' + \
                    'Train RMSE: ' + str(self.mError[e, 1]) + ' | ' + \
                    'Test RMSE: ' + str(self.mError[e, 2]))
                
                # Update the time
                round_time = time.time()
        
        # Run post processing
        self.postProcessing()
        
        # Print the total run duration of the function
        if timeit == True: 
            elapsed_time = time.time() - start_time
            print('Total running time: ' + \
                  str(np.round(elapsed_time / 60, 2)) + ' minutes' + '\n')
                    
        # Plot errors
        if plot == True:
            
            plt.plot(
                self.mError[:, 0], 
                self.mError[:, 1], 
                'b', 
                label = 'Train')
            
            plt.plot(
                self.mError[:, 0], 
                self.mError[:, 2], 
                'r', 
                label = 'Test')
            
            plt.xlabel('Epochs')
            plt.ylabel('Error')
            plt.title('Error plot')
            plt.legend(loc = 'upper right')
            
            plt.savefig('error.png')
            
            plt.show()

        
    # Print recommendations
    def recommend(self, user_id):
        
        # Find the column id of the user from the input user id
        iColId = int(self.bidictUsers[str(user_id)])
        
        # Get the movie ratings for the user and also the row indexes
        lRatings = list(self.mPreds[:, iColId])
        lMovies = list(range(0, len(lRatings)))
        lWatched = list(self.smTrain[:, iColId].toarray())
        
        # Filtered movies and ratings
        lFRatings = []
        lFMovies = []
        
        # Filter out the already watched movies
        for n in lWatched:
            if n == 0:
                lFRatings.append(lRatings.pop(0))
                lFMovies.append(lMovies.pop(0))
                
            else:
                lRatings.pop(0)
                lMovies.pop(0)
        
        # Sort the movies based on the ratings
        lFRatings, lFMovies = zip(*sorted(zip(lFRatings, lFMovies), reverse = True))
        lFRatings = list(lFRatings)[:10]
        lFMovies = list(lFMovies)[:10]
        
        lRecommendations = []
        
        # Get the names of the recommended movies
        for n in lFMovies:
            iId = self.bidictMovies.inverse[str(n)]
            sMovie = self.dictNames[str(iId)]
            
            lRecommendations.append(sMovie)
        
        return lRecommendations
    
        
if __name__ == "__main__":
    # Run the pre-processing of the data
    preProcess = data_processor('data/ratings.csv', 'data/movies.csv', 300000)
    
    # Initialise the Recommendation Engine
    latfactEngine = recommender(
        preProcess.getTrain(),
        preProcess.getTest(),
        preProcess.getMeta(),
        preProcess.getMovies(),
        preProcess.getUsers(),
        preProcess.getGlobalMean(),
        preProcess.getMovieMeans(),
        preProcess.getUserMeans())
    
    # Predict the unkown ratings
    # Model scaling with additional data: 10% model completion run-time:
    #    1000 : 0.002
    #    2000 : 0.005
    #    3000 : 0.008
    #    5000 : 0.012
    #   10000 : 0.025
    #   50000 : 0.120
    #  100000 : 0.244
    #  200000 : 0.468
    #  300000 : 0.784
    # 1000000 : 2.442
    latfactEngine.predict(
            latent = 2,
            lr = 1e-7,
            epoch = 50,
            lmbda = 1e-3,
            timeit = True,
            verbose = True,
            plot = True)
    
    # Make recommendations based on a user_id
    lRecommendations = latfactEngine.recommend('10')
    
    # Print recommendations
    print('Recommended movies: ')
    print(*lRecommendations, sep = '\n')