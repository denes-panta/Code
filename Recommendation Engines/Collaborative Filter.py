#Dataset used: MovieLens: 20M dataset - https://grouplens.org/datasets/movielens/
#Method implemented: 
#	https://www.youtube.com/watch?v=1JRrCEgiyHM
#	https://www.youtube.com/watch?v=h9gpufJFF-0
#	https://www.youtube.com/watch?v=6BTLobS7AU8
#	https://www.youtube.com/watch?v=VZKMyTaLI00

import gc
import time
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.model_selection import train_test_split as tt_split
import matplotlib.pyplot as plt
import warnings
import random

random.seed(117)


def error(m_preds, m_origin): 
    # Error of the prediction
    # m_preds = predictions
    # m_origin = original data
    n = np.count_nonzero(~np.isnan(m_origin)) # Number of non-zero, non NaN values in the original data
    RMSE = np.round(sqrt(np.nansum((m_preds - m_origin)**2) / n), 4) # Root Mean Squared Error
    MAE = np.round(np.nansum(np.abs(m_preds - m_origin)) / n, 4) # Mean Absolute Error

    return list([RMSE, MAE])


def u_col_print(df, cols = [], verbose = False): 
    # Returns the unique/max number of elements in specified columns in DF - Whichever is larger
    # df = dataframe of values
    # cols = columns that need to be checked
    # verbose = prints the max and the unique number of items  
    col_size = [0] * len(cols) # Size of the columns

    for i, lst in enumerate(cols): #Iterate through the dataframe
        n = df[lst].unique().shape[0] # Number of unique values in the specified column
        m = max(df[lst]) # Max values in the specified column
        if verbose == True: #Print details of the columns
            print('Number of unique items in %s = %d' % (lst, n))
            print('Largest item %s = %d' % (lst, m) + '\n')
        if (max(df[lst]) == df[lst].unique().shape[0]): #Check to see if max == number of unique values
            col_size[i] = df[lst].unique().shape[0] # if True, return number of unique values 
        else:
            col_size[i] = max(df[lst]) #if False, return max

    return col_size


def cc_matrix(df, col_size = []): 
    # Creates the user-item matrix       
    # df = dataframe
    # col_size = size of the columns
    matrix = np.zeros((col_size[0], col_size[1])) # Create a zero matrix, dims = input col_size of function u_col_print 
        
    for r in df.itertuples(index = False): #populate the matrix by iterating through the dataframe
        matrix[r[0]-1, r[1]-1] = r[2]
            
    return matrix #return the populated matrix


def normalize(matrix, center = True, scale = True, zeros = True):
    # Centering, Scaling and replacing NaNs with 0s, by rows.
    # matrix = input matrix
    # center = mean_center
    # scale = divide by std
    # zeros = replace NaN with 0s
    matrix[matrix == 0.0] = np.NaN
    
    v_mean = np.nanmean(matrix, axis = 1).reshape((matrix.shape[0], 1))
    v_std = np.nanstd(matrix, axis = 1).reshape((matrix.shape[0], 1))
    
    if (center == True): matrix = (matrix - v_mean)
    if (scale == True): matrix = matrix / v_std 
    if (zeros == True): matrix[np.isnan(matrix) == True] = 0
    
    return matrix
    

def dimension(train, test, transpose = True):
    # Creates the correlation matrix for the columns and Transposes the matrices if specified
    # train = train matrix
    # test = test matrix
    # transpose = to give some flexibility for creating user-user or item-item correlation
    if transpose == False: # Correlation table, and replace 0s with NaN in both train and test
        m_cor = np.round(pd.DataFrame(normalize(m_train, center = True, scale = False, zeros = False)).corr(), 2)
        train[train == 0] = np.NaN
        test[test == 0] = np.NaN
    elif transpose == True: # Correlation table, and replace 0s with NaN in both train and test + transpose the matrices
        m_cor = np.round(pd.DataFrame(normalize(m_train.T, center = True, scale = False, zeros = False)).corr(), 2)
        train = train.T
        train[train == 0] = np.NaN
        test = test.T
        test[test == 0] = np.NaN

    return train, test, m_cor #returns, train, test and correlation tables


def predict(train, m_cor,  k = 100, timeit = False, verbose = False):
    # Predict the missing Ratings
    # matrix = user/item matrix
    # corr = correlation of columns or rows
    # k = if integer, the number of nearest neighbours, 
    #     if float the minimum correlation between the items/users
    # timeit = to time the function or not
    if timeit == True: start_time = time.time()
    if verbose == True: round_time = time.time()

    m_preds = np.empty((train.shape[0], train.shape[1]))
    m_preds[:] = np.NaN

    oall_mean = np.nanmean(train) # mean of every non NaN values
    row_means = np.nanmean(train, axis = 1).reshape((train.shape[0]), 1) # mean of rows of every non NaN values
    col_means = np.nanmean(train, axis = 0).reshape((train.shape[1]), 1) # mean of cols of every non NaN values
            
    for r in range(train.shape[0]): # iterate over every row
        df_ratings = pd.DataFrame(train[r, :]) # select the values of the r-th item/user
        
        for c in range(train.shape[1]): # iterate over every col
            df_correls = pd.DataFrame(m_cor.iloc[:, c]) # select the correlation column between two items or users
            
            if type(k) == int: # select the kth nearest neighbours
                l_ind = np.asarray(df_ratings.index[(~np.isnan(df_ratings.iloc[:, 0])) & \
                                                    (~np.isnan(df_correls.iloc[:, 0])) & \
                                                    (df_correls.iloc[:, 0] > 0.0) & \
                                                    (df_correls.index[:] != c)]).tolist()
                m_indices = np.asarray([y[0] for y in sorted(zip(l_ind, df_correls.iloc[l_ind, 0]), key = lambda x: x[1], reverse = True)])[:k]
                l_indices = m_indices.tolist() # list of k-th nearest neighbours
            elif type(k) == float: # selects the neighbours that's correlation is above k
                l_ind = np.asarray(df_ratings.index[(~np.isnan(df_ratings.iloc[:, 0])) & \
                                                    (~np.isnan(df_correls.iloc[:, 0])) & \
                                                    (df_correls.iloc[:, 0] > k) & \
                                                    (df_correls.index[:] != c)]).tolist()
                l_indices = l_ind  # list of neighbours that's correlation is above k
                
            sim = np.NaN # set similarities to NaN, every iteration

            if len(l_indices) != 0: # only calculate sim if there are nearest neighbours
                m_ratings = df_ratings.iloc[l_indices, 0].as_matrix().reshape(len(l_indices), 1) # select the appropriate ratings (values and correlation both exist)
                m_similar = df_correls.iloc[l_indices, 0].as_matrix().reshape(len(l_indices), 1) # select the appropriate correlations (values and correlation both exist)
 
                m_adjust = (col_means[l_indices] - oall_mean) + (row_means[r] - oall_mean) + oall_mean # calculate the scores for the neighbours
                m_adjust = m_adjust.reshape(len(l_indices), 1) # reshape the values to a (n,1) array
                
                m_scores = m_similar * (m_ratings - m_adjust) # Calculate the similarity score part 1
                sim = np.sum(m_scores) / np.sum(m_similar) # Calculate the similarity score part 2

            m_glob_r = (row_means[r] - oall_mean) # global row adjustment for the r-th item/user
            m_glob_c = (col_means[c] - oall_mean) # global col adjustment for the r-th item/user

            m_global = oall_mean # Calculate the overall score for the r-th user, c-th item or the other way around 
            if (np.isnan(m_glob_r)) == False: m_global += m_glob_r # -||-
            if (np.isnan(m_glob_c)) == False: m_global += m_glob_c # -||-
            if (np.isnan(sim)) == False: m_global += sim # -||-
            
            m_preds[r, c] = np.round(m_global*2) / 2 # Round the prediction to the nearest .5
            if m_preds[r, c] > 5: m_preds[r, c] = 5 # set the too high prediction to the max value
            if m_preds[r, c] < 1: m_preds[r, c] = 1 # set the too low prediction to the min value
            
        if (verbose == True) and (r % np.round(m_train.shape[0]/10) == (np.round(m_train.shape[0]/10)-1)): #Prints duration for every 10% done 
            print('Completed: ' + str(np.round(r/m_train.shape[0]*100)) + ' % - Duration: ' + str(np.round((time.time() - round_time) / 60, 3)) + ' minutes')
            round_time = time.time()
        
    if timeit == True: #Prints run overall time for the entire function
        elapsed_time = time.time() - start_time
        print('Total running time: ' + str(np.round(elapsed_time / 60 , 2)) + ' minutes' + '\n')
    
    print('Train RMSE: ' + str(error(m_preds, train)[0]) + '| Train MAE: ' + str(error(m_preds, train)[1])) # prints the errors

    return m_preds # returns the predictions


##### Not useful - failed experiment #################################################################
#def tuner(train = None, test = None, m_cor = None, sequence = None , plot = False, timeit = False):
#    #Let's you try out various values for variable k for the function predict (here: sequence)
#    
#    table = np.zeros((len(sequence), 3))
#
#    d = 0
#
#    for i in sequence.tolist():
#        table[d, 0] = i
#        m_preds = predict(train, m_cor, k = i, timeit = timeit, verbose = False)
#        table[d, 1] = error(m_preds, test)[0]
#        table[d, 2] = error(m_preds, test)[1]
#        d += 1
#
#    if plot == True: 
#        plt.plot(table[:, 0], table[:, 1], 'b', label = 'RMSE')
#        plt.plot(table[:, 0], table[:, 2], 'r', label = 'MSE')
#        if type(i) == float: plt.xlabel('Correlation floor')
#        if type(i) == int: plt.xlabel('KNN')
#        plt.ylabel('Error')
#        plt.title('Tuner plot')
#        plt.legend(loc = 'upper right')
#        
#    return table
######################################################################################################


def recommend(user_id, train, preds, names, transposed = True):
    # Makes the movie recommendations
    # user_id = user id of the user for the recommendations
    # train = train data
    # preds = predictions
    # names = dataframe of the movie names
    # transposed = UserIDs are in rows or columns
    if transposed == True:  # If the users are in the columns 
        df_past = pd.DataFrame(train[:, user_id])
        df_user = pd.DataFrame(preds[:, user_id])
    if transposed == False: # If the users are in the rows
        df_past = pd.DataFrame(train[user_id, :])
        df_user = pd.DataFrame(preds[user_id, :])
    
    l_past = df_past.index[~np.isnan(df_past.iloc[:, 0])].tolist() # Watched movies
    l_top1 = df_user.loc[(~df_user.index[:].isin(l_past)) & (df_user.iloc[:, 0] >= 4.5), 0].index[:].tolist() # First Tier recommendations
    l_top2 = df_user.loc[(~df_user.index[:].isin(l_past)) & (df_user.iloc[:, 0] >= 3.5) & (df_user.iloc[:, 0] <= 4.0), 0].index[:].tolist() # Second Tier recommendations
    
    print('Top suggestions for user: ' + str(user_id)) # prints the suggestions 
    print(names.iloc[names.index[names.iloc[:, 0].isin(list(random.sample(l_top1, 5)))], [1, 2]])
    print('')
    print('User "%d" might also like: ' % (user_id))
    print(names.iloc[names.index[names.iloc[:, 0].isin(list(random.sample(l_top2, 10)))], [1, 2]])
    
    return list([l_top1, l_top2]) # returns the IDs of the recommended movies


#Script
#Taking only the a first X Users and the first Y Movies, because the solution doesn't scale well to large datasets.
warnings.filterwarnings('ignore')

df_ratings = pd.read_csv("F:/Code/Recommendation System/ratings.csv")
df_ratings = df_ratings.loc[df_ratings['userId'] <= 1000]
df_ratings = df_ratings.loc[df_ratings['movieId'] <= 1000]

df_movies = pd.read_csv("F:/Code/Recommendation System/movies.csv")
df_movies = df_movies.loc[df_movies['movieId'] <= 1000]

df_train, df_test = tt_split(df_ratings, test_size = 0.7)

m_train = cc_matrix(df_train, u_col_print(df_ratings, ['userId', 'movieId']))
m_test = cc_matrix(df_test, u_col_print(df_ratings, ['userId', 'movieId']))

del df_ratings, df_test, df_train
gc.collect()

m_train, m_test, m_cor = dimension(m_train, m_test, transpose = False)

m_preds = predict(m_train, m_cor, k = 100, timeit = True, verbose = True)
print('Test RMSE: ' + str(error(m_preds, m_test)[0]) + '| Test MAE: ' + str(error(m_preds, m_test)[1]))

l_recommendations = recommend(42, m_train, m_preds, df_movies, transposed = False)
