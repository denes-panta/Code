#Dataset used: MovieLens: 20M dataset - https://grouplens.org/datasets/movielens/
#Sources used in the implementation:
#    https://www.youtube.com/watch?v=E8aMcwmqsTg
#    https://www.youtube.com/watch?v=E8aMcwmqsTg
#    https://www.youtube.com/watch?v=GGWBMg0i9d4
#    https://www.youtube.com/watch?v=HY3Csl52PfE
#    https://www.slideshare.net/sscdotopen/latent-factor-models-for-collaborative-filtering

import gc
import time
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.model_selection import train_test_split as tt_split
import matplotlib.pyplot as plt
from scipy import sparse
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


def sparsity(df, cols = []):
    # Check the sparsity of the dataframe
    v_n_col = len(df[cols[0]].unique()) # unique values of the col column
    v_n_row = len(df[cols[1]].unique()) # unique values of the row column
    v_sparsity = np.round((len(df) / (v_n_col*v_n_row)) * 100, 2) #Calculate the sparsity
    print('Sparsity: %.2f %%' % (v_sparsity))
    
    
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


def csr_rows(csr_matrix): 
    # Calculate the row indexes of the CSR Matrix
    l_row = []
    
    for i, v in enumerate(csr_matrix.indptr):
        try: # Use try-except because the loop return an error for the last item
            for j in range(csr_matrix.indptr[i+1] - csr_matrix.indptr[i]):
                l_row.append(i)
        except:
            pass
    
    return l_row


def predict(train, test, latent = 10, lmbda = 0.1, lr = 0.01, epoch = 10, timeit = False, verbose = False, plot = True):
    #Predict the missing Ratings
    #matrix = user/item matrix
    #corr = correlation of columns or rows
    # k = if integer, the number of nearest neighbours, 
    #     if float the minimum correlation between the items/users
    # timeit = to time the function or not
    # verbose = print iterations and iteration duration
        
    if timeit == True: start_time = time.time()
    if verbose == True: round_time = time.time()

    U = 3 * np.random.rand(train.shape[0], latent) # Create random latent factors for first dimension
    V = 3 * np.random.rand(train.shape[1], latent) # Create random latent factors for second dimension

    m_error = np.zeros((epoch, 3)) # Create error table

    s_train = sparse.csr_matrix(train) # transform train matrix to sparse
    train[train == 0] = np.NaN  # replace 0s with NaNs for mean calculations
    
    v_Gmean = np.nanmean(train) # Global mean
    m_Cmean = np.nanmean(train, axis = 0).reshape((1, train.shape[1])) - v_Gmean # Column means
    m_Cmean[np.isnan(m_Cmean)] = 0
    m_Rmean = np.nanmean(train, axis = 1).reshape((train.shape[0], 1)) - v_Gmean # Row means
    m_Rmean[np.isnan(m_Rmean)] = 0
    
    train[np.isnan(train)] = 0 # replace NaNs with 0s
    
    l_cols = list(s_train.indices) # column indices of sparse matrix
    l_rows = csr_rows(s_train) # row indices of sparse matrix
    
    for e in range(epoch): # run through the number of epochs
        for s, v in enumerate(s_train): # run through the nmber of row-column pairs in the sparse matrix
            r = l_rows[s] # row
            c = l_cols[s] # column
            
            err = s_train[r, c] - (v_Gmean + m_Rmean[r, 0] + m_Cmean[0, c] +  np.dot(U[r, :], V[c, :].T)) # calculate error

            U[r, :] += lr * (err * V[c, :] - lmbda * U[r, :]) # SGD update latent factors
            V[c, :] += lr * (err * U[r, :] - lmbda * V[c, :]) # SGD update latent factors
            m_Rmean[r, 0] += lr * (err - lmbda * m_Rmean[r, 0]) # SGD update row mean
            m_Cmean[0, c] += lr * (err - lmbda * m_Cmean[0, c]) # SGD update col mean
            
        m_preds = np.dot(U, V.T) # Dot product of U and V Transposed
        
        m_error[e, 0] = e # Update error table
        m_error[e, 1] = error(m_preds, m_train)[0] # Update error table
        m_error[e, 2] = error(m_preds, m_test)[0] # Update error table
        
        if (verbose == True) and (e % (epoch / 10) == (epoch / 10)-1): # Print Duration after every 10% Done
            print('Completed: ' + str((e+1)/epoch*100) + ' % - Duration: ' + str(np.round((time.time() - round_time) / 60, 3)) + ' minutes')
            round_time = time.time()
    

    m_preds[m_preds > 5] = 5 # Set too high ratings to the Max rating
    m_preds[m_preds < 1] = 1 # Set too low ratings to the Min rating

    m_preds = np.round(m_preds*2) / 2 # round to the nearest .5

    if timeit == True: # Print the total run duration of the function
        elapsed_time = time.time() - start_time
        print('Total running time: ' + str(np.round(elapsed_time / 60, 2)) + ' minutes' + '\n')
    
    print('Train RMSE: ' + str(error(m_preds, train)[0]) + '| Train MAE: ' + str(error(m_preds, train)[1])) # printtrain Error
    print('Test RMSE: ' + str(error(m_preds, test)[0]) + '| Test MAE: ' + str(error(m_preds, test)[1])) # print test Error
    
    if plot == True: # Plot errors
        plt.plot(m_error[:, 0], m_error[:, 1], 'b', label = 'Train')
        plt.plot(m_error[:, 0], m_error[:, 2], 'r', label = 'Test')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Error plot')
        plt.legend(loc = 'upper right')
        
    return m_preds # return predictions


def recommend(user_id, train, preds, names):
    df_past = pd.DataFrame(train[user_id, :]) # Wathed movies ratings for the user
    df_past[df_past == 0] = np.NaN 
    df_user = pd.DataFrame(preds[user_id, :]) # predicted ratings for the user

    l_past = df_past.index[~np.isnan(df_past.iloc[:, 0])].tolist() #filtered ratings without NaN
    l_top1 = df_user.loc[(~df_user.index[:].isin(l_past)) & (df_user.iloc[:, 0] >= 4.5), 0].index[:].tolist() # Tier one suggestions
    l_top2 = df_user.loc[(~df_user.index[:].isin(l_past)) & (df_user.iloc[:, 0] >= 3.5) & (df_user.iloc[:, 0] <= 4.0), 0].index[:].tolist() # Tier two suggestions
    
    v_t1r = len(l_top1) # tier one recommendations
    if v_t1r > 5: v_t1r = 5
    v_t2r = len(l_top2) # tier two recommendations
    if v_t2r > 5: v_t2r = 10
    
    print('Top suggestions for user: ' + str(user_id)) # print suggestions
    print(names.iloc[names.index[names.iloc[:, 0].isin(list(random.sample(l_top1, v_t1r)))], [1, 2]])
    print('')
    print('User "%d" might also like: ' % (user_id))
    print(names.iloc[names.index[names.iloc[:, 0].isin(list(random.sample(l_top2, v_t2r)))], [1, 2]])
    
    return list([l_top1, l_top2]) # Return the suggested movie indices


#Taking only the a first X Users and the first Y Movies, because the solution doesn't scale well to large datasets.
warnings.filterwarnings('ignore')

df_ratings = pd.read_csv("F:/Code/Recommendation System/ratings.csv")
df_ratings = df_ratings.loc[df_ratings['userId'] <= 2000]
df_ratings = df_ratings.loc[df_ratings['movieId'] <= 2000]

df_movies = pd.read_csv("F:/Code/Recommendation System/movies.csv")
df_movies = df_movies.loc[df_movies['movieId'] <= 2000]

df_train, df_test = tt_split(df_ratings, test_size = 0.7)

m_train = cc_matrix(df_train, u_col_print(df_ratings, ['userId', 'movieId']))
m_test = cc_matrix(df_test, u_col_print(df_ratings, ['userId', 'movieId']))

sparsity(df_ratings, ['userId', 'movieId'])

del df_ratings, df_test, df_train
gc.collect()

m_preds = predict(m_train, m_test, latent = 2, lr = 1e-2, epoch = 1000, lmbda = 1e-1, timeit = True, verbose = True, plot = True)


