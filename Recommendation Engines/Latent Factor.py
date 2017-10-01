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
    #Error of the prediction
    n = np.count_nonzero(~np.isnan(m_origin))
    RMSE = np.round(sqrt(np.nansum((m_preds - m_origin)**2) / n), 4)
    MAE = np.round(np.nansum(np.abs(m_preds - m_origin)) / n, 4)

    return list([RMSE, MAE])


def sparsity(df, cols = []):
    v_n_col = len(df[cols[0]].unique())
    v_n_row = len(df[cols[1]].unique())
    v_sparsity = np.round((len(df) / (v_n_col*v_n_row)) * 100, 2)
    print('Sparsity: %.2f %%' % (v_sparsity))
    
    
def u_col_print(df, cols = [], verbose = False): 
    #Returns the unique/max number of elements in specified columns in DF
    #Whichever is larger
    
    for i, lst in enumerate(cols):
        try:
            n = df[lst].unique().shape[0]
            m = max(df[lst])
            if verbose == True:
                print('Number of unique items in %s = %d' % (lst, n))
                print('Largest item %s = %d' % (lst, m) + '\n')
            if (max(df[lst]) == df[lst].unique().shape[0]):
                cols[i] = df[lst].unique().shape[0]
            else:
                cols[i] = max(df[lst])
        except:
            print('No such column: %s' % (lst))
    
    return cols


def cc_matrix(df, cols = []): 
    #Creates the user-item matrix       
    
    matrix = np.zeros((cols[0], cols[1]))
        
    for r in df.itertuples(index = False):
        matrix[r[0]-1, r[1]-1] = r[2]
            
    return matrix


def csr_rows(csr_matrix):    
    l_row = []
    
    for i, v in enumerate(csr_matrix.indptr):
        try:
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
    #timeit = to time the function or not
        
    if timeit == True: start_time = time.time()
    if verbose == True: round_time = time.time()

    U = 3 * np.random.rand(train.shape[0], latent)
    V = 3 * np.random.rand(train.shape[1], latent)

    m_error = np.zeros((epoch, 3))

    s_train = sparse.csr_matrix(train)
    train[train == 0] = np.NaN
    
    v_Gmean = np.nanmean(train)
    m_Cmean = np.nanmean(train, axis = 0).reshape((1, train.shape[1])) - v_Gmean
    m_Cmean[np.isnan(m_Cmean)] = 0
    m_Rmean = np.nanmean(train, axis = 1).reshape((train.shape[0], 1)) - v_Gmean
    m_Rmean[np.isnan(m_Rmean)] = 0
    
    train[np.isnan(train)] = 0
    
    l_cols = list(s_train.indices) 
    l_rows = csr_rows(s_train)
    
    for e in range(epoch):
        for s, v in enumerate(s_train):
            r = l_rows[s]
            c = l_cols[s]
            
            err = s_train[r, c] - (v_Gmean + m_Rmean[r, 0] + m_Cmean[0, c] +  np.dot(U[r, :], V[c, :].T))

            U[r, :] += lr * (err * V[c, :] - lmbda * U[r, :])
            V[c, :] += lr * (err * U[r, :] - lmbda * V[c, :])
            m_Rmean[r, 0] += lr * (err - lmbda * m_Rmean[r, 0])
            m_Cmean[0, c] += lr * (err - lmbda * m_Cmean[0, c])
            
        m_preds = np.dot(U, V.T)
        
        m_error[e, 0] = e
        m_error[e, 1] = error(m_preds, m_train)[0]
        m_error[e, 2] = error(m_preds, m_test)[0]
        
        if (verbose == True) and (e % (epoch / 10) == (epoch / 10)-1): 
            print('Completed: ' + str((e+1)/epoch*100) + ' % - Duration: ' + str(np.round((time.time() - round_time) / 60, 3)) + ' minutes')
            round_time = time.time()
    

    m_preds[m_preds > 5] = 5
    m_preds[m_preds < 1] = 1

    m_preds = np.round(m_preds*2) / 2

    if timeit == True: 
        elapsed_time = time.time() - start_time
        print('Total running time: ' + str(np.round(elapsed_time / 60, 2)) + ' minutes' + '\n')
    
    print('Train RMSE: ' + str(error(m_preds, train)[0]) + '| Train MAE: ' + str(error(m_preds, train)[1]))
    print('Test RMSE: ' + str(error(m_preds, test)[0]) + '| Test MAE: ' + str(error(m_preds, test)[1]))
    
    if plot == True: 
        plt.plot(m_error[:, 0], m_error[:, 1], 'b', label = 'Train')
        plt.plot(m_error[:, 0], m_error[:, 2], 'r', label = 'Test')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Error plot')
        plt.legend(loc = 'upper right')
        
    return m_preds


def recommend(user_id, train, preds, names, transposed = True):
    df_past = pd.DataFrame(train[user_id, :])
    df_user = pd.DataFrame(preds[user_id, :])
    
    l_past = df_past.index[~np.isnan(df_past.iloc[:, 0])].tolist()
    l_top1 = df_user.loc[(~df_user.index[:].isin(l_past)) & (df_user.iloc[:, 0] >= 4.5), 0].index[:].tolist()
    l_top2 = df_user.loc[(~df_user.index[:].isin(l_past)) & (df_user.iloc[:, 0] >= 3.5) & (df_user.iloc[:, 0] <= 4.0), 0].index[:].tolist()
    
    print('Top suggestions for user: ' + str(user_id))
    print(names.iloc[names.index[names.iloc[:, 0].isin(list(random.sample(l_top1, 5)))], [1, 2]])
    print('')
    print('User "%d" might also like: ' % (user_id))
    print(names.iloc[names.index[names.iloc[:, 0].isin(list(random.sample(l_top2, 10)))], [1, 2]])
    
    return list([l_top1, l_top2])


#Taking only the a first X Users and the first Y Movies, because the solution doesn't scale well to large datasets.
warnings.filterwarnings('ignore')

df_ratings = pd.read_csv("F:/Code/Recommendation System/ratings.csv")
df_ratings = df_ratings.loc[df_ratings['userId'] <= 10000]
df_ratings = df_ratings.loc[df_ratings['movieId'] <= 10000]

df_movies = pd.read_csv("F:/Code/Recommendation System/movies.csv")
df_movies = df_movies.loc[df_movies['movieId'] <= 10000]

df_train, df_test = tt_split(df_ratings, test_size = 0.7)

m_train = cc_matrix(df_train, u_col_print(df_ratings, ['userId', 'movieId']))
m_test = cc_matrix(df_test, u_col_print(df_ratings, ['userId', 'movieId']))

sparsity(df_ratings, ['userId', 'movieId'])

del df_ratings, df_test, df_train
gc.collect()

m_preds = predict(m_train, m_test, latent = 1, lr = 1e-2, epoch = 1000, lmbda = 1e-1, timeit = True, verbose = True, plot = True)

l_recommendations = recommend(42, m_train, m_preds, df_movies)
