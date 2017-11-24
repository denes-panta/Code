#Dataset used: MovieLens: 20M dataset - https://grouplens.org/datasets/movielens/
#Sources used in the implementation:
#    https://www.youtube.com/watch?v=E8aMcwmqsTg
#    https://www.youtube.com/watch?v=E8aMcwmqsTg
#    https://www.youtube.com/watch?v=GGWBMg0i9d4
#    https://www.youtube.com/watch?v=HY3Csl52PfE
#    https://www.slideshare.net/sscdotopen/latent-factor-models-for-collaborative-filtering

import time
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.model_selection import train_test_split as tt_split
import matplotlib.pyplot as plt
from scipy import sparse
import warnings
import random

class recommender(object):

    def __init__(self, ratings, ratings_col, movies, movies_col, n_records):
        #Turn the warnings off
        warnings.filterwarnings('ignore')
        
        #Set seed
        random.seed(117)
        
        #Read in the data
        self.df_ratings = pd.read_csv(ratings)
        self.df_ratings = self.df_ratings.loc[
                self.df_ratings[ratings_col[0]] <= n_records]
        self.df_ratings = self.df_ratings.loc[
                self.df_ratings[ratings_col[1]] <= n_records]
    
        self.df_movies = pd.read_csv(movies)
        self.df_movies = self.df_movies.loc[
                self.df_movies[movies_col] <= n_records]
        
        #Create a train-test split
        self.df_train, self.df_test = tt_split(self.df_ratings, test_size = 0.7)
    
    #Create the ratings sparse matrix
    def train(self):
        self.m_train = self.cc_matrix(self.df_train, 
                                      self.u_col_print(self.df_ratings, 
                                                       ['userId', 'movieId']))
        self.m_test = self.cc_matrix(self.df_test, 
                                     self.u_col_print(self.df_ratings, 
                                                      ['userId', 'movieId']))

        self.sparsity(self.df_ratings, ['userId', 'movieId'])
        
        self.m_preds = self.predict(self.m_train, 
                                    self.m_test, 
                                    latent = 2, 
                                    lr = 1e-2, 
                                    epoch = 1000, 
                                    lmbda = 1e-1, 
                                    timeit = True, 
                                    verbose = True, 
                                    plot = True)

    #Make prediction
    def engine(self, user_id = 1):
        self.recommend(user_id, self.df_movies)
    
    #Error of the prediction
    def error(self, m_preds, m_origin): 
        #m_preds = predictions
        #m_origin = original data
        
        #Number of non-zero, non NaN values in the original data
        n = np.count_nonzero(~np.isnan(m_origin)) 
        #Root Mean Squared Error
        RMSE = np.round(sqrt(np.nansum((m_preds - m_origin)**2) / n), 4) 
        #Mean Absolute Error
        MAE = np.round(np.nansum(np.abs(m_preds - m_origin)) / n, 4)
    
        return list([RMSE, MAE])

    #Check the sparsity of the dataframe
    def sparsity(self, df, cols = []):
        v_n_col = len(df[cols[0]].unique()) 
        v_n_row = len(df[cols[1]].unique())
        v_sparsity = np.round((len(df) / (v_n_col*v_n_row)) * 100, 2)
        print('Sparsity: %.2f %%' % (v_sparsity))
    
    #Returns the unique/max number of elements in specified columns in DF
    #Whichever is larger
    def u_col_print(self, df, cols = [], verbose = False): 
        #df = dataframe of values
        #cols = columns that need to be checked
        #verbose = prints the max and the unique number of items  

        #Size of the columns
        col_size = [0] * len(cols)
        
        for ind, lst in enumerate(cols):
            n = df[lst].unique().shape[0]
            m = max(df[lst])
            
            #Print out the information on request
            if verbose == True:
                print('Number of unique items in %s = %d' % (lst, n))
                print('Largest item %s = %d' % (lst, m) + '\n')
            
            if (max(df[lst]) == df[lst].unique().shape[0]):
                col_size[ind] = df[lst].unique().shape[0] 
            else:
                col_size[ind] = max(df[lst])
    
        return col_size

    #Creates the user-item matrix       
    def cc_matrix(self, df, col_size = []): 
        #df = dataframe
        #col_size = size of the columns
        
        #Create the matrix
        matrix = np.zeros((col_size[0], col_size[1]))
        
        #Populate the matrix   
        for r in df.itertuples(index = False): 
            matrix[r[0]-1, r[1]-1] = r[2]
                
        return matrix
    
    #Calculate the row indexes of the CSR Matrix
    def csr_rows(self, csr_matrix): 
        l_row = []
        
        for i, v in enumerate(csr_matrix.indptr):
            #Use try-except because the loop return an error for the last item
            try: 
                for j in range(csr_matrix.indptr[i+1] - csr_matrix.indptr[i]):
                    l_row.append(i)
            except:
                pass
        
        return l_row

    #Predict the missing Ratings
    def predict(self, train, test, latent, lmbda, 
                lr, epoch, timeit, verbose, plot):
        #matrix = user/item matrix
        #corr = correlation of columns or rows
        # k = if integer, the number of nearest neighbours, 
        #     if float the minimum correlation between the items/users
        # timeit = to time the function or not
        # verbose = print iterations and iteration duration
            
        if timeit == True: start_time = time.time()
        if verbose == True: round_time = time.time()
        
        #Create random latent factors for first dimension
        U = 3 * np.random.rand(train.shape[0], latent) 
        #Create random latent factors for second dimension
        V = 3 * np.random.rand(train.shape[1], latent) 
        
        #Create error table
        m_error = np.zeros((epoch, 3)) 
    
        #Transform train matrix to sparse
        s_train = sparse.csr_matrix(train) 
        train[train == 0] = np.NaN
        
        #Global mean
        v_Gmean = np.nanmean(train)
        #Column means
        m_Cmean = np.nanmean(train, axis = 0)
        m_Cmean = m_Cmean.reshape((1, train.shape[1])) - v_Gmean 
        m_Cmean[np.isnan(m_Cmean)] = 0
        #Row means
        m_Rmean = np.nanmean(train, axis = 1)
        m_Rmean = m_Rmean.reshape((train.shape[0], 1)) - v_Gmean 
        m_Rmean[np.isnan(m_Rmean)] = 0
        
        train[np.isnan(train)] = 0
        
        #Column indices of sparse matrix
        l_cols = list(s_train.indices) 
        #Row indices of sparse matrix
        l_rows = self.csr_rows(s_train) 
        
        for e in range(epoch):
            for s, v in enumerate(s_train):
                r = l_rows[s]
                c = l_cols[s]
                
                #Calculate error
                err = s_train[r, c] - \
                    (v_Gmean + m_Rmean[r, 0] + \
                     m_Cmean[0, c] + \
                     np.dot(U[r, :], V[c, :].T)) 
                
                #SGD update latent factors
                U[r, :] += lr * (err * V[c, :] - lmbda * U[r, :]) 
                V[c, :] += lr * (err * U[r, :] - lmbda * V[c, :]) 
                #SGD update row mean
                m_Rmean[r, 0] += lr * (err - lmbda * m_Rmean[r, 0])
                #SGD update col mean
                m_Cmean[0, c] += lr * (err - lmbda * m_Cmean[0, c]) 
                
            m_preds = np.dot(U, V.T)
            
            #Update error table
            m_error[e, 0] = e 
            m_error[e, 1] = self.error(m_preds, self.m_train)[0]
            m_error[e, 2] = self.error(m_preds, self.m_test)[0]
            
            # Print Duration after every 10% Done
            if (verbose == True) and (e % (epoch / 10) == (epoch / 10)-1): 
                print('Completed: ' + str((e+1)/epoch*100) + \
                      ' % - Duration: ' + \
                      str(np.round((time.time() - round_time) / 60, 3)) + \
                      ' minutes')
                round_time = time.time()
        
        #Set too high ratings to the Max rating
        m_preds[m_preds > 5] = 5 
        #Set too low ratings to the Min rating
        m_preds[m_preds < 1] = 1 
    
        m_preds = np.round(m_preds*2) / 2
        
        #Print the total run duration of the function
        if timeit == True: 
            elapsed_time = time.time() - start_time
            print('Total running time: ' + \
                  str(np.round(elapsed_time / 60, 2)) + ' minutes' + '\n')

        #Print train Error
        print('Train RMSE: ' + str(self.error(m_preds, train)[0]) + \
              '| Train MAE: ' + str(self.error(m_preds, train)[1]))
        #Print test Error
        print('Test RMSE: ' + str(self.error(m_preds, test)[0]) + \
              '| Test MAE: ' + str(self.error(m_preds, test)[1])) 
        
        #Plot errors
        if plot == True: 
            plt.plot(m_error[:, 0], m_error[:, 1], 'b', label = 'Train')
            plt.plot(m_error[:, 0], m_error[:, 2], 'r', label = 'Test')
            plt.xlabel('Epochs')
            plt.ylabel('Error')
            plt.title('Error plot')
            plt.legend(loc = 'upper right')
            
        return m_preds

    def recommend(self, user_id, names):
        #Wathed movies ratings for the user
        df_past = pd.DataFrame(self.m_train[user_id, :])
        df_past[df_past == 0] = np.NaN 
        #Predicted ratings for the user
        df_user = pd.DataFrame(self.m_preds[user_id, :]) 
        
        #Filtered ratings without NaN
        l_past = df_past.index[~np.isnan(df_past.iloc[:, 0])].tolist()
        #Tier one suggestions
        l_top1 = df_user.loc[(~df_user.index[:].isin(l_past)) & 
                             (df_user.iloc[:, 0] >= 4.5), 0].index[:].tolist() 
        #Tier two suggestions
        l_top2 = df_user.loc[(~df_user.index[:].isin(l_past)) & 
                             (df_user.iloc[:, 0] >= 3.5) & 
                             (df_user.iloc[:, 0] <= 4.0), 0].index[:].tolist() 
        
        #Tier one recommendations
        v_t1r = len(l_top1) 
        if v_t1r > 5: v_t1r = 5
        #Tier two recommendations
        v_t2r = len(l_top2) 
        if v_t2r > 5: v_t2r = 10
        
        #Print suggestions
        print('Top suggestions for user: ' + str(user_id)) 
        print(names.iloc[names.index[names.iloc[:, 0].isin(
                list(random.sample(l_top1, v_t1r)))], [1, 2]])
        print('')
        print('User "%d" might also like: ' % (user_id))
        print(names.iloc[names.index[names.iloc[:, 0].isin(
                list(random.sample(l_top2, v_t2r)))], [1, 2]])
        
        return list([l_top1, l_top2])

if __name__ == "__main__":
    latfact = recommender(ratings = "F:/Code/Recommendation System/ratings.csv",
                          ratings_col = ['userId', 'movieId'],
                          movies = "F:/Code/Recommendation System/movies.csv",
                          movies_col = 'movieId',
                          n_records = 2000
                          )
    latfact.train()
    latfact.engine(user_id = 42)