#############################################################################
#The code is anything, but optimised so it takes roughly 33 minutes to train#
#############################################################################

#Dataset used: MovieLens: 20M dataset - https://grouplens.org/datasets/movielens/
#Method implemented: 
#	https://www.youtube.com/watch?v=1JRrCEgiyHM
#	https://www.youtube.com/watch?v=h9gpufJFF-0
#	https://www.youtube.com/watch?v=6BTLobS7AU8
#	https://www.youtube.com/watch?v=VZKMyTaLI00

import time
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.model_selection import train_test_split as tt_split
import warnings
import random

random.seed(117)

class recommender(object):
    def __init__(self, ratings, ratings_col, movies, movies_col, n_records):
        #Turn the warnings off
        warnings.filterwarnings('ignore')
        
        #Read in the data
        self.df_ratings = pd.read_csv(ratings)
        self.df_ratings = self.df_ratings.loc[self.df_ratings[ratings_col[0]] <= n_records]
        self.df_ratings = self.df_ratings.loc[self.df_ratings[ratings_col[1]] <= n_records]
    
        self.df_movies = pd.read_csv(movies)
        self.df_movies = self.df_movies.loc[self.df_movies[movies_col] <= n_records]
        
        #Create a train-test split
        self.df_train, self.df_test = tt_split(self.df_ratings, test_size = 0.7)
    
    #Create the ratings matrix
    def train(self):
        self.m_train = self.cc_matrix(self.df_train, 
                                      self.u_col_print(self.df_ratings, 
                                                       ['userId', 'movieId']
                                                       )
                                      )
                                      
        self.m_test = self.cc_matrix(self.df_test, 
                                     self.u_col_print(self.df_ratings, 
                                                      ['userId', 'movieId']
                                                      )
                                     )
        
        self.m_train, self.m_test, self.m_cor = self.dimension(self.m_train, 
                                                               self.m_test, 
                                                               transpose = False
                                                               )

        self.m_preds = self.predict(self.m_train, 
                                    self.m_cor, 
                                    k = 100, 
                                    timeit = True, 
                                    verbose = True
                                    )
        
        print('Test RMSE: ' + \
              str(self.error(self.m_preds, self.m_test)[0]) + \
              '| Test MAE: ' + \
              str(self.error(self.m_preds, self.m_test)[1])
              )

    #Make the recommendations
    def engine(self, user = 1):
        self.l_recommendations = self.recommend(user,
                                                self.m_train, 
                                                self.m_preds, 
                                                self.df_movies, 
                                                transposed = False
                                                )
        
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

    #Centering, Scaling and replacing NaNs with 0s, by rows.
    def normalize(self, matrix, center = True, scale = True, zeros = True):
        #matrix = input matrix
        #center = mean_center
        #scale = divide by std
        #zeros = replace NaN with 0s
        
        #replace zeros with NaNs
        matrix[matrix == 0.0] = np.NaN
    
        v_mean = np.nanmean(matrix, axis = 1).reshape((matrix.shape[0], 1))
        v_std = np.nanstd(matrix, axis = 1).reshape((matrix.shape[0], 1))
        
        if (center == True): matrix = (matrix - v_mean)
        if (scale == True): matrix = matrix / v_std 
        if (zeros == True): matrix[np.isnan(matrix) == True] = 0
        
        return matrix
    
    #Creates the correlation matrix for the columns and 
    #Transposes the matrices if specified
    def dimension(self, train, test, transpose = True):
        #train = train matrix
        #test = test matrix
        #transpose = creating user-user or item-item correlation
        
        #Correlation table, and replace 0s with NaN in both train and test
        if transpose == False:
            m_cor = np.round(pd.DataFrame(self.normalize(self.m_train, 
                                                         center = True, 
                                                         scale = False, 
                                                         zeros = False
                                                         )).corr(), 2)
            train[train == 0] = np.NaN
            test[test == 0] = np.NaN
        #Correlation table, and replace 0s with NaN in both train and test
        #transpose the matrices
        elif transpose == True: 
            m_cor = np.round(pd.DataFrame(self.normalize(self.m_train.T, 
                                                         center = True, 
                                                         scale = False, 
                                                         zeros = False
                                                         )).corr(), 2)
            train = train.T
            train[train == 0] = np.NaN
            test = test.T
            test[test == 0] = np.NaN
    
        return train, test, m_cor

    #Predict the missing Ratings
    def predict(self, train, m_cor,  k = 100, timeit = False, verbose = False):
        # matrix = user/item matrix
        # corr = correlation of columns or rows
        # k = if integer, the number of nearest neighbours, 
        #     if float the minimum correlation between the items/users
        # timeit = to time the function or not

        if timeit == True: start_time = time.time()
        if verbose == True: round_time = time.time()
    
        m_preds = np.empty((train.shape[0], train.shape[1]))
        m_preds[:] = np.NaN
    
        #Calculates the means for overall, users, items
        oall_mean = np.nanmean(train)
        row_means = np.nanmean(train, axis = 1).reshape((train.shape[0]), 1)
        col_means = np.nanmean(train, axis = 0).reshape((train.shape[1]), 1)
                
        for r in range(train.shape[0]):
            df_ratings = pd.DataFrame(train[r, :])
            
            for c in range(train.shape[1]): 
                df_correls = pd.DataFrame(m_cor.iloc[:, c])
                
                #selects the k-th nearest neighbours
                if type(k) == int: 
                    l_ind = np.asarray(
                            df_ratings.index[
                                    (~np.isnan(df_ratings.iloc[:, 0])) & \
                                    (~np.isnan(df_correls.iloc[:, 0])) & \
                                    (df_correls.iloc[:, 0] > 0.0) & \
                                    (df_correls.index[:] != c)]
                            ).tolist()
    
                    m_indices = np.asarray([y[0] for y in \
                                            sorted(
                                                    zip(l_ind, 
                                                        df_correls.iloc[l_ind, 0]
                                                        ), 
                                            key = lambda x: x[1], reverse = True)
                                            ])[:k]
                                                    
                    l_indices = m_indices.tolist()
                
                #selects the neighbours that's correlation is above k
                elif type(k) == float: 
                    l_ind = np.asarray(df_ratings.index[
                            (~np.isnan(df_ratings.iloc[:, 0])) & \
                            (~np.isnan(df_correls.iloc[:, 0])) & \
                            (df_correls.iloc[:, 0] > k) & \
                            (df_correls.index[:] != c)]).tolist()

                    l_indices = l_ind
                    
                sim = np.NaN
    
                #only calculate sim if there are nearest neighbours
                if len(l_indices) != 0:
                    #select the appropriate ratings
                    m_ratings = df_ratings.iloc[l_indices, 0]
                    m_ratings = m_ratings.as_matrix()
                    m_ratings = m_ratings.reshape(len(l_indices), 1) 
                    
                    #select the appropriate correlations
                    m_similar = df_correls.iloc[l_indices, 0]
                    m_similar = m_similar.as_matrix()
                    m_similar = m_similar.reshape(len(l_indices), 1) 
                    
                    #calculate the scores for the neighbours
                    m_adjust = (col_means[l_indices] - oall_mean) + \
                               (row_means[r] - oall_mean) + oall_mean 
                    m_adjust = m_adjust.reshape(len(l_indices), 1) 
                    
                    #Calculate the similarity score part 1
                    m_scores = m_similar * (m_ratings - m_adjust)                     
                    
                    #Calculate the similarity score part 2
                    sim = np.sum(m_scores) / np.sum(m_similar) 
                
                #Global row adjustment for the r-th item/user
                m_glob_r = (row_means[r] - oall_mean)
                #Global col adjustment for the r-th item/user
                m_glob_c = (col_means[c] - oall_mean) 
                
                #Calculate the overall score
                m_global = oall_mean 
                if (np.isnan(m_glob_r)) == False: m_global += m_glob_r
                if (np.isnan(m_glob_c)) == False: m_global += m_glob_c
                if (np.isnan(sim)) == False: m_global += sim
                
                #Round the prediction to the nearest .5
                m_preds[r, c] = np.round(m_global*2) / 2
                #Set the too high prediction to the max value
                if m_preds[r, c] > 5: m_preds[r, c] = 5
                #Set the too low prediction to the min value
                if m_preds[r, c] < 1: m_preds[r, c] = 1 

            #Print round time                
            if (verbose == True):
                if (r % np.round(self.m_train.shape[0]/10) == (np.round(self.m_train.shape[0]/10)-1)):
                    print('Completed: ' + \
                          str(np.round(r/self.m_train.shape[0]*100)) + \
                          ' % - Duration: ' + str(np.round((time.time() - \
                                                            round_time) / 60, 3)) \
                          + ' minutes')
                    round_time = time.time()
        
        #Prints run overall time for the entire function
        if timeit == True: 
            elapsed_time = time.time() - start_time
            print('Total running time: ' + \
                  str(np.round(elapsed_time / 60 , 2)) + \
                  ' minutes' + '\n')
        
        #Print the errors
        print('Train RMSE: ' + \
              str(self.error(m_preds, train)[0]) + \
              '| Train MAE: ' + \
              str(self.error(m_preds, train)[1])) 
    
        return m_preds  

    #Makes the movie recommendations
    def recommend(self, user_id, train, preds, names, transposed = True):
        #user_id = user id of the user for the recommendations
        #train = train data
        #preds = predictions
        #names = dataframe of the movie names
        #transposed = UserIDs are in rows or columns

        #If the users are in the columns 
        if transposed == True:  
            df_past = pd.DataFrame(train[:, user_id])
            df_user = pd.DataFrame(preds[:, user_id])
        #If the users are in the rows
        if transposed == False: 
            df_past = pd.DataFrame(train[user_id, :])
            df_user = pd.DataFrame(preds[user_id, :])
        
        #Watched movies
        l_past = df_past.index[~np.isnan(df_past.iloc[:, 0])].tolist()
        #First Tier recommendations
        l_top1 = df_user.loc[(~df_user.index[:].isin(l_past)) & 
                             (df_user.iloc[:, 0] >= 4.5), 0].index[:].tolist() 
        #Second Tier recommendations
        l_top2 = df_user.loc[(~df_user.index[:].isin(l_past)) & 
                             (df_user.iloc[:, 0] >= 3.5) & 
                             (df_user.iloc[:, 0] <= 4.0), 0].index[:].tolist() 
        
        #Print the suggestions
        print('Top suggestions for user: ' + str(user_id))  
        print(names.iloc[names.index[names.iloc[:, 0].isin(
                list(random.sample(l_top1, 5)))], [1, 2]])
        print('')
        print('User "%d" might also like: ' % (user_id))
        print(names.iloc[names.index[names.iloc[:, 0].isin(
                list(random.sample(l_top2, 10)))], [1, 2]])
        
        return list([l_top1, l_top2])


if __name__ == "__main__":
    colfil = recommender(ratings = "F:/Code/Recommendation System/ratings.csv",
                         ratings_col = ['userId', 'movieId'],
                         movies = "F:/Code/Recommendation System/movies.csv",
                         movies_col = 'movieId',
                         n_records = 1000
                         )

    colfil.train()
    colfil.engine(user = 42)
    