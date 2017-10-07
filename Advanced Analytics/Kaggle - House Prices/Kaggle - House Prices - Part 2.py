#Libraries
import pandas as pd
import numpy as np
from scipy import stats
import sklearn.linear_model as lm
from sklearn.model_selection import cross_val_score

#Functions
def write_to_file(pred_array, name):
    m_temp = np.zeros((len(pred_array), 2))
    
    for i in range(len(m_temp)):
        m_temp[i, 0] = i + 1461
        m_temp[i, 1] = pred_array[i]
        
    df_sample = pd.DataFrame(m_temp,columns = ['Id', 'SalePrice']).astype(int)
    
    del m_temp, i
    
    df_sample.to_csv("F:/Code/Kaggle/House/" + name, index = False)
    return print("Submission saved to csv file.")


#Data import
np.random.seed(117)
df_all = pd.read_csv("F:/Code/Kaggle/House/imputed.csv")
end_train_y = pd.read_csv("F:/Code/Kaggle/House/train.csv").loc[:,'SalePrice']

df_all.loc[(df_all['YrSold'] - df_all['YearRemodAdd'] < 0), 'YearRemodAdd'] = df_all['YrSold']
df_all.loc[np.max(df_all.loc[:,'YrSold']) < df_all.loc[:,'GarageYrBlt'], 'GarageYrBlt'] = 2007

df_all['Street'] = (df_all['Street'] == 'Pave').astype(int)

df_neighborhood = pd.concat([df_all.ix[0:len(end_train_y), 'Neighborhood'], end_train_y], axis = 1)
df_neighborhood = df_neighborhood.groupby('Neighborhood')['SalePrice'].mean().reset_index()
df_neighborhood = df_neighborhood.iloc[np.argsort(df_neighborhood.iloc[:, 1])]
df_neighborhood.describe()
df_all['Class'] = 'UltraRich'
df_all.loc[df_all['Neighborhood'].isin(df_neighborhood.ix[df_neighborhood.iloc[:, 1] < 300000, 0]) , 'Class'] = 'Rich'
df_all.loc[df_all['Neighborhood'].isin(df_neighborhood.ix[df_neighborhood.iloc[:, 1] < 200000, 0]) , 'Class'] = 'Wealthy'
df_all.loc[df_all['Neighborhood'].isin(df_neighborhood.ix[df_neighborhood.iloc[:, 1] < 180000, 0]) , 'Class'] = 'Middle'
df_all.loc[df_all['Neighborhood'].isin(df_neighborhood.ix[df_neighborhood.iloc[:, 1] < 130000, 0]) , 'Class'] = 'Lower'
del df_neighborhood

df_all = df_all.drop('Neighborhood', 1)

df_all = df_all.drop('MoSold', 1)

df_all['CentralAir'] = (df_all['CentralAir'] == 'Y').astype(bool).astype(int)

df_all.loc[(df_all['BldgType'] == 'TwnhsE') | (df_all['BldgType'] == 'TwnhsI'), 'BldgType'] = 'Twnhs'

df_all.loc[(df_all['Alley'] == 'Pave'), 'ExterQual'] = 2
df_all.loc[(df_all['Alley'] == 'Grvl'), 'ExterQual'] = 1
df_all.loc[(df_all['Alley'] == 'None'), 'ExterQual'] = 0

df_all.loc[(df_all['ExterQual'] == 'Ex'), 'ExterQual'] = 5
df_all.loc[(df_all['ExterQual'] == 'Gd'), 'ExterQual'] = 4
df_all.loc[(df_all['ExterQual'] == 'TA'), 'ExterQual'] = 3
df_all.loc[(df_all['ExterQual'] == 'Fa'), 'ExterQual'] = 2
df_all.loc[(df_all['ExterQual'] == 'Po'), 'ExterQual'] = 1

df_all.loc[(df_all['ExterCond'] == 'Ex'), 'ExterCond'] = 5
df_all.loc[(df_all['ExterCond'] == 'Gd'), 'ExterCond'] = 4
df_all.loc[(df_all['ExterCond'] == 'TA'), 'ExterCond'] = 3
df_all.loc[(df_all['ExterCond'] == 'Fa'), 'ExterCond'] = 2
df_all.loc[(df_all['ExterCond'] == 'Po'), 'ExterCond'] = 1

df_all.loc[(df_all['BsmtQual'] == 'Ex'), 'BsmtQual'] = 5
df_all.loc[(df_all['BsmtQual'] == 'Gd'), 'BsmtQual'] = 4
df_all.loc[(df_all['BsmtQual'] == 'TA'), 'BsmtQual'] = 3
df_all.loc[(df_all['BsmtQual'] == 'Fa'), 'BsmtQual'] = 2
df_all.loc[(df_all['BsmtQual'] == 'Po'), 'BsmtQual'] = 1
df_all.loc[(df_all['BsmtQual'] == 'None'), 'BsmtQual'] = 0
df_all['BsmtQual'] = df_all['BsmtQual'].astype(int)

df_all.loc[(df_all['BsmtCond'] == 'Ex'), 'BsmtCond'] = 5
df_all.loc[(df_all['BsmtCond'] == 'Gd'), 'BsmtCond'] = 4
df_all.loc[(df_all['BsmtCond'] == 'TA'), 'BsmtCond'] = 3
df_all.loc[(df_all['BsmtCond'] == 'Fa'), 'BsmtCond'] = 2
df_all.loc[(df_all['BsmtCond'] == 'Po'), 'BsmtCond'] = 1
df_all.loc[(df_all['BsmtCond'] == 'None'), 'BsmtCond'] = 0
df_all['BsmtCond'] = df_all['BsmtCond'].astype(int)

df_all.loc[(df_all['BsmtExposure'] == 'Gd'), 'BsmtExposure'] = 4
df_all.loc[(df_all['BsmtExposure'] == 'Av'), 'BsmtExposure'] = 3
df_all.loc[(df_all['BsmtExposure'] == 'Mn'), 'BsmtExposure'] = 2
df_all.loc[(df_all['BsmtExposure'] == 'No'), 'BsmtExposure'] = 1
df_all.loc[(df_all['BsmtExposure'] == 'None'), 'BsmtExposure'] = 0
df_all['BsmtExposure'] = df_all['BsmtExposure'].astype(int)

df_all.loc[(df_all['BsmtFinType1'] == 'GLQ'), 'BsmtFinType1'] = 6
df_all.loc[(df_all['BsmtFinType1'] == 'ALQ'), 'BsmtFinType1'] = 5
df_all.loc[(df_all['BsmtFinType1'] == 'BLQ'), 'BsmtFinType1'] = 4
df_all.loc[(df_all['BsmtFinType1'] == 'Rec'), 'BsmtFinType1'] = 3
df_all.loc[(df_all['BsmtFinType1'] == 'LwQ'), 'BsmtFinType1'] = 2
df_all.loc[(df_all['BsmtFinType1'] == 'Unf'), 'BsmtFinType1'] = 1
df_all.loc[(df_all['BsmtFinType1'] == 'None'), 'BsmtFinType1'] = 0
df_all['BsmtFinType1'] = df_all['BsmtFinType1'].astype(int)

df_all.loc[(df_all['BsmtFinType2'] == 'GLQ'), 'BsmtFinType2'] = 6
df_all.loc[(df_all['BsmtFinType2'] == 'ALQ'), 'BsmtFinType2'] = 5
df_all.loc[(df_all['BsmtFinType2'] == 'BLQ'), 'BsmtFinType2'] = 4
df_all.loc[(df_all['BsmtFinType2'] == 'Rec'), 'BsmtFinType2'] = 3
df_all.loc[(df_all['BsmtFinType2'] == 'LwQ'), 'BsmtFinType2'] = 2
df_all.loc[(df_all['BsmtFinType2'] == 'Unf'), 'BsmtFinType2'] = 1
df_all.loc[(df_all['BsmtFinType2'] == 'None'), 'BsmtFinType2'] = 0
df_all['BsmtFinType2'] = df_all['BsmtFinType2'].astype(int)

df_all.loc[(df_all['KitchenQual'] == 'Ex'), 'KitchenQual'] = 5
df_all.loc[(df_all['KitchenQual'] == 'Gd'), 'KitchenQual'] = 4
df_all.loc[(df_all['KitchenQual'] == 'TA'), 'KitchenQual'] = 3
df_all.loc[(df_all['KitchenQual'] == 'Fa'), 'KitchenQual'] = 2
df_all.loc[(df_all['KitchenQual'] == 'Po'), 'KitchenQual'] = 1
df_all['KitchenQual'] = df_all['KitchenQual'].astype(int)


#Skewedness
end_train_y = np.log1p(end_train_y)

integers = df_all.dtypes[df_all.dtypes != "object"].index

s_skewed = df_all.ix[:, integers].apply(lambda x: stats.skew(x.dropna()))
s_skewed = s_skewed[(s_skewed > 1)]
s_skewed = s_skewed.index

df_all[s_skewed] = np.log1p(df_all[s_skewed])
df_all = pd.get_dummies(df_all)

end_train_x = df_all.iloc[0:len(end_train_y), :]
end_test_x = df_all.iloc[len(end_train_y):, :]


#Models
#Lasso
alphas = [0.0001, 0.0005, 0.001, 0.01, 0.1, 0.5, 1, 1.5, 2, 2.25, 2.5, 2.75, 3]

rmse_lasso = []
for alpha in alphas:
    rmse_lasso.append(cross_val_score(lm.Lasso(alpha = alpha),
                                      X = end_train_x,
                                      y = end_train_y,
                                      cv = 5,
                                      scoring = "neg_mean_squared_error"
                                      ).mean() * -1
    )

rmse_lasso = pd.Series(rmse_lasso, index = alphas)
rmse_lasso.plot(title = "Validation")

model_lasso = lm.Lasso(alpha = rmse_lasso[rmse_lasso == rmse_lasso.min()].index[0])
model_lasso.fit(end_train_x, end_train_y)

end_train_x = end_train_x.drop(end_train_x.iloc[:, model_lasso.coef_ == 0].columns, 1)
end_test_x = end_test_x.drop(end_test_x.iloc[:, model_lasso.coef_ == 0].columns, 1)

model_lasso.fit(end_train_x, end_train_y)
end_test_pl = np.exp(model_lasso.predict(end_test_x))


#Ridge
rmse_ridge = []
for alpha in alphas:
    rmse_ridge.append(cross_val_score(lm.Ridge(alpha = alpha),
                                       X = end_train_x,
                                       y = end_train_y,
                                       cv = 5,
                                       scoring = "neg_mean_squared_error"
                                       ).mean() * -1
    )

rmse_ridge = pd.Series(rmse_ridge, index = alphas)
rmse_ridge.plot(title = "Validation")
rmse_ridge.min()

model_ridge = lm.Ridge(alpha = rmse_ridge[rmse_ridge == rmse_ridge.min()].index[0])
model_ridge.fit(end_train_x, end_train_y)
end_test_pr = np.exp(model_ridge.predict(end_test_x))


#Evaluate
rmse_history = np.zeros((0,2))
rmse_history = np.append(rmse_history, np.array(([[rmse_lasso.min(), rmse_ridge.min()]])), axis = 0)
print(rmse_lasso.min())
print(rmse_ridge.min())


#Write
end_test_r = pd.read_csv("F:/Code/Kaggle/House/submission_r.csv")
end_test_p = (end_test_pr + end_test_pl + end_test_r.iloc[:,1]) / 3
write_to_file((end_test_pr + end_test_pl) / 2, "submission_p.csv")
write_to_file(end_test_p, "submission.csv")
