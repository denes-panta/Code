#Data: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
###Libraries & Randomization
set.seed(117)
setwd("F:/Code/R/House")
library(functional)
library(e1071)
library(dummies)
library(gbm)
library(glmnet)
library(xgboost)
library(caret)
library(randomForest)
library(plyr)
library(parallel)
library(doParallel)
library(caret)


###Functions
dfnull <- function(df){
  df_null <- data.frame(apply(apply(df, 2, FUN = 'is.na'), 2, FUN = 'sum'))
  colnames(df_null)[1] <- "# of Null"
  return(df_null)
}


###Import
df_train_x <- read.csv("F:/Code/Code/Samples/Deep Learning - PyTorch/Regression/train.csv", stringsAsFactors = F)
df_test_x <- read.csv("F:/Code/Code/Samples/Deep Learning - PyTorch/Regression/test.csv", stringsAsFactors = F)
df_train_x[, 1] <- NULL
df_test_x[, 1] <- NULL

df_train_y <- data.frame(SalePrice = df_train_x[, ncol(df_train_x)])
df_train_x[, ncol(df_train_x)] <- NULL

df_all_x <- rbind(df_train_x, df_test_x)


###Pre-processing
##Imputation
#NAs to None
df_all_x[is.na(df_all_x[, 'Alley']), 'Alley'] <- 'None'
df_all_x[is.na(df_all_x[, 'PoolQC']), 'PoolQC'] <- 'None'
df_all_x[is.na(df_all_x[, 'Fence']), 'Fence'] <- 'None'
df_all_x[is.na(df_all_x[, 'MiscFeature']), 'MiscFeature'] <- 'None'
df_all_x[is.na(df_all_x[, 'FireplaceQu']), 'FireplaceQu'] <- 'None'
df_all_x[is.na(df_all_x[, 'BsmtFinType1']), 'BsmtFinType1'] <- 'None'
df_all_x[is.na(df_all_x[, 'GarageType']), 'GarageType'] <- 'None'

df_all_x[is.na(df_all_x[, 'MasVnrArea']), 'MasVnrArea'] <- 0
df_all_x[is.na(df_all_x[, 'BsmtFinSF1']), 'BsmtFinSF1'] <- 0
df_all_x[is.na(df_all_x[, 'BsmtFinSF2']), 'BsmtFinSF2'] <- 0
df_all_x[is.na(df_all_x[, 'BsmtUnfSF']), 'BsmtUnfSF'] <- 0
df_all_x[is.na(df_all_x[, 'TotalBsmtSF']), 'TotalBsmtSF'] <- 0

df_all_x[is.na(df_all_x[, 'MasVnrType']) & (is.na(df_all_x[, 'MasVnrArea']) == T), 'MasVnrType'] <- 'None'
df_all_x[is.na(df_all_x[, 'BsmtQual']) & (is.na(df_all_x[, 'BsmtCond']) == T), 'BsmtQual'] <- 'None'
df_all_x[is.na(df_all_x[, 'BsmtCond']) & df_all_x[, 'BsmtQual'] == 'None', 'BsmtCond'] <- 'None'
df_all_x[is.na(df_all_x[, 'BsmtExposure']) & df_all_x[, 'BsmtCond'] == 'None', 'BsmtExposure'] <- 'None'
df_all_x[is.na(df_all_x[, 'BsmtFinType2']) & df_all_x[, 'BsmtCond'] == 'None', 'BsmtFinType2'] <- 'None'
df_all_x[is.na(df_all_x[, 'GarageYrBlt']) & df_all_x[, 'GarageType'] == 'None', 'GarageYrBlt'] <- 0
df_all_x[is.na(df_all_x[, 'GarageFinish']) & df_all_x[, 'GarageType'] == 'None', 'GarageFinish'] <- 'None'
df_all_x[is.na(df_all_x[, 'GarageQual']) & df_all_x[, 'GarageType'] == 'None', 'GarageQual'] <- 'None'
df_all_x[is.na(df_all_x[, 'GarageCond']) & df_all_x[, 'GarageType'] == 'None', 'GarageCond'] <- 'None'

df_all_x[(max(df_all_x[,'YrSold']) < df_all_x[,'GarageYrBlt']) & (is.na(df_all_x[,'GarageYrBlt']) == F), 'GarageYrBlt'] <- 2007

df_all_x[(df_all_x['YrSold'] - df_all_x['YearRemodAdd'] < 0), 'YearRemodAdd'] <- df_all_x[(df_all_x['YrSold'] - df_all_x['YearRemodAdd'] < 0), 'YrSold']

#Mice
library(mice)
v_char <- names(df_all_x[, sapply(df_all_x, class) == 'character'])
df_all_x[v_char] <- lapply(df_all_x[v_char], as.factor)

df_all_x[,'BsmtFullBath'] <- as.factor(df_all_x[,'BsmtFullBath'])
df_all_x[,'BsmtHalfBath'] <- as.factor(df_all_x[,'BsmtHalfBath'])
df_all_x[,'GarageYrBlt'] <- as.factor(df_all_x[,'GarageYrBlt'])
df_all_x[,'GarageCars'] <- as.factor(df_all_x[,'GarageCars'])

imp <- mice(data = df_all_x,
            m = 5,
            method = 'rf'
)

df_all_x <- complete(imp)

df_all_x[,'BsmtFullBath'] <- as.integer(df_all_x[,'BsmtFullBath'])
df_all_x[,'BsmtHalfBath'] <- as.integer(df_all_x[,'BsmtHalfBath'])
df_all_x[,'GarageYrBlt'] <- as.integer(df_all_x[,'GarageYrBlt'])
df_all_x[,'GarageCars'] <- as.integer(df_all_x[,'GarageCars'])

write.csv(df_all_x, "F:/Code/Code/Samples/Deep Learning - PyTorch/Regression/imputed.csv", row.names = F)


##Engineerig
df_all_x <- read.csv("F:/Code/Code/Samples/Deep Learning - PyTorch/Regression/imputed.csv", stringsAsFactors = F)
df_null <- dfnull(df_all_x)

#Categorical to Ordinal
df_all_x[(df_all_x['ExterQual'] == 'Ex'), 'ExterQual'] = 5
df_all_x[(df_all_x['ExterQual'] == 'Gd'), 'ExterQual'] = 4
df_all_x[(df_all_x['ExterQual'] == 'TA'), 'ExterQual'] = 3
df_all_x[(df_all_x['ExterQual'] == 'Fa'), 'ExterQual'] = 2
df_all_x[(df_all_x['ExterQual'] == 'Po'), 'ExterQual'] = 1
df_all_x[, 'ExterQual'] <- as.integer(df_all_x[,'ExterQual'])

df_all_x[(df_all_x['ExterCond'] == 'Ex'), 'ExterCond'] = 5
df_all_x[(df_all_x['ExterCond'] == 'Gd'), 'ExterCond'] = 4
df_all_x[(df_all_x['ExterCond'] == 'TA'), 'ExterCond'] = 3
df_all_x[(df_all_x['ExterCond'] == 'Fa'), 'ExterCond'] = 2
df_all_x[(df_all_x['ExterCond'] == 'Po'), 'ExterCond'] = 1
df_all_x[, 'ExterCond'] <- as.integer(df_all_x[, 'ExterCond'])

df_all_x[(df_all_x['BsmtQual'] == 'Ex'), 'BsmtQual'] = 5
df_all_x[(df_all_x['BsmtQual'] == 'Gd'), 'BsmtQual'] = 4
df_all_x[(df_all_x['BsmtQual'] == 'TA'), 'BsmtQual'] = 3
df_all_x[(df_all_x['BsmtQual'] == 'Fa'), 'BsmtQual'] = 2
df_all_x[(df_all_x['BsmtQual'] == 'Po'), 'BsmtQual'] = 1
df_all_x[(df_all_x['BsmtQual'] == 'None'), 'BsmtQual'] = 0
df_all_x[, 'BsmtQual'] <- as.integer(df_all_x[, 'BsmtQual'])

df_all_x[(df_all_x['BsmtCond'] == 'Ex'), 'BsmtCond'] = 5
df_all_x[(df_all_x['BsmtCond'] == 'Gd'), 'BsmtCond'] = 4
df_all_x[(df_all_x['BsmtCond'] == 'TA'), 'BsmtCond'] = 3
df_all_x[(df_all_x['BsmtCond'] == 'Fa'), 'BsmtCond'] = 2
df_all_x[(df_all_x['BsmtCond'] == 'Po'), 'BsmtCond'] = 1
df_all_x[(df_all_x['BsmtCond'] == 'None'), 'BsmtCond'] = 0
df_all_x[, 'BsmtCond'] <- as.integer(df_all_x[, 'BsmtCond'])

df_all_x[(df_all_x['BsmtExposure'] == 'Gd'), 'BsmtExposure'] = 4
df_all_x[(df_all_x['BsmtExposure'] == 'Av'), 'BsmtExposure'] = 3
df_all_x[(df_all_x['BsmtExposure'] == 'Mn'), 'BsmtExposure'] = 2
df_all_x[(df_all_x['BsmtExposure'] == 'No'), 'BsmtExposure'] = 1
df_all_x[(df_all_x['BsmtExposure'] == 'None'), 'BsmtExposure'] = 0
df_all_x[, 'BsmtExposure'] <- as.integer(df_all_x[, 'BsmtExposure'])

df_all_x[(df_all_x['BsmtFinType1'] == 'GLQ'), 'BsmtFinType1'] = 6
df_all_x[(df_all_x['BsmtFinType1'] == 'ALQ'), 'BsmtFinType1'] = 5
df_all_x[(df_all_x['BsmtFinType1'] == 'BLQ'), 'BsmtFinType1'] = 4
df_all_x[(df_all_x['BsmtFinType1'] == 'Rec'), 'BsmtFinType1'] = 3
df_all_x[(df_all_x['BsmtFinType1'] == 'LwQ'), 'BsmtFinType1'] = 2
df_all_x[(df_all_x['BsmtFinType1'] == 'Unf'), 'BsmtFinType1'] = 1
df_all_x[(df_all_x['BsmtFinType1'] == 'None'), 'BsmtFinType1'] = 0
df_all_x[, 'BsmtFinType1'] <- as.integer(df_all_x[, 'BsmtFinType1'])

df_all_x[(df_all_x['BsmtFinType2'] == 'GLQ'), 'BsmtFinType2'] = 6
df_all_x[(df_all_x['BsmtFinType2'] == 'ALQ'), 'BsmtFinType2'] = 5
df_all_x[(df_all_x['BsmtFinType2'] == 'BLQ'), 'BsmtFinType2'] = 4
df_all_x[(df_all_x['BsmtFinType2'] == 'Rec'), 'BsmtFinType2'] = 3
df_all_x[(df_all_x['BsmtFinType2'] == 'LwQ'), 'BsmtFinType2'] = 2
df_all_x[(df_all_x['BsmtFinType2'] == 'Unf'), 'BsmtFinType2'] = 1
df_all_x[(df_all_x['BsmtFinType2'] == 'None'), 'BsmtFinType2'] = 0
df_all_x[, 'BsmtFinType2'] <- as.integer(df_all_x[, 'BsmtFinType2'])

df_all_x[(df_all_x['KitchenQual'] == 'Ex'), 'KitchenQual'] = 5
df_all_x[(df_all_x['KitchenQual'] == 'Gd'), 'KitchenQual'] = 4
df_all_x[(df_all_x['KitchenQual'] == 'TA'), 'KitchenQual'] = 3
df_all_x[(df_all_x['KitchenQual'] == 'Fa'), 'KitchenQual'] = 2
df_all_x[(df_all_x['KitchenQual'] == 'Po'), 'KitchenQual'] = 1
df_all_x[, 'KitchenQual'] = as.integer(df_all_x[, 'KitchenQual'])

df_all_x[(df_all_x['PoolQC'] == 'Ex'), 'PoolQC'] = 4
df_all_x[(df_all_x['PoolQC'] == 'Gd'), 'PoolQC'] = 3
df_all_x[(df_all_x['PoolQC'] == 'TA'), 'PoolQC'] = 2
df_all_x[(df_all_x['PoolQC'] == 'Fa'), 'PoolQC'] = 1
df_all_x[(df_all_x['PoolQC'] == 'None'), 'PoolQC'] = 0
df_all_x[, 'PoolQC'] = as.integer(df_all_x[, 'PoolQC'])

df_all_x['CentralAir'] <- as.integer(df_all_x['CentralAir'] == 'Y')

df_all_x[(df_all_x['Alley'] == 'Pave'), 'Alley'] = 2
df_all_x[(df_all_x['Alley'] == 'Grvl'), 'Alley'] = 1
df_all_x[(df_all_x['Alley'] == 'None'), 'Alley'] = 0
df_all_x[, 'Alley'] = as.integer(df_all_x[, 'Alley'])

df_all_x[(df_all_x['Street'] == 'Pave'), 'Street'] = 1
df_all_x[(df_all_x['Street'] == 'Grvl'), 'Street'] = 0
df_all_x[, 'Street'] = as.integer(df_all_x[, 'Street'])

df_all_x[(df_all_x['GarageFinish'] == 'Fin'), 'GarageFinish'] = 3
df_all_x[(df_all_x['GarageFinish'] == 'RFn'), 'GarageFinish'] = 2
df_all_x[(df_all_x['GarageFinish'] == 'Unf'), 'GarageFinish'] = 1
df_all_x[(df_all_x['GarageFinish'] == 'None'), 'GarageFinish'] = 0
df_all_x[, 'GarageFinish'] = as.integer(df_all_x[, 'GarageFinish'])

#Low Var or Cor
df_all_x['MoSold'] <- NULL

#Neighborhood
df_neighborhood <- data.frame(Neighborhood = df_all_x[(1:length(df_train_y$SalePrice)), 'Neighborhood'], SalePrice = df_train_y)
df_neighborhood <- aggregate(df_neighborhood[, 'SalePrice'] ~ df_neighborhood[, 'Neighborhood'], FUN = 'mean')
names(df_neighborhood) <- c('Neighborhood', 'SalePrice')
df_neighborhood <- df_neighborhood[order(df_neighborhood$SalePrice), ]
print(df_neighborhood)

df_all_x['Class'] = 5
df_all_x[df_all_x$Neighborhood %in% df_neighborhood[df_neighborhood[, 'SalePrice'] < 300000, 'Neighborhood'], 'Class'] = 4
df_all_x[df_all_x$Neighborhood %in% df_neighborhood[df_neighborhood[, 'SalePrice'] < 200000, 'Neighborhood'], 'Class'] = 3
df_all_x[df_all_x$Neighborhood %in% df_neighborhood[df_neighborhood[, 'SalePrice'] < 180000, 'Neighborhood'], 'Class'] = 2
df_all_x[df_all_x$Neighborhood %in% df_neighborhood[df_neighborhood[, 'SalePrice'] < 130000, 'Neighborhood'], 'Class'] = 1

df_all_x['Neighborhood'] <- NULL
rm(df_neighborhood)

df_all_x[(df_all_x['BldgType'] == 'TwnhsE') | (df_all_x['BldgType'] == 'TwnhsI'), 'BldgType'] <- 'Twnhs'

#Near Zero Variance
nzv <- nearZeroVar(dummy.data.frame(df_all_x), saveMetrics = T)
nzv[nzv$nzv,]

#Correlation
# v_num <- names(df_all_x[, sapply(df_all_x, class) == 'integer'])
# descrCor <- cor(df_all_x[v_num])
# summary(descrCor[upper.tri(descrCor)])
#
# highlyCorDescr <- findCorrelation(descrCor, cutoff = .7, verbose = T)
# df_all_x[v_num[highlyCorDescr]] <- NULL
#
# v_num <- names(df_all_x[, sapply(df_all_x, class) == 'integer'])
# descrCor_after <- cor(df_all_x[v_num])
# summary(descrCor_after[upper.tri(descrCor_after)])

# #LinearCombos
# v_num <- names(df_all_x[, sapply(df_all_x, class) == 'integer'])
# comboInfo <- findLinearCombos(df_all_x[v_num])
# print(comboInfo)
#
# df_all_x[,comboInfo$remove] <- NULL

#Box - Cox
df_train_y$SalePrice <- log1p(df_train_y$SalePrice)
hist(df_train_y$SalePrice)

v_num <- names(df_all_x[, sapply(df_all_x, class) == 'integer'])

#createDataPartition for Validation
v_char <- names(df_all_x[, sapply(df_all_x, class) == 'character'])
df_all_x[v_char] <- lapply(df_all_x[v_char], as.factor)