#Caret-xgbTree, Stratified Sampling, 90% of data, 5 times
#Data: https://datahack.analyticsvidhya.com/contest/black-friday/

###Libraries & Randomization
library(functional)
library(e1071)
library(dummies)
library(caret)
library(randomForest)
library(plyr)
library(parallel)
library(doParallel)
library(caret)
library(fifer)
library(xgboost)

set.seed(117)

###Functions
##Returns the number of NAs by column in a data.frame
dfnull <- function(df){
  df_null <- data.frame(apply(apply(df, 2, FUN = 'is.na'), 2, FUN = 'sum'))
  colnames(df_null)[1] <- "# of Null"

  return(df_null)
}

#Replaces categorical values with integers for Product_ID column
replc <- function(df_col, vec){
  i = 1

  for (ID in v_uPID){
    df_col <- mapvalues(df_col, from = c(ID), to = c(as.character(i)))
    i <- i + 1
  }

  return(df_col)
}

###Import
df_train_x <- read.csv("F:/Code/Vidhya/ABC/train.csv", stringsAsFactors = F)
df_test_x <- read.csv("F:/Code/Vidhya/ABC/test.csv", stringsAsFactors = F)

df_train_y <- data.frame(Purchase = df_train_x[, ncol(df_train_x)])
df_train_x[, ncol(df_train_x)] <- NULL

df_all_x <- rbind(df_train_x, df_test_x)

str(df_all_x)
df_null <- dfnull(df_all_x)

###Submission DataFrame
end_test_preds <- data.frame(matrix(ncol = 3, nrow = length(df_test_x$User_ID)))
end_test_preds[, 1] <- df_test_x$User_ID
end_test_preds[, 2] <- df_test_x$Product_ID

###Pre-processing
#NAs
df_all_x[is.na(df_all_x[, 'Product_Category_1']), 'Product_Category_1'] <- 0
df_all_x[is.na(df_all_x[, 'Product_Category_2']), 'Product_Category_2'] <- 0
df_all_x[is.na(df_all_x[, 'Product_Category_3']), 'Product_Category_3'] <- 0

#Marital_Status
df_all_x$Marital_Status <- NULL

#Occupation
#df_all_x$Occupation <- NULL

#Current_City_years
df_all_x[(df_all_x['Stay_In_Current_City_Years'] == '4+'), 'Stay_In_Current_City_Years'] = 4
df_all_x$Stay_In_Current_City_Years <- as.integer(df_all_x$Stay_In_Current_City_Years)
df_all_x$Stay_In_Current_City_Years <- NULL

#Gender
df_all_x$Gender <- revalue(df_all_x$Gender, c("F"="0", "M"="1"))
df_all_x$Gender <- as.integer(df_all_x$Gender)

#City
dum_city <- dummy(df_all_x$City_Category)
df_all_x$City_Category <- NULL
df_all_x <- cbind(df_all_x, dum_city)
rm(dum_city)
df_all_x$City_CategoryC <- NULL

#Age
df_all_x$Age <- revalue(df_all_x$Age, c("0-17"="0", "18-25"="1", "26-35"="2", "36-45"="3", "46-50"="4", "51-55"="5", "55+"= "6"))
df_all_x$Age <- as.integer(df_all_x$Age)

#Product_ID
v_uPID <- sort(unique(df_all_x$Product_ID))
df_all_x$Product_ID <- replc(df_all_x$Product_ID, v_uPID)
df_all_x$Product_ID <- as.integer(df_all_x$Product_ID)

#Train and test dataframes
df_train_all <- df_all_x[1:nrow(df_train_y), ]
df_train_all <- cbind(df_train_all, df_train_y)
df_test_x <- df_all_x[(nrow(df_train_y)+1):nrow(df_all_x),]

rm(df_all_x)
rm(df_train_y)
rm(df_train_x)
rm(df_null)

###Feature Validation
# Create Validation - Test sets - (Comment it out once it has ran)
# df_validation_smpl <- stratified(df = df_train_all,
#                                  group = c('Product_ID'),
#                                  size = 0.1,
#                                  replace = F,
#                                  bothSets = F
# )
# 
# val_ID <- as.integer(rownames(df_validation_smpl))
# 
# df_validation_smpl <- df_train_all[val_ID, ]
# df_validation_test <- df_train_all[-val_ID, ]
# l_rmse <- list()

#Feature Selection & Engineering
# fitControl <- trainControl(method = "cv",
#                            number = 5,
#                            search = 'grid',
#                            allowParallel = T,
#                            verbose = T
# )
# 
# gridControl <- expand.grid(mtry = seq(6, 8, 1))
# 
# #Parallelisation
# no_cores <- makeCluster(detectCores() - 1)
# registerDoParallel(no_cores)
# 
# tuner_rfo <- train(form = Purchase ~ ., 
#                    data = df_validation_smpl,
#                    method = "parRF",
#                    trControl = fitControl,
#                    tuneGrid = gridControl
# )
# 
# stopCluster(no_cores)
# registerDoSEQ()
# 
# print(tuner_rfo)
# 
# df_validation_preds <- data.frame(matrix(ncol = 2, nrow = length(df_validation_test$Purchase)))
# 
# df_validation_preds <- predict(object = tuner_rfo,
#                                newdata = df_validation_test,
#                                n.trees = model_tuner_rfo$bestTune
# )
# 
# l_rmse <- c(l_rmse, RMSE(df_validation_preds, df_validation_test$Purchase))

###Run model & predict
df_test_preds <- data.frame(matrix(ncol = 5, nrow = length(df_test_x$User_ID)))

for (i in 1:5){
  print(paste('Iteration: ', as.character(i)))
  df_train_smpl <- stratified(df = df_train_all, 
                              group = c('Product_ID'), 
                              size = 0.9, 
                              replace = F, 
                              bothSets = F
                              )
  
  df_train_val <- df_train_all[-as.integer(rownames(df_train_smpl)), ]
  
  #Y Variable
  hist(df_train_all$Purchase, breaks = 25) #Original Predicted Variable Distribution
  hist(df_train_smpl$Purchase, breaks = 25) #Sampled data Distribution

  #RandomForest
  fitControl <- trainControl(method = "cv",
                             number = 5,
                             search = 'grid',
                             allowParallel = T,
                             verbose = T
                             )
  
  gridControl <- expand.grid(nrounds = c(1251),
                             max_depth = c(8),
                             eta = c(1e-1), 
                             gamma = c(1e-6),
                             colsample_bytree = c(0.7),
                             min_child_weight = c(5),
                             subsample = c(0.8)
                             )
  
  #Parallelisation
  no_cores <- makeCluster(detectCores() - 1)
  registerDoParallel(no_cores)
  
  tuner_rfo <- train(form = Purchase ~ ., 
                     data = df_train_smpl,
                     method = "xgbTree",
                     trControl = fitControl,
                     tuneGrid = gridControl
                     )
  
  stopCluster(no_cores)
  registerDoSEQ()
  
  print(tuner_rfo)
  
  #Make predictions
  df_test_preds[, i] <- predict(object = tuner_rfo,
                                newdata = df_test_x,
                                n.trees = model_tuner_rfo$bestTune
                                )
  
}

###Write
end_test_preds[, 3] <- round(rowMeans(df_test_preds[, 1:5], dims = 1))
colnames(end_test_preds) <- colnames(read.csv("F:/Code/Vidhya/ABC/Sample_submission.csv"))
str(end_test_preds)

write.csv(end_test_preds, "F:/Code/Vidhya/ABC/ABC_submission.csv", row.names = F)
