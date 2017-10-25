#Ensemble xgboost, Boosted Logistic Regression, RAndom Forest with mice
#Data: https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/#activity_id

###Libraries & Randomization
library(functional)
library(e1071)
library(dummies)
library(caret)
library(randomForest)
library(plyr)
library(dplyr)
library(parallel)
library(doParallel)
library(caret)
library(fifer)
library(xgboost)
library(mice)
library(AUC)

set.seed(117)

###Functions

#Returns the number of NAs by column in a data.frame
dfnull <- function(df){
  df_null <- data.frame(apply(apply(df, 2, FUN = 'is.na'), 2, FUN = 'sum'))
  colnames(df_null)[1] <- "# of Null"

  return(df_null)
}

#Fill in NAs in Item_weight column
replcna <- function(df_na_col, df_ref_col){
  n = NROW(df_na_col)

  for (i in 1:n){
    if (is.na.data.frame(df_na_col[i] == T)){
      for (j in 1:n){
        if (df_ref_col[i] == df_ref_col[j] && is.na.data.frame(df_na_col[j]) == F){
          df_na_col[i] <- df_na_col[j]
          break
        }
      }
    } 
  }

  return(df_na_col)
}

#Replaces categorical values with integers for Item_Identifier column
replc <- function(df_col){
  i = 1
  vec = v_uPID <- sort(unique(df_col))
                       
  for (ID in v_uPID){
    df_col <- mapvalues(df_col, from = c(ID), to = c(as.character(i)))
    i <- i + 1
  }
  
  return(df_col)
}


###Import
df_train_x <- read.csv("F:/Code/Vidhya/Loan Prediction/train.csv", stringsAsFactors = F)
df_test_x <- read.csv("F:/Code/Vidhya/Loan Prediction/test.csv", stringsAsFactors = F)

df_train_y <- data.frame(Loan_Status = df_train_x[, ncol(df_train_x)])
df_train_id <- data.frame(Loan_Status = df_train_x[, "Loan_ID"])
df_test_id <- data.frame(Loan_Status = df_test_x[, "Loan_ID"])

df_train_x[, "Loan_ID"] <- NULL
df_test_x[, "Loan_ID"] <- NULL
df_train_x[, ncol(df_train_x)] <- NULL

df_all_raw <- rbind(df_train_x, df_test_x)
df_all_raw[df_all_raw == ""] <- NA

str(df_all_raw)
df_null <- dfnull(df_all_raw) 

###Pre-processing
#Gender
df_all_raw$Gender <- revalue(df_all_raw$Gender, c("Male"="0", "Female"="1"))
df_all_raw$Gender <- as.integer(df_all_raw$Gender)

#Gender
df_all_raw$Married <- revalue(df_all_raw$Married, c("No"="0", "Yes"="1"))
df_all_raw$Married <- as.integer(df_all_raw$Married)

#Education
df_all_raw$Education <- revalue(df_all_raw$Education, c("Not Graduate"="0", "Graduate"="1"))
df_all_raw$Education <- as.integer(df_all_raw$Education)

#Self Employed
df_all_raw$Self_Employed <- revalue(df_all_raw$Self_Employed, c("No"="0", "Yes"="1"))
df_all_raw$Self_Employed <- as.integer(df_all_raw$Self_Employed)

#Dependents
df_all_raw$Dependents[df_all_raw$Dependents == "3+" & is.na(df_all_raw$Dependents) == F] = "3"
df_all_raw$Dependents <- as.integer(df_all_raw$Dependents)

#Property Area
df_all_raw <- cbind(df_all_raw, dummy(df_all_raw$Property_Area))
df_all_raw$Property_Area <- NULL
df_all_raw$df_all_rawRural <- NULL

#Loan_Status
df_train_y$Loan_Status <- revalue(df_train_y$Loan_Status, c("N"="0", "Y"="1"))
df_train_y$Loan_Status <- as.character(df_train_y$Loan_Status)

#NAs with Mice
imp <- mice(df_all_raw[, 1:(ncol(df_all_raw)-2)], m = 5, method = "pmm")

#Imputation
df_all_x <- cbind(complete(imp, 1), df_all_raw$df_all_rawSemiurban, df_all_raw$df_all_rawUrban)

#Train and test dataframes
df_train_all <- df_all_x[1:nrow(df_train_y), ]
df_train_all <- cbind(df_train_all, df_train_y)
df_test_x <- df_all_x[(nrow(df_train_y)+1):nrow(df_all_x),]

rm(df_train_x)
rm(df_null)
str(df_train_all)

###Feature Validation & Selection
first = T
id <- createDataPartition(df_train_all$Loan_Status, p = 0.8, list = F)
val_train_all <- df_train_all[id, ]
val_test_all <- df_train_all[-id, ]

l_auc <- list()
for (i in c(0, 0, 1, 2, 3, 4, 5)){
  cat("Varimp floor: ",  as.character(i), "\n")
  
  if (first == F){
    val_train_sel <- cbind(val_train_all[, which(varImp(fullModel)[1] > i)], Loan_Status = val_train_all$Loan_Status)
  }else{
    val_train_sel <- val_train_all
  }

  fitControl <- trainControl(method = "repeatedcv",
                             number = 4,
                             repeats = 3,
                             search = 'grid',
                             allowParallel = T,
                             verbose = T
  )
  
  if (ncol(val_train_sel) > 12){
    max = 12
  } else {
    max = ncol(val_train_sel)-1
  }
  
  gridControl <- expand.grid(mtry = seq(1, max, 1))
  
  #Parallelisation
  no_cores <- makeCluster(detectCores() - 1)
  registerDoParallel(no_cores)
  
  tuner_rfo <- train(form = Loan_Status ~ .,
                     data = val_train_sel,
                     method = "rf",
                     trControl = fitControl,
                     tuneGrid = gridControl,
                     importance = T
  )
  
  stopCluster(no_cores)
  registerDoSEQ()
  
  print(tuner_rfo)

  if (first == T){
    first = F
    fullModel <- tuner_rfo$finalModel
  }
  
  df_validation_preds <- data.frame(matrix(ncol = 1, nrow = length(val_test_all$Loan_Status)))
  colnames(df_validation_preds) <- c("Predictions")

  df_validation_preds[,1] <- predict(object = tuner_rfo,
                                     newdata = val_test_all[, colnames(val_train_sel)],
                                     n.trees = model_tuner_rfo$bestTune
  )
  
  l_auc <- c(l_auc, auc(accuracy(df_validation_preds$Predictions, as.factor(val_test_all$Loan_Status))))
}


###Run model & predict
col = 1
n_model = 3
df_test_preds <- data.frame(matrix(nrow = nrow(df_test_x), ncol = imp$m * n_model))

#Xgboost
for (i in 1:imp$m){
  cat("Imputation round: ", i, "\n")
  
  df_all_x <- cbind(complete(imp, i), df_all_raw$df_all_rawSemiurban, df_all_raw$df_all_rawUrban)
  df_train_all <- df_all_x[1:nrow(df_train_y), ]
  df_train_all <- cbind(df_train_all, df_train_y)
  df_test_x <- df_all_x[(nrow(df_train_y)+1):nrow(df_all_x),]
  
  df_train_sel <- cbind(df_train_all[, which(varImp(fullModel)[1] > 0)], Loan_Status = df_train_all$Loan_Status)
  df_test_sel <- df_test_x[, which(varImp(fullModel)[1] > 0)]
  
  fitControl <- trainControl(method = "repeatedcv",
                             number = 4,
                             repeats = 3,
                             search = 'grid',
                             allowParallel = T,
                             verbose = T
  )
  
  gridControl <- expand.grid(nrounds = c(100),
                             max_depth = c(1),
                             eta = c(11e-2),
                             gamma = c(1),
                             colsample_bytree = c(0.5),
                             min_child_weight = c(1),
                             subsample = c(0.5)
                             )
  
  no_cores <- makeCluster(detectCores() - 1)
  registerDoParallel(no_cores)
  
  tuner_rfo <- train(form = Loan_Status ~ .,
                     data = df_train_sel,
                     method = "xgbTree",
                     trControl = fitControl,
                     tuneGrid = gridControl,
                     importance = T
  )
  
  stopCluster(no_cores)
  registerDoSEQ()
  
  print(tuner_rfo)
  
  df_test_preds[, col] <- predict(object = tuner_rfo,
                                  newdata = df_test_sel,
                                  n.trees = model_tuner_rfo$bestTune
  )
  
  col = col + 1
}

#RandomForest
for (i in 1:imp$m){
  cat("Imputation round: ", i, "\n")
  
  df_all_x <- cbind(complete(imp, i), df_all_raw$df_all_rawSemiurban, df_all_raw$df_all_rawUrban)
  df_train_all <- df_all_x[1:nrow(df_train_y), ]
  df_train_all <- cbind(df_train_all, df_train_y)
  df_test_x <- df_all_x[(nrow(df_train_y)+1):nrow(df_all_x),]
  
  df_train_sel <- cbind(df_train_all[, which(varImp(fullModel)[1] > 0)], Loan_Status = df_train_all$Loan_Status)
  df_test_sel <- df_test_x[, which(varImp(fullModel)[1] > 0)]
  
  fitControl <- trainControl(method = "repeatedcv",
                             number = 4,
                             repeats = 3,
                             search = 'grid',
                             allowParallel = T,
                             verbose = T
  )
  
  if (ncol(val_train_sel) > 12){
    max = 12
  } else {
    max = ncol(val_train_sel)-1
  }
  
  gridControl <- expand.grid(mtry = seq(1, max, 1))
  
  #Parallelisation
  no_cores <- makeCluster(detectCores() - 1)
  registerDoParallel(no_cores)
  
  tuner_rfo <- train(form = Loan_Status ~ .,
                     data = val_train_sel,
                     method = "rf",
                     trControl = fitControl,
                     tuneGrid = gridControl,
                     importance = T
  )
  
  stopCluster(no_cores)
  registerDoSEQ()
  
  print(tuner_rfo)
  
  df_test_preds[, col] <- predict(object = tuner_rfo,
                                 newdata = df_test_sel,
                                 n.trees = model_tuner_rfo$bestTune
  )
  
  col = col + 1
}

#LogitBoost
for (i in 1:imp$m){
  cat("Imputation round: ", i, "\n")
  
  df_all_x <- cbind(complete(imp, i), df_all_raw$df_all_rawSemiurban, df_all_raw$df_all_rawUrban)
  df_train_all <- df_all_x[1:nrow(df_train_y), ]
  df_train_all <- cbind(df_train_all, df_train_y)
  df_test_x <- df_all_x[(nrow(df_train_y)+1):nrow(df_all_x),]
  
  df_train_sel <- cbind(df_train_all[, which(varImp(fullModel)[1] > 0)], Loan_Status = df_train_all$Loan_Status)
  df_test_sel <- df_test_x[, which(varImp(fullModel)[1] > 0)]
  
  fitControl <- trainControl(method = "repeatedcv",
                             number = 4,
                             repeats = 3,
                             search = 'grid',
                             allowParallel = T,
                             verbose = T
  )
  
  gridControl <- expand.grid(nIter = c(100))
  
  #Parallelisation
  no_cores <- makeCluster(detectCores() - 1)
  registerDoParallel(no_cores)
  
  tuner_rfo <- train(form = Loan_Status ~ .,
                     data = val_train_sel,
                     method = "LogitBoost",
                     trControl = fitControl,
                     tuneGrid = gridControl,
                     importance = T
  )
  
  stopCluster(no_cores)
  registerDoSEQ()
  
  print(tuner_rfo)
  
  df_test_preds[, col] <- predict(object = tuner_rfo,
                                  newdata = df_test_sel,
                                  n.trees = model_tuner_rfo$bestTune
  )
  
  col = col + 1
}

#Make predictions
df_test_preds[is.na(df_test_preds) == T] <- 0
df_test_preds[] <- lapply(df_test_preds[], as.character)
df_test_preds[] <- lapply(df_test_preds[], as.integer)
end_test_preds <- cbind(df_test_id, round(rowMeans(df_test_preds)))
end_test_preds[, 2] <- as.character(end_test_preds[, 2])
names(end_test_preds) <- c("Loan_ID", "Loan_Status")
end_test_preds$Loan_Status <- revalue(end_test_preds$Loan_Status, c("1"="Y", "0"="N"))

###Write
str(end_test_preds)
write.csv(end_test_preds, "F:/Code/Vidhya/Loan Prediction/Loan_submission.csv", row.names = F)
