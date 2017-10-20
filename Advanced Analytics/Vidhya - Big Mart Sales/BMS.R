#Random Forest Stratified Sampling 0.8 20x
#Data: https://datahack.analyticsvidhya.com/contest/practice-problem-big-mart-sales-iii/

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
df_train_x <- read.csv("F:/Code/Vidhya/Big Mart Sales/train.csv", stringsAsFactors = F)
df_test_x <- read.csv("F:/Code/Vidhya/Big Mart Sales/test.csv", stringsAsFactors = F)

df_train_y <- data.frame(Purchase = df_train_x[, ncol(df_train_x)])
df_train_x[, ncol(df_train_x)] <- NULL

df_all_x <- rbind(df_train_x, df_test_x)

str(df_all_x)
df_null <- dfnull(df_all_x)

###Submission DataFrame
end_test_preds <- data.frame(matrix(ncol = 3, nrow = length(df_test_x$Item_Identifier)))
end_test_preds[, 1] <- df_test_x$Item_Identifier
end_test_preds[, 2] <- df_test_x$Outlet_Identifier
colnames(end_test_preds) <- c("Item_Identifier", "Outlet_Identifier", "Item_Outlet_Sales")

###Pre-processing
#NAs
df_all_x$Item_Weight <- replcna(df_all_x$Item_Weight, df_all_x$Item_Identifier)

#Item Indentifier
df_all_x$Item_Identifier <- replc(df_all_x$Item_Identifier)
df_all_x$Item_Identifier <- as.integer(df_all_x$Item_Identifier)

#Item Fat Content
df_all_x[df_all_x$Item_Fat_Content %in% c("LF", "low fat", "Low Fat"), "Item_Fat_Content"] <- "0"
df_all_x[df_all_x$Item_Fat_Content %in% c("reg", "Regular"), "Item_Fat_Content"] <- "1"
df_all_x$Item_Fat_Content <- as.integer(df_all_x$Item_Fat_Content)

#Item Type
df_all_x <- cbind(df_all_x, dummy(df_all_x$Item_Type))
df_all_x$Item_Type <- NULL

#Outlet Identifier
df_all_x$Outlet_Identifier <- substr(df_all_x$Outlet_Identifier, 4, 6)
df_all_x$Outlet_Identifier <- as.integer(df_all_x$Outlet_Identifier)

#Outlet Size
df_all_x[df_all_x$Outlet_Size == "", "Outlet_Size"] = '0'
df_all_x[df_all_x$Outlet_Size == "Small", "Outlet_Size"] = '1'
df_all_x[df_all_x$Outlet_Size == "Medium", "Outlet_Size"] = '2'
df_all_x[df_all_x$Outlet_Size == "High", "Outlet_Size"] = '3'
df_all_x$Outlet_Size <- as.integer(df_all_x$Outlet_Size)

#Outlet Location Type
df_all_x[df_all_x$Outlet_Location_Type == "Tier 1", "Outlet_Location_Type"] = '0'
df_all_x[df_all_x$Outlet_Location_Type == "Tier 2", "Outlet_Location_Type"] = '1'
df_all_x[df_all_x$Outlet_Location_Type == "Tier 3", "Outlet_Location_Type"] = '2'
df_all_x$Outlet_Location_Type <- as.integer(df_all_x$Outlet_Location_Type)

#Outlet Type
df_all_x <- cbind(df_all_x, dummy(df_all_x$Outlet_Type))
df_all_x$Outlet_Type <- NULL

#Train and test dataframes
df_train_all <- df_all_x[1:nrow(df_train_y), ]
df_train_all <- cbind(df_train_all, df_train_y)
df_test_x <- df_all_x[(nrow(df_train_y)+1):nrow(df_all_x),]

rm(df_all_x)
rm(df_train_y)
rm(df_train_x)
rm(df_null)


###Feature Validation & Selection
l_rmse <- list()
first = T
# id <- createDataPartition(df_train_all$Purchase, p = 0.8, list = F)
val_train_all <- df_train_all[id, ]
val_test_all <- df_train_all[-id, ]

for (i in c(0, 0, 5, 10, 12, 15, 18, 20, 25)){
  cat("Varimp floor: ",  as.character(i), "\n")
  
  if (first == F){
    val_train_sel <- cbind(val_train_all[, which(varImp(fullModel) > i)], Purchase = val_train_all$Purchase)
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
  
  if (ncol(val_train_sel) > 10){
    max = 10
  } else {
    max = ncol(val_train_sel)-1
  }
  
  gridControl <- expand.grid(mtry = seq(1, max, 1))
  
  #Parallelisation
  no_cores <- makeCluster(detectCores() - 1)
  registerDoParallel(no_cores)
  
  tuner_rfo <- train(form = Purchase ~ .,
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
  
  df_validation_preds <- data.frame(matrix(ncol = 2, nrow = length(val_test_all$Purchase)))
  
  df_validation_preds <- predict(object = tuner_rfo,
                                 newdata = val_test_all[, colnames(val_train_sel)],
                                 n.trees = model_tuner_rfo$bestTune
  )
  
  l_rmse <- c(l_rmse, RMSE(df_validation_preds, val_test_all$Purchase))
}


###Run model & predict
df_test_preds <- data.frame(matrix(nrow = nrow(df_test_x), ncol = 20))

for (i in 1:20){
  cat("Sample: ", as.character(i), "\n")

  df_train_smpl <- stratified(df = df_train_all, 
                              group = c('Item_Identifier'), 
                              size = 0.8, 
                              replace = F, 
                              bothSets = F
  )

  df_train_sel <- cbind(df_train_smpl[, which(varImp(fullModel) > 15)], Purchase = df_train_smpl$Purchase)
  df_test_sel <- cbind(df_test_x[, which(varImp(fullModel) > 15)])
  
  fitControl <- trainControl(method = "repeatedcv",
                             number = 4,
                             repeats = 3,
                             search = 'grid',
                             allowParallel = T,
                             verbose = T
  )
  
  if (ncol(df_train_sel) > 10){
    max = 10
  } else {
    max = ncol(df_train_sel)-1
  }
  
  gridControl <- expand.grid(mtry = seq(1, max, 1))
  
  no_cores <- makeCluster(detectCores() - 1)
  registerDoParallel(no_cores)
  
  tuner_rfo <- train(form = Purchase ~ .,
                     data = df_train_sel,
                     method = "rf",
                     trControl = fitControl,
                     tuneGrid = gridControl,
                     importance = T
  )
  
  stopCluster(no_cores)
  registerDoSEQ()
  
  print(tuner_rfo)
  
  df_test_preds[, i] <- predict(object = tuner_rfo,
                                newdata = df_test_sel,
                                n.trees = model_tuner_rfo$bestTune
  )
  
}

#Make predictions
end_test_preds[, 3] <- round(rowMeans(df_test_preds[, 1:20], dims = 1))

###Write
str(end_test_preds)
write.csv(end_test_preds, "F:/Code/Vidhya/Big Mart Sales/BMS_submission.csv", row.names = F)
