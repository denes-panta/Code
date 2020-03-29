# Libraries
library(plyr)
library(dplyr)
library(caret)
library(DMwR)

# Change workplace to the Grover directory
setwd('D:\\Code\\Code\\Samples\\Transactions - Fraud Detection')

# set seed
set.seed(1)

# Functions

# Model validation
validate <- function(model, dfTest){
  
  vPredicted <- predict(model, dfTest)
  
  dfError <- data.frame(
    Preds = vPredicted ,
    Valid = vLabels
    )
  
  result <- confusionMatrix(
    data = dfError$Preds,
    reference = dfError$Valid
    )
  
  return(result)

}

# Ensemble validation
ensemble <- function(dfEnsemble) {
  dfEnsemble[] <- lapply(dfEnsemble, as.character)
  dfEnsemble[dfEnsemble == 'good'] <- 0
  dfEnsemble[dfEnsemble == 'bad'] <- 1
  dfEnsemble[] <- lapply(dfEnsemble, as.integer)

  dfEnsemble$kknn <- dfEnsemble$kknn * 2
  dfEnsemble$svmlr <- dfEnsemble$svmlr * 1
  dfEnsemble$proto <- dfEnsemble$proto * 1
  
  dfEnsemble <- transform(
    dfEnsemble, 
    final = round(rowSums(dfEnsemble) / (ncol(dfEnsemble) + 3))
    )
  
  dfEnsemble[dfEnsemble$final == 0, 'final'] <- 'good'
  dfEnsemble[dfEnsemble$final == 1, 'final'] <- 'bad'
  
  dfEnsemble$final <- as.factor(dfEnsemble$final)
  
  vPredicted <- dfEnsemble$final
  
  dfError <- data.frame(
    Preds = vPredicted,
    Valid = vLabels
    )
  
  result <- confusionMatrix(
    data = dfError$Preds,
    reference = dfError$Valid
    )
  
  return(result)
}


# Read in the files
dfTrain <- read.csv('resources\\training.csv', sep = ';', stringsAsFactors = F)
dfTest <- read.csv('resources\\validation.csv', sep = ';', stringsAsFactors = F)

# Exploration and missing value handling

str(dfTrain)
str(dfTest)
# Every variable seems to be character, which mean that there are missing values
# or some non-numerical labels mixed in with the numbers, 
# and their meaning is unknown
# Let's assume that ? means missing data and everything else has information value

# Replace ? with NA
dfTrain[dfTrain == '?'] <- NA
dfTest[dfTest == '?'] <- NA

# Check for missing data
print(colSums(is.na(dfTrain)))
print(colSums(is.na(dfTest)))

# Variable X seems to be an ID variable, and contains no information value
# To simplify things, this will be used to distinguis between train and test
# in a merged file

dfTrain$X <- 'train'
dfTest$X <- 'test'

dfData <- rbind(dfTrain, dfTest)

# Since we are talking about financial transactions, check the distribution of
# the predicted variable for class imbalance
table(dfData$y)

dfData <- rename(
  dfData,
  Class = y
  )

dfData$Class <- as.factor(dfData$Class)

# Since there is no information about the variables, and the reason behind the
# missing values, the missing values will be marked with a neutral category in
# the categorical variable
# There are too many missing values to delete these,
# and imputation would introduce noise if the missing values don't belong to
# either category

# X.0 is categorical with 2 categories and some missing values
table(dfData$x.0)
sum(is.na(dfData$x.0))

dfData[is.na(dfData$x.0), 'x.0'] <- '-1'

dfData$x.0 <- as.factor(dfData$x.0)

# X.1 is numerical with some missing values and decimal commas instead of dots
table(dfData$x.1)
sum(is.na(dfData$x.1))

dfData$x.1 <- gsub(",", ".", dfData$x.1)
dfData[is.na(dfData$x.1), 'x.1'] <- -1

dfData$x.1 <- as.numeric(dfData$x.1)

# X.2 is numerical with categories and no missing values
# Create a new variable to contain the labels
# Set the labels to -1 in the x.2 and everything else but the labels in x.2_cat
table(dfData$x.2)
sum(is.na(dfData$x.2))

dfData$x.2_cat <- -1
dfData$x.2_cat[dfData$x.2 == 'f'] <- 'f'
dfData$x.2_cat[dfData$x.2 == 't'] <- 't'
dfData$x.2_cat <- as.factor(dfData$x.2_cat)

dfData$x.2 <- mapvalues(dfData$x.2, c('t', 'f'), c('-1', '-1'))
dfData$x.2 <- as.numeric(dfData$x.2)

# X.3 - x.6 are categorical with some missing values

# X.3
table(dfData$x.3)
sum(is.na(dfData$x.3))

dfData[is.na(dfData$x.3), 'x.3'] <- '-1'

dfData$x.3 <- as.factor(dfData$x.3)

# X.4
table(dfData$x.4)
sum(is.na(dfData$x.4))


dfData[is.na(dfData$x.4), 'x.4'] <- '-1'
dfData$x.4 <- as.factor(dfData$x.4)

# X.5
table(dfData$x.5)
sum(is.na(dfData$x.5))

dfData[is.na(dfData$x.5), 'x.5'] <- '-1'
dfData$x.5 <- as.factor(dfData$x.5)

# X.6
table(dfData$x.6)
sum(is.na(dfData$x.6))

dfData[is.na(dfData$x.6), 'x.6'] <- '-1'

dfData$x.6 <- as.factor(dfData$x.6)

# x.7 is numerical with categories and no missing values
table(dfData$x.7)
sum(is.na(dfData$x.7))

dfData$x.7_cat <- -1
dfData$x.7_cat[dfData$x.7 == 'f'] <- 'f'
dfData$x.7_cat[dfData$x.7 == 't'] <- 't'
dfData$x.7_cat <- as.factor(dfData$x.7_cat)

dfData$x.7 <- mapvalues(dfData$x.7, c('t', 'f'), c('-1', '-1'))
dfData$x.7 <- as.numeric(dfData$x.7)

# x.8 and x.9 are categorical with no misssing values
table(dfData$x.8)
sum(is.na(dfData$x.8))

dfData$x.8 <- as.factor(dfData$x.8)

table(dfData$x.9)
sum(is.na(dfData$x.9))

dfData$x.9 <- as.factor(dfData$x.9)

# x.10 seems to be categorical, with numbers representing the categories
# and with no missing values
table(dfData$x.10)
sum(is.na(dfData$x.10))

dfData$x.10 <- as.factor(dfData$x.10)

# x.11 is categorical with no missing values
table(dfData$x.11)
sum(is.na(dfData$x.11))

dfData$x.11 <- as.factor(dfData$x.11)

# x.12 is categorical with no missing values
table(dfData$x.12)
sum(is.na(dfData$x.12))

dfData$x.12 <- as.factor(dfData$x.12)

# x.13 is numerical with labels and some missing values
table(dfData$x.13)
sum(is.na(dfData$x.13))

dfData$x.13_cat <- -1
dfData$x.13_cat[dfData$x.13 == 'f'] <- 'f'
dfData$x.13_cat <- as.factor(dfData$x.13_cat)

dfData[is.na(dfData$x.13), 'x.13'] <- '-1'
dfData$x.13 <- mapvalues(dfData$x.13, c('f'), c('-1'))
dfData$x.13 <- as.numeric(dfData$x.13)

# x.14 is numerical with labels and no missing values
table(dfData$x.14)
sum(is.na(dfData$x.14))

dfData$x.14_cat <- -1
dfData$x.14_cat[dfData$x.14 == 'f'] <- 'f'
dfData$x.14_cat[dfData$x.14 == 't'] <- 't'
dfData$x.14_cat <- as.factor(dfData$x.14_cat)

dfData$x.14 <- mapvalues(dfData$x.14, c('t', 'f'), c('-1', '-1'))
dfData$x.14 <- as.numeric(dfData$x.14)

# x.20 is categorical with some missing values
table(dfData$x.20)
sum(is.na(dfData$x.20))

dfData[is.na(dfData$x.20), 'x.20'] <- '-1'
dfData$x.20 <- as.factor(dfData$x.20)

# x.17 and x.18 are numerical with some missing values and decimal commas

# x.17
table(dfData$x.17)
sum(is.na(dfData$x.17))
dfData[is.na(dfData$x.17), 'x.17'] <- -1
dfData$x.17 <- gsub(",", ".", dfData$x.17)
dfData$x.17 <- as.numeric(dfData$x.17)

# x.18
table(dfData$x.18)
sum(is.na(dfData$x.18))
dfData[is.na(dfData$x.18), 'x.18'] <- -1
dfData$x.18 <- gsub(",", ".", dfData$x.18)
dfData$x.18 <- as.numeric(dfData$x.18)

str(dfData)

# x.19 is numberical with labels and some missing values 
table(dfData$x.19)
sum(is.na(dfData$x.19))

dfData$x.19_cat <- -1
dfData$x.19_cat[dfData$x.19 == 'f'] <- 'f'
dfData$x.19_cat[dfData$x.19 == 't'] <- 't'
dfData$x.19_cat <- as.factor(dfData$x.19_cat)

dfData$x.19 <- mapvalues(dfData$x.19, c('f'), c('-1'))
dfData[is.na(dfData$x.19), 'x.19'] <- '-1'
dfData$x.19 <- as.numeric(dfData$x.19)

# x.16 is categorical with some missing values
table(dfData$x.16)
sum(is.na(dfData$x.16))

dfData[is.na(dfData$x.16), 'x.16'] <- '-1'
dfData$x.16 <- as.factor(dfData$x.16)

# check variables for zero variance
nearZeroVar(dfData, saveMetrics = T)

# Separate the training and testing datasets
dfTrain <- dfData[dfData$X == 'train', ]
dfTrain$X <- NULL

dfTest <- dfData[dfData$X == 'test', ]
dfTest$X <- NULL

# Extract the test class variable into a separate vector
vLabels <- dfTest$Class
dfTest$Class <- NULL

# Sub-Sample the data
# Results of various sub-sampling methods:
#
# Up-sampling:
#   Accuracy : 0.6939         
#   95% CI : (0.651, 0.7344)
#   Sensitivity : 0.8095         
#   Specificity : 0.4416
#
# Down-sampling:
#   Accuracy : 0.7245          
#   95% CI : (0.6826, 0.7636)
#   Sensitivity : 0.8929          
#   Specificity : 0.3571 
#
# SMOTE:
#   Accuracy : 0.7653          
#   95% CI : (0.7252, 0.8022)
#   Sensitivity : 0.9315          
#   Specificity : 0.4026  
#
# ROSE: 
#   Accuracy : 0.5286          
#   95% CI : (0.4833, 0.5735)
#   Sensitivity : 0.6012          
#   Specificity : 0.3701 


# SMOTE Sub-sampling
# K parameter tuning
#
# K == 1
#   Accuracy : 0.5306          
#   95% CI : (0.4853, 0.5755)
#   Sensitivity : 0.6042          
#   Specificity : 0.3701  
#
# K == 2
#   Accuracy : 0.7857          
#   95% CI : (0.7467, 0.8212)
#   Sensitivity : 0.9673          
#   Specificity : 0.3896
#
# K == 3
#   Accuracy : 0.7612          
#   95% CI : (0.7209, 0.7983)
#   Sensitivity : 0.9375          
#   Specificity : 0.3766
#
# K == 4
#   Accuracy : 0.751           
#   95% CI : (0.7103, 0.7887)
#   Sensitivity : 0.9077          
#   Specificity : 0.4091
#
# K == 5
#   Accuracy : 0.6             
#   95% CI : (0.5551, 0.6437)
#   Sensitivity : 0.7083          
#   Specificity : 0.3636    

# Over-, Under-sampling percentage parameter tuning
#
# O == 50, U == 300
#   Accuracy : 0.5286 
#   95% CI : (0.4833, 0.5735)
#   Sensitivity : 0.6012
#   Specificity : 0.3701 
#
# O == 100, U == 200
#   Accuracy : 0.6449          
#   95% CI : (0.6007, 0.6873)
#   Sensitivity : 0.7768          
#   Specificity : 0.3571  
#
# O == 200, U == 150
#   Accuracy : 0.7082          
#   95% CI : (0.6657, 0.7481)
#   Sensitivity : 0.8690          
#   Specificity : 0.3571
#
# O == 300, U == 133.4
#   Accuracy : 0.7429         
#   95% CI : (0.7017, 0.781)
#   Sensitivity : 0.8988
#   Specificity : 0.4026
#
# O == 400, U == 125
#   Accuracy : 0.7612          
#   95% CI : (0.7209, 0.7983)
#   Sensitivity : 0.9405          
#   Specificity : 0.3701
#
# O == 500, U == 120
#   Accuracy : 0.7122         
#   95% CI : (0.6699, 0.752)
#   Sensitivity : 0.8631         
#   Specificity : 0.3831
#
# O == 800, U == 112.5
#   Accuracy : 0.6653         
#   95% CI : (0.6216, 0.707)
#   Sensitivity : 0.8006         
#   Specificity : 0.3701 
#
# O == 1600, U == 106.25
#   Accuracy : 0.6408          
#   95% CI : (0.5966, 0.6834)
#   Sensitivity : 0.7589          
#   Specificity : 0.3831

dfTrain_smote_400 <- SMOTE(
  Class ~ ., 
  data  = dfTrain,
  k = 2,
  perc.over = 400,
  perc.under = 125,
  )

table(dfTrain_smote_400$Class) 

# Train an initial caret randomForest model with downSampling
ctrl <- trainControl(
  method = 'repeatedcv',
  number = 5,
  repeats = 3,
  verboseIter = T,
  search = 'grid',
  summaryFunction = twoClassSummary, 
  classProbs = T,
  savePredictions = T
  )

# Bad performing models
#
# Random forest
#   Accuracy : 0.5816
#   95% CI : (0.5366, 0.6257)
#   Sensitivity : 0.6577
#   Specificity : 0.4156

# Glm
#   Accuracy : 0.5755
#   95% CI : (0.5304, 0.6197)
#   Sensitivity : 0.6786
#   Specificity : 0.3506

# Boosted Logistic Regression
#   Accuracy : 0.6109          
#   95% CI : (0.5646, 0.6557)
#   Sensitivity : 0.7203 
#   Specificity : 0.3826

# Regularized random forest
#   Accuracy : 0.5306
#   95% CI : (0.4853, 0.5755)
#   Sensitivity : 0.5923
#   Specificity : 0.3961

# Regularized Logistic Regression
#   Accuracy : 0.5816 
#   95% CI : (0.5366, 0.6257)
#   Sensitivity : 0.6577 
#   Specificity : 0.4156

# Least Squares Support Vector Machine with Radial Basis Function Kernel
#   Accuracy : 0.6694 
#   95% CI : (0.6258, 0.7109)
#   Sensitivity : 0.7887
#   Specificity : 0.4091

# XgbTree
# nrounds = 250, max_depth = 2, eta = 0.3, gamma = 0, colsample_bytree = 0.6, min_child_weight = 1, subsample = 0.875
#   Accuracy : 0.6776 
#   95% CI : (0.6342, 0.7188)
#   Sensitivity : 0.7917
#   Specificity : 0.4286

# XgbLinear
# nrounds = 250, lambda = 1e-04, alpha = 0.01, eta = 0.3
#   Accuracy : 0.5245 
#   95% CI : (0.4792, 0.5695)
#   Sensitivity : 0.5952
#   Specificity : 0.3701


## Not so useful models for the ensemble

# L2 Regularized Support Vector Machine (dual) with Linear Kernel
#   Accuracy : 0.7694
#   95% CI : (0.7295, 0.806)
#   Sensitivity : 0.8185
#   Specificity : 0.6623

# KNN k == 1
#   Accuracy : 0.8143
#   95% CI : (0.777, 0.8478)
#   Sensitivity : 0.8929
#   Specificity : 0.6429
#
# KNN k == 2
#   Accuracy : 0.7327
#   95% CI : (0.6911, 0.7714)
#   Sensitivity : 0.7292
#   Specificity : 0.7403
#
# KNN k == 3
#   Accuracy : 0.6612
#   95% CI : (0.6174, 0.7031)
#   Sensitivity : 0.6131
#   Specificity : 0.7662


# Useful models for the ensemble
# KKNN
#   Accuracy : 0.8837 
#   95% CI : (0.8519, 0.9107)
#   Sensitivity : 0.9405
#   Specificity : 0.7597 
tunegrid <- expand.grid(
  .kmax = 5,
  .distance = seq(0.5, 1, 0.05),
  .kernel = 'optimal'
  )

model_kknn_all <- caret::train(
  Class ~ .,
  data = dfTrain_smote_400,
  method = 'kknn',
  tuneGrid = tunegrid,
  trControl = ctrl
  )

validate(model_kknn_all, dfTest)

# L2 Regularized Linear Support Vector Machines with Class Weights
#   Accuracy : 0.7776
#   95% CI : (0.7381, 0.8136)
#   Sensitivity : 0.8690
#   Specificity : 0.5779
svm_ctrl <- trainControl(
  method = 'repeatedcv',
  number = 5,
  repeats = 3,
  verboseIter = T,
  search = 'grid',
  classProbs = F,
  savePredictions = T
  )

tunegrid <- expand.grid(
  .cost = c(8, 16, 32), 
  .Loss = 'L1',
  .weight = seq(9, 11, 1)
  )

model_svmlr_all <- caret::train(
  Class ~ .,
  data = dfTrain_smote_400,
  method = 'svmLinearWeights2',
  tuneGrid = tunegrid,
  trControl = svm_ctrl
  )

validate(model_svmlr_all, dfTest)

# Greedy Prototype Selection
#   Accuracy : 0.8163
#   95% CI : (0.7791, 0.8496)
#   Sensitivity : 0.8958
#   Specificity : 0.6429
tunegrid <- expand.grid(
  .eps = 5, 
  .Minkowski = 0.5
  )

model_proto_all <- caret::train(
  Class ~ .,
  data = dfTrain_smote_400,
  method = 'protoclass',
  tuneGrid = tunegrid,
  trControl = svm_ctrl
  )

validate(model_proto_all, dfTest)

# Create the ensemble model
#   Accuracy : 0.8939 
#   95% CI : (0.8632, 0.9197)
#   Sensitivity : 0.9345  
#   Specificity : 0.8052
dfEnsemble <- data.frame(
  kknn = predict(model_kknn_all, dfTest),
  proto = predict(model_proto_all, dfTest),
  svmlr = predict(model_svmlr_all, dfTest)
  )

ensemble(dfEnsemble)
