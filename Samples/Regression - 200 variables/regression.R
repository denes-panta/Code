
#### Library import
library(caret)
library(moments)

#### Set seed
set.seed(1)


#### Loading
setwd('D:\\Code\\Code\\Samples\\Regression - 200 variables')

dfTrain <- read.csv('resources\\train.csv', stringsAsFactors = F)
dfTest <- read.csv('resources\\test.csv', stringsAsFactors = F)


#### Functions

# Get the mode of a column with NAs
mode <- function(column) {
  vUnique <- unique(column)
  vUnique[which.max(tabulate(match(column, vUnique)))]
  
  if(is.na(vUnique[1]) == T) {
    
    return(vUnique[2]) 
    
  } else {
    
    return(vUnique[1])
    
  }
}

# Fill the NAs in each column of a data.frame from a a vector of values
fill_na <- function(df, inputs) {
  for(c in 1:ncol(df)) {
    df[is.na(df[, c]), c] <- inputs[c]
  }
  
  return(df)
}

# Get the number of unique elements of a vector
len_unique <- function(column) { 
  
  return(length(unique(column))) 
  
}

# Get p values from the ANOVAs for the categorical variables
anova_categorical <- function(df) {
  vPvals <- c()
  
  for(c in 1:ncol(df)) {
    if(names(df)[c] == 'target') {
      
      vPvals <- c(vPvals, 0)
      
    } else {
      vFormula <- paste('target ~ ', names(df)[c])
      
      model_lm_aov <- lm(formula = vFormula, df)
      model_aov <- aov(model_lm_aov)
      vPvals <- c(vPvals, summary(model_aov)[[1]][['Pr(>F)']][1])
    }
  }
  
  return(vPvals)
  
}

# Model validation
validate <- function(model, dfTest, vTarget){
  
  vPredicted <- predict(model, dfTest)
  
  dfError <- data.frame(
    Preds = vPredicted ,
    Valid = vTarget
  )
  
  return(RMSE(dfError$Preds, dfError$Valid))
  
}


#### Missing value imputation

# Extact the target
vTarget <- dfTrain$target
dfTrain$target <- NULL

# Merge the training and test sets for missing value handling
dfTrain$set <- 'train'
dfTest$set <- 'test'

dfData <- rbind(dfTrain, dfTest)

# Check for duplicates
dfData[duplicated(dfData)]

# Check for missing values
colSums(is.na(dfData))

# Change the missing values to NA
dfData[dfData == 'NaN'] <- NA
dfData[dfData == '<undefined>'] <- NA

# Extract the set variable
vSet = dfData$set
dfData$set <- NULL

# Separate the categorical and numerical variables
dfChar <- dfData[, sapply(dfData, class) == 'character']
dfNum <- dfData[, sapply(dfData, class) == 'numeric']

# Check for binomially distributed variables
vBinom <- sapply(dfNum[complete.cases(dfNum), ], len_unique)
vBinom <- vBinom[vBinom == 2]

# Extract the found variables from the numerical to the categorical data.frame
dfChar <- cbind(dfChar, dfNum[, names(dfNum) %in% names(vBinom)])
dfNum <- dfNum[!names(dfNum) %in% names(vBinom)]
dfChar[, ] <- lapply(dfChar[, ], as.character)

# Impute the missing categorical variables with the mode
dfChar <- fill_na(dfChar, sapply(dfChar, mode))

# Impute the missing continuous variables with the median
dfNum <- fill_na(dfNum, sapply(dfNum[complete.cases(dfNum), ], median))

# Merge the two imputed data.frames
dfData <- cbind(dfNum, dfChar)

# Recreate the set variable
dfData$set <- vSet

# Separate the training and test sets
dfTrain <- dfData[dfData$set == 'train', ]
dfTest <- dfData[dfData$set == 'test', ]

# Recreate the target variable
dfTrain$target <- vTarget

# Remove the set variables
dfTrain$set <- NULL
dfTest$set <- NULL


#### Exploratory data analysis

# Separate the numerical and categorical variables from the train data
dfChar <- dfTrain[, sapply(dfTrain, class) == 'character']
dfNum <- dfTrain[, sapply(dfTrain, class) == 'numeric']
dfChar$target <- dfNum$target

### Continuous variables

# The correlation shows that most continuous variables 
dfCor <- as.data.frame(cor(dfNum))
hist(dfCor$target, breaks = 20)

# Create a data.frame for the meta data from the continous variables
dfMeta_Num <- data.frame(corr = dfCor$target)
row.names(dfMeta_Num) <- names(dfCor)

# Check for normally distributed variables using the shapiro-wilk test
lSW <- lapply(dfNum, shapiro.test)
dfMeta_Num$SW_p_val <- sapply(lSW, "[[", 2)

# Check for skewness and kurtosis
dfMeta_Num$skew <- sapply(dfNum, skewness)
dfMeta_Num$kurt <- sapply(dfNum, kurtosis)

# Remove the variable with close to 0 correlation (between -0.05 and 0.05)
dfMeta_Num <- dfMeta_Num[(dfMeta_Num$corr <= -0.05 | dfMeta_Num$corr >= 0.05), ]
dfNum <- dfNum[names(dfNum) %in% row.names(dfMeta_Num)]


### Categorical

# Create meta data.frame for the categorical variables
dfMeta_Cat <- data.frame(anove_p = anova_categorical(dfChar))
row.names(dfMeta_Cat) <- names(dfChar)

# Keep only the significant varaibles based on the anova test
dfMeta_Cat <- dfMeta_Cat[dfMeta_Cat$anove_p <= 0.05, , drop = F]
dfChar <- dfChar[names(dfChar) %in% row.names(dfMeta_Cat)]

# Apply Tukey test on each remaining multi categorical variable
model_lm_var004 <- lm(formula = target ~ var004, dfChar)
model_aov_var004 <- aov(model_lm_var004)
summary(model_aov_var004)
TukeyHSD(model_aov_var004)

# There seems to be no difference between a and b categories so merge these
dfChar$var004[dfChar$var004 == 'a'] <- 'b'

# Apply linear model for the categorical variables
model_lm_cat <- lm(formula = target ~ ., dfChar)
summary(model_lm_cat)


## Combine the two dataframes
dfTrain <- cbind(dfChar, dfNum)


### Remove outliers based on continous variables
boxplot(dfNum)

# Var005 had some outliers
vCutoff <- quantile(dfNum$var005, c(.01, .99))
dfNum <- dfNum[(dfNum$var005 > vCutoff[1] & dfNum$var005 < vCutoff[2]), ]

boxplot(dfNum)

# Var008 seems to be fine

# var009 seems to have some outliers, but removing them hurts the performance

# var024 seems to have some outliers, but removing them hurts the performance

# var044 seems to have some outliers, but removing them hurts the performance

# var045 seems to be fine

# var072 seems to have some outliers, but removing them hurts the performance

# var074 seems to have some outliers, but removing them hurts the performance

# var125 had some outliers
vCutoff <- quantile(dfNum$var125, c(.02, .99))
dfNum <- dfNum[(dfNum$var125 > vCutoff[1] & dfNum$var125 < vCutoff[2]), ]

boxplot(dfNum)

# var130 seems to have some outliers, but removing them hurts the performance

# var170 seems to be fine

# var175 had some outliers
vCutoff <- quantile(dfNum$var175, c(.01, .99))
dfNum <- dfNum[(dfNum$var175 > vCutoff[1] & dfNum$var175 < vCutoff[2]), ]

boxplot(dfNum)

# var178 had some outliers
vCutoff <- quantile(dfNum$var178, c(.01, .99))
dfNum <- dfNum[(dfNum$var178 > vCutoff[1] & dfNum$var178 < vCutoff[2]), ]

boxplot(dfNum)

# Apply a linear model to test performance
model_lm_num <- lm(formula = target ~ ., dfNum)
summary(model_lm_num)


#### Save the preprocessed data
write.csv(dfTrain, 'train_processed.csv', row.names = F)


#### Use the caret package to train various algorithms

# Turn the categorical variables into factors
dfTrain[, 1:4] <- lapply(dfTrain[, 1:4], as.factor)

vId <- createDataPartition(dfTrain$target, times = 1, p = 0.8, list = F)

dfTrain_t <- dfTrain[vId, ]
dfTrain_v <- dfTrain[-vId, ]

# Define train control
ctrl <- trainControl(
  method = 'repeatedcv',
  number = 5,
  repeats = 3,
  verboseIter = T,
  search = 'grid',
  savePredictions = T
  )

# Elastic net regression - RMSE : 0.7392751
model_enet <- caret::train(
  target ~ .,
  data = dfTrain_t,
  method = 'enet',
  tuneLength = 10,  
  trControl = ctrl
  )

validate(model_enet, dfTrain_v, dfTrain_v$target)

# kknn - RMSE : 0.5844215
model_kknn <- caret::train(
  target ~ .,
  data = dfTrain_t,
  method = 'kknn',
  tuneLength = 10,  
  trControl = ctrl
  )

validate(model_kknn, dfTrain_v, dfTrain_v$target)

# svmLinear - RMSE : 0.8127583
model_svmlin <- caret::train(
  target ~ .,
  data = dfTrain_t,
  method = 'svmLinear',
  trControl = ctrl
  )

validate(model_svmlin, dfTrain_v, dfTrain_v$target)

# svmRadial - RMSE : 0.5174954
model_svmrad <- caret::train(
  target ~ .,
  data = dfTrain_t,
  method = 'svmRadial',
  tuneLength = 10,  
  trControl = ctrl
  )

validate(model_svmrad, dfTrain_v, dfTrain_v$target)

# Randomforest - RMSE : 0.5016872
model_rf <- caret::train(
  target ~ .,
  data = dfTrain_t,
  method = 'rf',
  tuneLength = 10,
  trControl = ctrl
  )

validate(model_rf, dfTrain_v, dfTrain_v$target)

# XgbTree - RMSE : 0.5157453
model_xgbtree <- caret::train(
  target ~ .,
  data = dfTrain_t,
  method = 'xgbTree',
  tuneLength = 5,
  trControl = ctrl
  )

validate(model_xgbtree, dfTrain_v, dfTrain_v$target)


#### Train an randomforest model on the full training dataset and make predictions
model_rf_full <- caret::train(
  target ~ .,
  data = dfTrain,
  method = 'rf',
  tuneLength = 10,
  trControl = ctrl
  )


# Select the necesary variables
dfTest <- dfTest[, names(dfTest) %in% names(dfTrain)]

# Apply the same pre-processing as it was for the training data
dfTest$var004[dfTest$var004 == 'a'] <- 'b'
dfTest[, 14:17] <- lapply(dfTest[, 14:17], as.factor)

# Make predictions
dfPreds <- data.frame(preds = predict(model_rf_full, dfTest))

# Write the predictions to file
write.csv(dfPreds, 'preds.csv')
