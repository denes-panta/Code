############################################################
#Based on Practices from: 'R Data Mining Blueprints (2016)'#
############################################################

#Functions
dfnull <- function(df){
  df_null <- data.frame(apply(apply(df, 2, FUN = 'is.na'), 2, FUN = 'sum'))
  colnames(df_null)[1] <- "# of Null"
  return(df_null)
}

set.seed(117)

#Import data
df_art<- read.csv("F:/Code/Code/Data Science Samples/Classification/ArtPiece.csv", stringsAsFactors = T)
str(df_art)

dfNull <- dfnull(df_art)

df_art <- df_art[complete.cases(df_art),]

df_art$IsGood.Purchase <- as.integer(df_art$IsGood.Purchase)
df_art$Is.It.Online.Sale <- as.integer(df_art$Is.It.Online.Sale)

#Dummify the categorical variables
library(dummies)

df_art <- cbind(df_art, dummy(df_art$Brush.Size))
df_art$df_artNULL <- NULL
df_art$Brush.Size <- NULL

#Create training and validation sets
library(caret)

tt_id <- createDataPartition(df_art$IsGood.Purchase, p = 0.7, list = F)

df_train <- df_art[tt_id, ]
df_test <- df_art[-tt_id, ]

#Logistic Regression
glm_fit <- glm(IsGood.Purchase ~ ., 
               family = binomial(logit),
               data = df_train
               )

#Model Analysis
summary(glm_fit)
exp(confint(glm_fit))
confint(glm_fit)
anova(glm_fit, test = "Chisq")

plot(glm_fit$fitted)

#Model Analysis
library(MASS)
library(pROC)

fit_step <- stepAIC(glm_fit, method = "both")
summary(fit_step)

vif(fit_step)

df_train$prob <- predict(fit_step, type = "response")
train_ROC <- roc(IsGood.Purchase ~ prob, data = df_train)
print(train_ROC)
plot(train_ROC)

df_train$prob <- ifelse(df_train$prob > 0.5, "Yes", "No")
train_table <- table(df_train$prob, df_train$IsGood.Purchase)
prop.table(train_table)

#Prediction on test data
df_test$goodP <- predict(glm_fit, newdata = df_test, type = "response")
test_ROC <- roc(IsGood.Purchase ~ goodP, data = df_test)
print(test_ROC)
plot(test_ROC)

df_test$goodP <- ifelse(df_test$goodP > 0.5, "Yes", "No")
test_table <- table(df_test$goodP, df_test$IsGood.Purchase)
prop.table(test_table)
