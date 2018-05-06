#Data is confidential

#Functions
dfnull <- function(df){
  df_null <- data.frame(apply(apply(df, 2, FUN = 'is.na'), 2, FUN = 'sum'))
  colnames(df_null)[1] <- "# of Null"

  return(df_null)
}

dffill <- function(df, col1, col2){
  V <- 0
  for (i in 1:nrow(df)){
    if (is.na(df[i, col1]) == T){
      v <- df[i, col2]
      df[i, col1] <- fn$sqldf("SELECT $col1
                               FROM df
                               WHERE $col1 IS NOT NULL AND $col2 = $v
                               LIMIT 1"
                              )
    }
  }
  return(df)
}

#Import data
df_cld <- read.csv("F:/Code/Glovo/Courier_lifetime_data.csv", stringsAsFactors = F)
df_cwd <- read.csv("F:/Code/Glovo/Courier_weekly_data.csv", stringsAsFactors = F)

colnames(df_cld) <- c("c", "f_1", "f_2")

#Join the dataframes into one
library(sqldf)

df_data <- sqldf("SELECT *
                 FROM df_cwd INNER JOIN df_cld ON df_cwd.courier = df_cld.c"
                 )

#Data Mungings
rm(df_cld)
rm(df_cwd)

df_data$c <- NULL
df_null <- dfnull(df_data)

#Data imputation
#Without knowing what f_2 stands for, it is really difficult to select the correct variables and way for imputation.
#The thinking behind the solution, is to impute only the latest month available in the data, and use the imputed valuea for the other months of the courier

df_impu <- sqldf("SELECT *
                  FROM df_data
                  WHERE df_data.f_2 IS NULL
                  GROUP BY df_data.courier
                  HAVING df_data.week = MAX(df_data.week)"
                )

df_miss <- sqldf("SELECT *
                  FROM df_data
                  WHERE df_data.f_2 IS NULL" 
                )

df_full <- sqldf("SELECT *
                  FROM df_data
                  WHERE df_data.f_2 IS NOT NULL"
                )

df_imputable <- rbind(df_full, df_impu)

library(mice)
miceMod <- mice(df_imputable[, !names(df_imputable) %in% c("courier", "week", "f_1a", "f_1b", "f_1c")], method = "rf")
df_miceOutput <- cbind(df_imputable[, names(df_imputable) %in% c("courier", "week")], complete(miceMod), df_imputable[, names(df_imputable) %in% c("f_1a", "f_1b", "f_1c")])
anyNA(df_miceOutput)
rm(miceMod)

names(df_miceOutput) <- names(df_miss)
df_data <- rbind(df_miceOutput, df_miss)
df_data <- df_data[order(df_data$courier), ]
rownames(df_data) <- seq(1, nrow(df_data))

rm(df_imputable)
rm(df_full)
rm(df_miss)
rm(df_miceOutput)
rm(df_impu)
rm(df_null)

df_data <- dffill(df_data, "f_2", "courier")

#Dummifing f_1 and f_2
library(dummies)
m_temp <- dummy(df_data$f_1)

df_data <- cbind(df_data, m_temp)
df_data$f_1 <- NULL
df_data$f_1c <- NULL

m_temp <- dummy(df_data$f_2)
df_data <- cbind(df_data, m_temp)
df_data$f_2 <- NULL
df_data$f_266 <- NULL
rm(m_temp)

#Exploratory Data Analysis
hist(df_data$feature_3) # The population from which the data was generated, follows a Poisson distribution

library(corrplot)
corrplot(cor(df_data[, 3:19]), method = "number")
#Problematic variables due to high +/- correlation:
#Feature_2 with Feature_3 and Feature_11
#Feature_3 with Feature_11 and Feature_16
#Feature_4 with Feature_5 and Feature_9
#Feature_5 with Feature_9
#Feature_8 with Feature_13,
#Feature_9 with Feature_4, Feature_5 and Feature_14
#Feature_11 with Feature_2, and Feature_3
#Maybe Feature_13 with Feature_14

#Labeling and taking care of the bias
df_data <- sqldf("SELECT *, 
                        (CASE 
                           WHEN mw >= 8 THEN 0
                           ELSE 1
                         END) AS 'Category'
                  FROM df_data 
                       LEFT JOIN (SELECT courier AS 'c', MAX(week) AS 'mw'
                                  FROM df_data
                                  GROUP BY courier) AS 'sub'
                       ON sub.c = df_data.courier"
                  )
        
df_data$mw <- NULL
df_data$c <- NULL
df_data <- df_data[df_data$week < 8,]

#Create training and test datasets
library(caret)
id <- createDataPartition(df_data$Category, p = 0.8, list = F)

df_train <- df_data[id, ]
df_test <- df_data[-id, ]

#Logistic regression
glm_fit <- glm(Category ~ ., family = binomial(logit), data = df_train)

#Model Analysis
library(pscl)
library(MASS)
library(pROC)
library(car)

summary(glm_fit)
exp(confint(glm_fit))
confint(glm_fit)
plot(glm_fit$fitted)

anova(glm_fit, test = "Chisq")
pR2(glm_fit)

fit_step <- stepAIC(glm_fit, method = "both")
summary(fit_step)
vif(fit_step)

#Create predictions
df_preds <- cbind(data.frame(Predictions = predict(fit_step, df_test[, -c(ncol(df_test))], type = 'response')), Original = df_test[, ncol(df_test)])
df_preds$Predictions <- ifelse(df_preds$Predictions >= 0.5, 1, 0)
conf_matrix <- table(df_preds$Predictions, df_preds$Original)
print(conf_matrix)

#Evaluating the model
#Since there was no constrictions whether false positives or false negatives have higher impact, the evaluation metric is the Receiver Operating Characteristic 
roccurve <- roc(df_preds$Original, df_preds$Predictions)
plot(roccurve)
auc(roccurve)

