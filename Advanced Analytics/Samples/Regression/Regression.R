############################################################
#Based on Practices from: 'R Data Mining Blueprints (2016)'#
############################################################

#Import data
df_cars <- read.csv("F:/Code/Code/Data Science Samples/Regression/Cars93_1.csv")

#Pre-processing
df_cars$Rear.seat.room[is.na(df_cars$Rear.seat.room)] <- median(df_cars$Rear.seat.room, na.rm = T)
df_cars$Luggage.room[is.na(df_cars$Luggage.room)] <- median(df_cars$Luggage.room, na.rm = T)

#Correlation
library(corrplot)

m_cor <- cor(df_cars)
corrplot(m_cor, method = "number")

#Fit the whole dataset and diagnose
lm_fit <- lm(MPG.Overall ~ ., data = df_cars)
summary(lm_fit)

summary.aov(lm_fit)
confint(lm_fit, level = 0.95)

par(mfrow = c(2, 2))
plot(lm_fit, col = "dodgerblue4")

influence.measures(lm_fit)
influenceIndexPlot(lm_fit, id.n = 3)
par(mfrow = c(1, 1))
influencePlot(lm_fit, id.n = 3, col = "red")

#Influenctial points
#Multiple R-squared:  0.7837,	Adjusted R-squared:  0.7481 
lm_fit <- lm(MPG.Overall ~ ., data = df_cars[-c(42, 28), ])
summary(lm_fit)

#Outliers
#Multiple R-squared:  0.8097,	Adjusted R-squared:  0.7775
qqPlot(lm_fit, id.n = 5)
influenceIndexPlot(lm_fit, id.n = 3)
influencePlot(lm_fit, id.n = 3, col = "blue")

lm_fit <- lm(MPG.Overall ~ ., data = df_cars[-c(42, 28, 39, 60, 83, 91, 5), ])
summary(lm_fit)

#Variance Inflation Factors to eliminate multicollinearity
vif(lm_fit)

vif_fit <- lm(MPG.Overall ~ ., data = df_cars[-c(42, 28, 39, 60, 83, 91, 5), !names(df_cars) %in% c("Weight", "EngineSize", "Width", "Horsepower")])
summary(vif_fit)
vif(vif_fit)

rows_o <- c(42, 28, 39, 60, 83, 91, 5)
cols_o <- c("Weight", "EngineSize", "Width", "Horsepower")

lm_fit <- lm(MPG.Overall ~ ., data = df_cars[-rows_o, !names(df_cars) %in% cols_o])
summary(lm_fit)

#Stepwise variable selection for final model
lm_model <- step(lm_fit, method = "both")

