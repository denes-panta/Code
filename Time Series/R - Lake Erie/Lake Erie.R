#Import libraries
library(tseries)
library(forecast)
library(Metrics)

#Data munging
dfData <- read.csv("F:\\Code\\Code\\Time Series\\Lake Erie\\Monthly Water Levels.csv")

#Data Munging
colnames(dfData) <- c("Month", "Meter")
dfData[, "Month"] <- as.Date(paste0(dfData[, "Month"], "-01"))
dfTrain <- dfData[1:(nrow(dfData)-24), ]
dfValid <- dfData[(nrow(dfData)-23):nrow(dfData), ]

#Convert the DataFrame to Time Series
iStartY <- as.numeric(substr(as.character(min(dfTrain[, "Month"])), 1, 4))
iStartM <- as.numeric(substr(as.character(min(dfTrain[, "Month"])), 6, 7))

iEndY <- as.numeric(substr(as.character(max(dfTrain[, "Month"])), 1, 4))
iEndM <- as.numeric(substr(as.character(max(dfTrain[, "Month"])), 6, 7))

tsData <- ts(data = dfTrain[, "Meter"], 
              start = c(iStartY, iStartM),
              end = c(iEndY, iEndM),
              frequency = 12
              )

#Descriptive Statistics
#Run a few basic metrics
start(tsData)
end(tsData)
frequency(tsData)
cycle(tsData)

#Summarize the data
summary(tsData)

#Plot the TimeSeries and the regression line
plot(tsData)
abline(reg = lm(tsData~time(tsData)))

#Plot the Seasonal-Trend Decomposition
plot(stl(tsData, s.window = "periodic"))

#Plot the Seasonal effect
boxplot(tsData~cycle(tsData))

#Check for stationarity
adf.test(tsData, alternative = "stationary", k = 0)
PP.test(tsData)
kpss.test(tsData, null = "Level")
kpss.test(tsData, null = "Trend")

#ACF - PACF
acf(tsData)
pacf(tsData)

acf(diff(tsData))
pacf(diff(tsData))

#Fit Arima model
modelArima <- arima(x = tsData, 
                    order = c(0, 1, 0), 
                    seasonal = list(order = c(0, 1, 2), period = 12)
                    )
#Check AIC
print(modelArima)

#Create predictions for 1 year forward
tsPreds <- predict(modelArima, 
                   n.ahead = 1*24
                   )

#Get the Error between the predictions and real values
cat("RMSE: ", rmse(as.matrix(tsPreds$pred), as.matrix(dfValid$Meter)), "AiC: ", modelArima$aic)

#Visualise the prediction
ts.plot(tsData,
        tsPreds$pred, 
        log = "y", 
        lty = c(1, 3)
        )
