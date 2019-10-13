### Libraries, seed and data load
library('rmarkdown')
library('randomForest')
library('doParallel')
library('dummies')
library('mice')
library('abind')
library('plyr')
library('ggplot2')
library('corrplot')
library('reshape2')
library('Metrics')
library('magrittr')

setwd('D:\\Code\\Interviews\\dataroots\\data')
set.seed(1)

dfTrip <- read.csv('trip_data.csv', stringsAsFactors = F)
dfStation <- read.csv('station_data.csv', stringsAsFactors = F)
dfWeather <- read.csv('weather_data_imp.csv', stringsAsFactors = F)

### Pre-processing
# Turn trips into demand data and prepare for merge
dfTrip$dem_day <- substr(dfTrip$Start.Date, 1, 2)
dfTrip$dem_month <- substr(dfTrip$Start.Date, 4, 5)
dfTrip$dem_year <- substr(dfTrip$Start.Date, 7, 10)
dfTrip$dem_hour <- substr(dfTrip$Start.Date, 12, 13)
dfTrip$Start.Date <- substr(dfTrip$Start.Date, 1, 10)
dfTrip$Trip.ID <- NULL
dfTrip$End.Date <- NULL
dfTrip$End.Station <- NULL

names(dfTrip)[names(dfTrip) == 'Start.Station'] <- 'station'

# Create the customer and subscriber data frames
dfCustomer <- data.frame(
  Date = dfTrip$Start.Date,
  Hour = dfTrip$dem_hour,
  Station = dfTrip$station,
  User = dfTrip$Subscriber.Type
)
dfCustomer$Date <- as.character(dfCustomer$Date)
dfCustomer$Subscriber = 0
dfCustomer$Customer = 0
dfCustomer[dfCustomer$User == 'Subscriber', 'Subscriber'] <- 1
dfCustomer[dfCustomer$User == 'Customer', 'Customer'] <- 1
dfCustomer$User <- NULL
dfSubscriber <- dfCustomer
dfSubscriber$Customer <- NULL
dfCustomer$Subscriber <- NULL
dfSubscriber <- aggregate(Subscriber ~ ., data = dfSubscriber, FUN = sum)
dfCustomer <- aggregate(Customer ~ ., data = dfCustomer, FUN = sum)
dfTrip$Subscriber.Type <- NULL

# Create the Federal variable
dfTrip$Federal <- 0
dfTrip[dfTrip$Start.Date == '01/01/2014', 'Federal'] <- 1
dfTrip[dfTrip$Start.Date == '20/01/2014', 'Federal'] <- 1
dfTrip[dfTrip$Start.Date == '17/02/2014', 'Federal'] <- 1
dfTrip[dfTrip$Start.Date == '26/05/2014', 'Federal'] <- 1
dfTrip[dfTrip$Start.Date == '04/07/2014', 'Federal'] <- 1
dfTrip[dfTrip$Start.Date == '01/09/2014', 'Federal'] <- 1
dfTrip[dfTrip$Start.Date == '13/10/2014', 'Federal'] <- 1
dfTrip[dfTrip$Start.Date == '11/11/2014', 'Federal'] <- 1
dfTrip[dfTrip$Start.Date == '27/11/2014', 'Federal'] <- 1
dfTrip[dfTrip$Start.Date == '25/12/2014', 'Federal'] <- 1
dfTrip[dfTrip$Start.Date == '31/12/2014', 'Federal'] <- 1

dfTrip[dfTrip$Start.Date == '01/01/2015', 'Federal'] <- 1
dfTrip[dfTrip$Start.Date == '19/01/2015', 'Federal'] <- 1
dfTrip[dfTrip$Start.Date == '16/02/2015', 'Federal'] <- 1
dfTrip[dfTrip$Start.Date == '25/05/2015', 'Federal'] <- 1
dfTrip[dfTrip$Start.Date == '03/07/2015', 'Federal'] <- 1
dfTrip[dfTrip$Start.Date == '04/07/2015', 'Federal'] <- 1
dfTrip[dfTrip$Start.Date == '07/09/2015', 'Federal'] <- 1
dfTrip[dfTrip$Start.Date == '12/10/2015', 'Federal'] <- 1
dfTrip[dfTrip$Start.Date == '11/11/2015', 'Federal'] <- 1
dfTrip[dfTrip$Start.Date == '26/11/2015', 'Federal'] <- 1
dfTrip[dfTrip$Start.Date == '31/12/2015', 'Federal'] <- 1

# Create the bike_demand variable
dfTrip$bike_demand <- 1

# Create the weekend variable
dfTrip$weekdays <- weekdays(as.Date(dfTrip$Start.Date))
dfTrip$weekdays <- mapvalues(
  dfTrip$weekdays, 
  from = c('Mittwoch', 
           'Dienstag', 
           'Montag', 
           'Sonntag', 
           'Freitag', 
           'Donnerstag', 
           'Samstag'
  ),
  to = c(1, 2, 3, 4, 5, 6, 7)
)
dfTrip$weekdays <- as.numeric(dfTrip$weekdays)

# Prepare station data for merge
dfStation$Zip <- 0
dfStation[dfStation$City == 'San Jose', 'Zip'] <- 95113
dfStation[dfStation$City == 'Redwood City', 'Zip'] <- 94063
dfStation[dfStation$City == 'Palo Alto', 'Zip'] <- 94301
dfStation[dfStation$City == 'San Francisco', 'Zip'] <- 94107
dfStation[dfStation$City == 'Mountain View', 'Zip'] <- 94041

# Prepare the weather data for merge
dfWeather[dfWeather$Events == '', 'Events'] <- 'None'

# Create dummy variables for the Events
dfTemp <- dummy(dfWeather$Events, sep = '_')
dfWeather <- cbind(dfWeather, dfTemp)
dfWeather$Events <- NULL
dfWeather$Events_None <- NULL

vDate <- dfWeather$Date
dfWeather$Date <- NULL

# Count the missing values by column
na_count <- sapply(dfWeather, function(y) sum(length(which(is.na(y)))))
md.pattern(dfWeather)

# Missing data imputation
model_imp <- mice(dfWeather, m = 30, maxit = 50, meth = 'rf', seed = 1)
summary(model_imp)
densityplot(model_imp)

temp3d <- abind(
  complete(model_imp, 1),
  complete(model_imp, 2),
  complete(model_imp, 3),
  complete(model_imp, 4),
  complete(model_imp, 5),
  complete(model_imp, 6),
  complete(model_imp, 7),
  complete(model_imp, 8),
  complete(model_imp, 9),
  complete(model_imp, 10),
  complete(model_imp, 11),
  complete(model_imp, 12),
  complete(model_imp, 13),
  complete(model_imp, 14),
  complete(model_imp, 15),
  complete(model_imp, 16),
  complete(model_imp, 17),
  complete(model_imp, 18),
  complete(model_imp, 19),
  complete(model_imp, 20),
  complete(model_imp, 21),
  complete(model_imp, 22),
  complete(model_imp, 23),
  complete(model_imp, 24),
  complete(model_imp, 25),
  complete(model_imp, 26),
  complete(model_imp, 27),
  complete(model_imp, 28),
  complete(model_imp, 29),
  complete(model_imp, 30),
  along = 3)

dfWeather_imp <- as.data.frame(rowMeans(temp3d, dims = 2))
dfWeather_imp$Date <- vDate

write.csv(x = dfWeather_imp, file = 'weather_data_imp.csv', row.names = F)
dfWeather <- dfWeather_imp

# Create the Main Dataset by joining the preprocessed datasets
dfData <- aggregate(bike_demand ~ ., data = dfTrip, FUN = sum)

dfData <- merge(
  x = dfData, 
  y = dfStation, 
  by.x = 'station', 
  by.y = 'Id', 
  all.x = T
)

dfData <- merge(
  x = dfData,
  y = dfWeather,
  by.x = c('Zip', 'Start.Date'), 
  by.y = c('Zip', 'Date'), 
  all.x = T
)

dfData <- merge(
  x = dfData,
  y = dfCustomer,
  by.x = c('Start.Date', 'dem_hour', 'station'), 
  by.y = c('Date', 'Hour', 'Station'), 
  all.x = T
)

dfData <- merge(
  x = dfData,
  y = dfSubscriber,
  by.x = c('Start.Date', 'dem_hour', 'station'), 
  by.y = c('Date', 'Hour', 'Station'), 
  all.x = T
)

dfData <- dfData[
  order(dfData$dem_year, 
        dfData$dem_month, 
        dfData$dem_day, 
        dfData$dem_hour
  ), ]

write.csv(dfData, 'dfData_Full.csv', row.names = F)


### Exploratory data analysis
dfData <- read.csv('dfData_Full.csv')

# Check correlation between weather and demand Customer Variable
dfCorr <- aggregate(Customer ~ Start.Date, dfData, sum)
dfCorr <- merge(
  x = dfCorr,
  y = dfWeather,
  by.x = c('Start.Date'), 
  by.y = c('Date'), 
  all.x = T
)

dfCorr$Start.Date <- NULL
dfCorr$Events_Rain.Thunderstorm <- NULL
dfCorr$Events_Rain <- NULL
dfCorr$Events_Fog.Rain <- NULL
dfCorr$Events_Fog <- NULL
dfCorr$Zip <- NULL

corrplot(cor(dfCorr), method = 'circle')

# Check correlation between weather and demand Subscriber Variable
dfCorr <- aggregate(Customer ~ Start.Date, dfData, sum)
dfCorr <- merge(
  x = dfCorr,
  y = dfWeather,
  by.x = c('Start.Date'), 
  by.y = c('Date'), 
  all.x = T
)

dfCorr$Start.Date <- NULL
dfCorr$Events_Rain.Thunderstorm <- NULL
dfCorr$Events_Rain <- NULL
dfCorr$Events_Fog.Rain <- NULL
dfCorr$Events_Fog <- NULL
dfCorr$Zip <- NULL

corrplot(cor(dfCorr), method = 'circle')

# Check the demand for the stations
dfData %>%
  melt(id.vars = 'station', measure.vars = c('Customer', 'Subscriber')) %>%
  ggplot(
    aes(x = station, y = value, fill = variable)) +
    geom_bar(stat = "identity") +
    ggtitle('Bike Demand by Station') +
    xlab('Station number') +
    ylab('Bike Demand')

# Check the demand for each weekday
dfData %>%
  melt(id.vars = 'weekdays', measure.vars = c('Customer', 'Subscriber')) %>%
  ggplot(
    aes(x = weekdays, y = value, fill = variable)) +
    geom_bar(stat = "identity") +
    ggtitle('Bike Demand by weekday') +
    xlab('Weekday (1 = Monday, 7 = Sunday') +
    ylab('Bike Demand')

# Check the demand for each month
dfData %>%
  melt(id.vars = 'dem_month', measure.vars = c('Customer', 'Subscriber')) %>%
  ggplot(
    aes(x = dem_month, y = value, fill = variable)) +
    geom_bar(stat = "identity") +
    ggtitle('Bike Demand by Month') +
    xlab('Month (1 = January, 12 = December') +
    ylab('Bike Demand')

# Check the demand for each hour
dfData %>%
  melt(id.vars = 'dem_hour', measure.vars = c('Customer', 'Subscriber')) %>%
  ggplot(
    aes(x = dem_hour, y = value, fill = variable)) +
    geom_bar(stat = "identity") +
    ggtitle('Bike Demand by Hour') +
    xlab('Hour') +
    ylab('Bike Demand')

# Check the relative demand for Federal holidays and regular days
dfData %>%
  melt(id.vars = 'Federal', measure.vars = c('Customer', 'Subscriber')) %>%
  aggregate(formula = value ~ ., data = ., FUN = sum) %>%
  as.data.frame() %>% 
  {. ->> dfFederalDemand}

dfFederalTotal = data.frame(aggregate(
  formula = value ~ Federal, 
  data = dfFederalDemand, 
  FUN = sum
))

dfFederalDemand <- merge(
  x = dfFederalDemand,
  y = dfFederalTotal,
  by.x = c('Federal'), 
  by.y = c('Federal'), 
  all.x = T
)

dfFederalDemand$value <- dfFederalDemand$value.x / dfFederalDemand$value.y

ggplot(
  data = dfFederalDemand,
  aes(x = Federal, y = value, fill = variable)) +
  geom_bar(stat = "identity") +
  ggtitle('Relative Bike Demand by Holiday/Non Holiday') +
  xlab('1 = Holiday, 0 = Regular day') +
  ylab('Percent')

# Check the total demand for each city by user group
dfData %>%
  melt(id.vars = 'City', measure.vars = c('Customer', 'Subscriber')) %>%
  aggregate(formula = value ~ ., data = ., FUN = sum) %>%
  ggplot(
    aes(x = City, y = value, fill = variable)) +
    geom_bar(stat = "identity") +
    ggtitle('Bike Demand by City') +
    xlab('City') +
    ylab('Bike Demand')

# Check the relative demand for each city by user group
dfData %>%
  melt(id.vars = 'City', measure.vars = c('Customer', 'Subscriber')) %>%
  aggregate(formula = value ~ ., data = ., FUN = sum) %>%
  as.data.frame() %>% 
  {. ->> dfCityDemand}

dfCityTotal = data.frame(aggregate(
  formula = value ~ City, 
  data = dfCityDemand, 
  FUN = sum
  ))

dfCityDemand <- merge(
  x = dfCityDemand,
  y = dfCityTotal,
  by.x = c('City'), 
  by.y = c('City'), 
  all.x = T
)

dfCityDemand$value <- dfCityDemand$value.x / dfCityDemand$value.y

ggplot(
  data = dfCityDemand,
  aes(x = City, y = value, fill = variable)) +
  geom_bar(stat = "identity") +
  ggtitle('Bike Demand by City User ration') +
  xlab('City') +
  ylab('Percent')

# check the demand for federal holidays agains normal days
dfData %>%
  melt(id.vars = 'Federal', measure.vars = c('Customer', 'Subscriber')) %>%
  aggregate(formula = value ~ ., data = ., FUN = sum) %>%
  as.data.frame() %>%
  ggplot(
    aes(x = Federal, y = value, fill = variable)) +
    geom_bar(stat = "identity") +
    ggtitle('Mean Bike Demand Holidays vs Non Holidays') +
    xlab('Federal holiday (1 = Holiday, 0 = Regular)') +
    ylab('Bike Demand')


### Variable Removal and dummifing
# Delete Variables that contain duplicate information
dfData$Start.Date <- NULL
dfData$Name <- NULL
dfData$City <- NULL

# Create the variable for the ratio of subscribers and customers
dfData$SubscriberPercent <- dfData$Subscriber / (dfData$Customer + dfData$Subscriber)
dfData$Customer <- NULL
dfData$Subscriber <- NULL

# Dummify the zip code
dfTemp <- dummy(dfData$Zip, sep = '_')
dfData <- cbind(dfData, dfTemp)
dfData$Zip <- NULL


### Feature Selection
# Load the full model
model_rf41 <- readRDS('model_rf41.rds')

# Variable importance plot  
varImpPlot(model_rf41)

# Filter the dataset based on variable importance of the full model
dfVarImp <- data.frame(model_rf41$importance)
dfVarImp <- dfVarImp[order(-dfVarImp$X.IncMSE), ]
vMust <- c('dem_day', 'dem_month', 'dem_year', 'dem_hour', 'dem_station', 'bike_demand')
vVars <- rownames(dfVarImp[dfVarImp$X.IncMSE > 0.15, ])
vVars <- c(vVars, vMust)
dfData <- dfData[, names(dfData) %in% vVars]

# Create the training and test datasets
vSepPoint <- as.integer(nrow(dfData) * 0.90)
dfTrain <- dfData[1:vSepPoint, ]
dfTestX <- dfData[(vSepPoint + 1):nrow(dfData), ]
dfTestY <- as.data.frame(dfTestX$bike_demand)
names(dfTestY) <- 'bike_demand'
dfTestX$bike_demand <- NULL
  
  # Train model with the significant variables
  model_rf <- randomForest(
    bike_demand ~ ., 
    data = dfTrain,
    xtest = dfTestX,
    ytest = dfTestY$bike_demand,
    ntree = 80, 
    mtry = as.integer(sqrt(length(dfTestX))),
    replace = F,
    importance = T,
    keep.forest = T,
    do.trace = T
  )

# Plot the errors
plot(model_rf$test$mse)

### Model Parameter tuning 
# Find the most appropriate mtry parameter
dfDataY <- as.data.frame(dfData$bike_demand)
names(dfDataY) <- 'bike_demand'
dfDataX <- dfData
dfDataX$bike_demand <- NULL

model_tune_rf <- tuneRF(
  x = dfDataX,
  y = dfDataY$bike_demand,
  ntreeTry = 50,
  stepFactor = 2,
  improve = 1e-5,
  trace = T,
  plot = T,
  doBest = F
)

saveRDS(model_tune_rf, 'model_tune_rf.rds')

### Train the final model with the best parameters
model_rf_final <- randomForest(
  bike_demand ~ ., 
  data = dfTrain,
  xtest = dfTestX,
  ytest = dfTestY$bike_demand,
  ntree = 100, 
  mtry = 6,
  replace = F,
  importance = T,
  keep.forest = T,
  do.trace = T
)

# Plot the MSE and R2 graphs
plot(model_rf_final$test$mse)
plot(model_rf_final$test$rsq)

dfRMSE <- data.frame(
  Preds = predict(model_rf_final, dfTestX),
  Test = dfTestY$bike_demand
  )

rmse(dfRMSE$Preds, dfRMSE$Test)
mae(dfRMSE$Preds, dfRMSE$Test)

# Save model
saveRDS(model_rf, 'model_final_rf.rds')
