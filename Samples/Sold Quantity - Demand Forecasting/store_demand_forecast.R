# Load the libraries
library(rnoaa)
library(dplyr)
library(data.table)
library(anytime)
library(lubridate)
library(hydroTSM)
library(ggplot2)
library(corrplot)
library(reshape2)
library(rworldmap)
library(randomForest)
library(xgboost)
library(caret)
library(Metrics)

# Change working directory
setwd('c:/test_input')

# Set seed
set.seed(1)


### Loading and Basic Pre-processing

# Load the Sales Data
dfSales <- read.csv('input_data/sales_data.csv', stringsAsFactors = F, sep = ';')
colnames(dfSales)[1] <- 'date'

# Replace decimal commas with dots
dfSales$unit_price <- gsub(",", ".", dfSales$unit_price)
dfSales$unit_price <- as.numeric(dfSales$unit_price)

# Load the Store Master data
dfStore <- read.csv('input_data/store_master.csv', stringsAsFactors = F, sep = ';')
colnames(dfStore)[1] <- 'id'

# Replace decimal commas with dots
dfStore$latitude <- gsub(",", ".", dfStore$latitude)
dfStore$latitude <- as.numeric(dfStore$latitude)
dfStore$longitude <- gsub(",", ".", dfStore$longitude)
dfStore$longitude <- as.numeric(dfStore$longitude)


### Try to get the weather data from the ghcnd stations

# Get the list of all available stations
if(!file.exists('ghcnd_stations.csv')) {
  dfGhcndStations <- ghcnd_stations()
  write.csv(x = dfGhcndStations, file = 'ghcnd_stations.csv', row.names = F)
  
} else {
  dfGhcndStations <- read.csv('ghcnd_stations.csv', stringsAsFactors = F)
  
}

# Get the 50 nearest stations for each store location
if(!file.exists('nearby_stations.csv')) {
  lStations <- meteo_nearby_stations(
    lat_lon_df = dfStore,
    lat_colname = 'latitude',
    lon_colname = 'longitude',
    station_data = dfGhcndStations,
    var = 'all',
    limit = 50
  )
  
  dfStations <- data.frame(
    matrix(
      unlist(lStations), 
      nrow = length(lStations), 
      byrow = T
    )
  )
  
  dfStations <- cbind(
    id = dfStore$id, 
    min_date = tapply(dfSales$date, dfSales$store, min),
    max_date = tapply(dfSales$date, dfSales$store, max),
    dfStations
  )
  
  dfStations <- rbindlist(
    list(
      dfStations[, c(1, 2, 3, seq(4, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(5, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(6, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(7, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(8, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(9, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(10, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(11, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(12, length(dfStations), 50))],      
      dfStations[, c(1, 2, 3, seq(13, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(14, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(15, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(16, length(dfStations), 50))],      
      dfStations[, c(1, 2, 3, seq(17, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(18, length(dfStations), 50))],         
      dfStations[, c(1, 2, 3, seq(19, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(20, length(dfStations), 50))],         
      dfStations[, c(1, 2, 3, seq(21, length(dfStations), 50))],         
      dfStations[, c(1, 2, 3, seq(22, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(23, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(24, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(25, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(26, length(dfStations), 50))],      
      dfStations[, c(1, 2, 3, seq(27, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(28, length(dfStations), 50))],         
      dfStations[, c(1, 2, 3, seq(29, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(30, length(dfStations), 50))],         
      dfStations[, c(1, 2, 3, seq(31, length(dfStations), 50))],         
      dfStations[, c(1, 2, 3, seq(32, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(33, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(34, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(35, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(36, length(dfStations), 50))],      
      dfStations[, c(1, 2, 3, seq(37, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(38, length(dfStations), 50))],         
      dfStations[, c(1, 2, 3, seq(39, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(40, length(dfStations), 50))],         
      dfStations[, c(1, 2, 3, seq(41, length(dfStations), 50))],         
      dfStations[, c(1, 2, 3, seq(42, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(43, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(44, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(45, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(46, length(dfStations), 50))],      
      dfStations[, c(1, 2, 3, seq(47, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(48, length(dfStations), 50))],         
      dfStations[, c(1, 2, 3, seq(49, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(50, length(dfStations), 50))],         
      dfStations[, c(1, 2, 3, seq(51, length(dfStations), 50))],         
      dfStations[, c(1, 2, 3, seq(52, length(dfStations), 50))],
      dfStations[, c(1, 2, 3, seq(53, length(dfStations), 50))]      
    ),
    use.names = F
  )
  
  colnames(dfStations) <- c('store',
                            'min_date',
                            'max_date',
                            'station', 
                            'name', 
                            'lat', 
                            'lon', 
                            'dist'
  )
  
  dfStations <- dfStations[order(dfStations$store, dfStations$dist), ]
  write.csv(x = dfStations, file = 'nearby_stations.csv', row.names = F)
  
} else {
  dfStations <- read.csv('nearby_stations.csv', stringsAsFactors = F)
  
}

# Function to get the weather data for the stations and timeframes
get_weather_data <- function(dfInput, date1, date2) {
  vLen <- nrow(dfInput)
  lWeather <- list()
  
  for(i in 1:vLen) {
    lWeather[[i]] <- meteo_tidy_ghcnd(
      stationid = dfInput$station[i], 
      date_min = date1[i], 
      date_max = date2[i], 
      var = c('TMAX','TMIN', 'PRCP')
    )
  }
  
  return(lWeather)
}

lWeather <- get_weather_data(
  dfStations, 
  dfStations$min_date, 
  dfStations$max_date
  )

# Function to check if each station has at least some data
check_weather_data <- function(lInput) {
  vMissing <- 0
  
  for (i in 1:length(lInput)) {
    if (i %% 50 == 1) {
      vFound <- F
    }
    if (i %% 50 == 0 && vFound == F) {
      vMissing <- vMissing + 1
    }
    
    if (length(lInput[[i]]) == 5) {
      vFound <- T
    }
  }
  
  return(vMissing)  
  
}

check_weather_data(lWeather)

# Function to flag out the stations that have no data available
clean_weather_data <- function(lInput) {
  
  for (i in 1:length(lInput)) {
    if (length(lInput[[i]]) < 5) {
      lInput[[i]] <- NA
    }
    
  }
  
  return(lInput)  
  
}

lWCleaned <- clean_weather_data(lWeather)

# Function to get how many datapoints each station has
get_data_amount <- function(lInput) {
  vDataPoints <- c()
  
  for (i in 1:length(lInput)) {
    if (length(lInput[[i]]) != 1) {
      vDataPoints <- c(vDataPoints, nrow(lInput[[i]]))
      
    } else {
      vDataPoints <- c(vDataPoints, 0)
      
    }
    
  }
  
  return(vDataPoints)
  
}

dfStations$data_points <- get_data_amount(lWCleaned)

# Function to add store ids to the weather data
set_store_id <- function(lInput, vStore_ids) {
  
  for (i in 1:length(lInput)) {
    if (length(lInput[[i]]) != 1) {
      lInput[[i]]$store <- vStore_ids[i]
      
    }
    
  }  
  
  return(lInput)
  
}

lWCleaned <- set_store_id(lWCleaned, dfStations$store)

# Remove the stations with no data from the weather data
lWCleaned <- lWCleaned[!is.na(lWCleaned)]

# Create a dataframe from the extracted data
dfWeather <- bind_rows(lWCleaned)

# Remove the stations without data from the Station Master dataframe
dfStations <- dfStations[dfStations$data_points != 0, ]

# Merge the Stations data with the weather data
dfWS <- merge(
  dfWeather, 
  dfStations, 
  by.x = c('id', 'store'), 
  by.y = c('station', 'store'), 
  all.y = T
  )

# Remove Factors
dfWS$date <- as.character(dfWS$date)
dfWS$name <- as.character(dfWS$name)
dfWS$lat <- as.numeric(as.character(dfWS$lat))
dfWS$lon <- as.numeric(as.character(dfWS$lon))
dfWS$dist <- as.numeric(as.character(dfWS$dist))

# Calculate the mean values for each available store and date pairs
# and create a Data data frame from the weather, station and sales data
dfData <- merge(
  dfSales, 
  as.data.frame(aggregate(prcp ~ store + date, dfWS, mean)),
  by.x = c('store', 'date'), 
  by.y = c('store', 'date'), 
  all.x = T
  )

dfData <- merge(
  dfData, 
  as.data.frame(aggregate(tmin ~ store + date, dfWS, mean)),
  by.x = c('store', 'date'), 
  by.y = c('store', 'date'), 
  all.x = T
  )

dfData <- merge(
  dfData, 
  as.data.frame(aggregate(tmax ~ store + date, dfWS, mean)),
  by.x = c('store', 'date'), 
  by.y = c('store', 'date'), 
  all.x = T
  )

dfData <- merge(
  dfData, 
  dfStore,
  by.x = 'store', 
  by.y = 'id', 
  all.x = T
  )


### Feature Engineering

# Extract features from date
dfData$date <- as.POSIXct(dfData$date, format = '%Y-%m-%d')
dfData$year <- year(dfData$date)
dfData$month <- month(dfData$date)
dfData$day <- day(dfData$date)
dfData$weekday <- weekdays(dfData$date)
dfData$season <- time2season(dfData$date, out.fmt = 'seasons')
dfData$date <- NULL

# visualize the data on a map to see which holidays need to be added
mapWorld <- getMap(resolution = "low")

plot(
  mapWorld, 
  xlim = c(-25, 60), 
  ylim = c(35, 71), 
  asp = 1
  )

points(dfData$longitude, dfData$latitude, col = "red", cex = .6)

# Mark bank holidays and the previous 2 days
dfData$holiday <- 0

# Christmas
dfData[dfData$month == 12 & dfData$day == 23, 'holiday'] <- 1
dfData[dfData$month == 12 & dfData$day == 24, 'holiday'] <- 1
dfData[dfData$month == 12 & dfData$day == 25, 'holiday'] <- 2
dfData[dfData$month == 12 & dfData$day == 26, 'holiday'] <- 2

# 2017
dfData[dfData$year == 2017 & dfData$month == 08 & dfData$day == 26, 'holiday'] <- 1
dfData[dfData$year == 2017 & dfData$month == 08 & dfData$day == 27, 'holiday'] <- 1
dfData[dfData$year == 2017 & dfData$month == 08 & dfData$day == 28, 'holiday'] <- 2

dfData[dfData$year == 2017 & dfData$month == 05 & dfData$day == 25, 'holiday'] <- 1
dfData[dfData$year == 2017 & dfData$month == 05 & dfData$day == 28, 'holiday'] <- 1
dfData[dfData$year == 2017 & dfData$month == 05 & dfData$day == 29, 'holiday'] <- 2

dfData[dfData$year == 2017 & dfData$month == 04 & dfData$day == 29, 'holiday'] <- 1
dfData[dfData$year == 2017 & dfData$month == 04 & dfData$day == 30, 'holiday'] <- 1
dfData[dfData$year == 2017 & dfData$month == 05 & dfData$day == 01, 'holiday'] <- 2

dfData[dfData$year == 2017 & dfData$month == 04 & dfData$day == 15, 'holiday'] <- 1
dfData[dfData$year == 2017 & dfData$month == 04 & dfData$day == 16, 'holiday'] <- 1
dfData[dfData$year == 2017 & dfData$month == 04 & dfData$day == 17, 'holiday'] <- 2

dfData[dfData$year == 2017 & dfData$month == 04 & dfData$day == 12, 'holiday'] <- 1
dfData[dfData$year == 2017 & dfData$month == 04 & dfData$day == 13, 'holiday'] <- 1
dfData[dfData$year == 2017 & dfData$month == 04 & dfData$day == 14, 'holiday'] <- 2

dfData[dfData$year == 2017 & dfData$month == 04 & dfData$day == 01, 'holiday'] <- 1
dfData[dfData$year == 2017 & dfData$month == 01 & dfData$day == 02, 'holiday'] <- 2


# 2018
dfData[dfData$year == 2018 & dfData$month == 08 & dfData$day == 25, 'holiday'] <- 1
dfData[dfData$year == 2018 & dfData$month == 08 & dfData$day == 26, 'holiday'] <- 1
dfData[dfData$year == 2018 & dfData$month == 08 & dfData$day == 27, 'holiday'] <- 2

dfData[dfData$year == 2018 & dfData$month == 05 & dfData$day == 26, 'holiday'] <- 1
dfData[dfData$year == 2018 & dfData$month == 05 & dfData$day == 27, 'holiday'] <- 1
dfData[dfData$year == 2018 & dfData$month == 05 & dfData$day == 28, 'holiday'] <- 2

dfData[dfData$year == 2018 & dfData$month == 05 & dfData$day == 05, 'holiday'] <- 1
dfData[dfData$year == 2018 & dfData$month == 05 & dfData$day == 06, 'holiday'] <- 1
dfData[dfData$year == 2018 & dfData$month == 05 & dfData$day == 07, 'holiday'] <- 2

dfData[dfData$year == 2018 & dfData$month == 03 & dfData$day == 31, 'holiday'] <- 1
dfData[dfData$year == 2018 & dfData$month == 04 & dfData$day == 01, 'holiday'] <- 1
dfData[dfData$year == 2018 & dfData$month == 04 & dfData$day == 02, 'holiday'] <- 2

dfData[dfData$year == 2018 & dfData$month == 03 & dfData$day == 28, 'holiday'] <- 1
dfData[dfData$year == 2018 & dfData$month == 03 & dfData$day == 29, 'holiday'] <- 1
dfData[dfData$year == 2018 & dfData$month == 03 & dfData$day == 30, 'holiday'] <- 2

dfData[dfData$year == 2017 & dfData$month == 12 & dfData$day == 30, 'holiday'] <- 1
dfData[dfData$year == 2017 & dfData$month == 12 & dfData$day == 31, 'holiday'] <- 1
dfData[dfData$year == 2018 & dfData$month == 01 & dfData$day == 01, 'holiday'] <- 2


# 2019
dfData[dfData$year == 2019 & dfData$month == 08 & dfData$day == 24, 'holiday'] <- 1
dfData[dfData$year == 2019 & dfData$month == 08 & dfData$day == 25, 'holiday'] <- 1
dfData[dfData$year == 2019 & dfData$month == 08 & dfData$day == 26, 'holiday'] <- 2

dfData[dfData$year == 2019 & dfData$month == 05 & dfData$day == 25, 'holiday'] <- 1
dfData[dfData$year == 2019 & dfData$month == 05 & dfData$day == 26, 'holiday'] <- 1
dfData[dfData$year == 2019 & dfData$month == 05 & dfData$day == 27, 'holiday'] <- 2

dfData[dfData$year == 2019 & dfData$month == 05 & dfData$day == 04, 'holiday'] <- 1
dfData[dfData$year == 2019 & dfData$month == 05 & dfData$day == 05, 'holiday'] <- 1
dfData[dfData$year == 2019 & dfData$month == 05 & dfData$day == 06, 'holiday'] <- 2

dfData[dfData$year == 2019 & dfData$month == 04 & dfData$day == 20, 'holiday'] <- 1
dfData[dfData$year == 2019 & dfData$month == 04 & dfData$day == 21, 'holiday'] <- 1
dfData[dfData$year == 2019 & dfData$month == 04 & dfData$day == 22, 'holiday'] <- 2

dfData[dfData$year == 2019 & dfData$month == 04 & dfData$day == 17, 'holiday'] <- 1
dfData[dfData$year == 2019 & dfData$month == 04 & dfData$day == 18, 'holiday'] <- 1
dfData[dfData$year == 2019 & dfData$month == 04 & dfData$day == 19, 'holiday'] <- 2

dfData[dfData$year == 2018 & dfData$month == 12 & dfData$day == 30, 'holiday'] <- 1
dfData[dfData$year == 2018 & dfData$month == 12 & dfData$day == 31, 'holiday'] <- 1
dfData[dfData$year == 2019 & dfData$month == 01 & dfData$day == 01, 'holiday'] <- 2

dfData$holiday <- as.character(dfData$holiday)

# Check to see if the holiday categories are signifcant
# Linear regression
lm_holiday <- lm(qty ~ holiday, data = dfData)

# ANOVA
aov_holiday <- aov(lm_holiday)
summary(aov_holiday)

# Post-hoc Analysis (Tukey test)
tukey_holiday <- TukeyHSD(aov_holiday)
tukey_holiday
plot(tukey_holiday)

# Create a backup for the data with the missing values
write.csv(x = dfData, file = 'data_missing.csv', row.names = F)
dfData <- read.csv('data_missing.csv', stringsAsFactors = F)

# Handling Missing Values
# Imputation is not really an option, and there aren't that many incomplete cases
print(colSums(is.na(dfData)))

# The most sensible option here, is to remove any row that is missing values
dfData <- dfData[complete.cases(dfData), ]

# Select 10 items based on data availability
dfItems <- dfData[, c('qty', 'year', 'item')]
dfItems$qty <- 1
dfItemsData <- aggregate(qty ~ item, dfItems, sum)
dfItems$year <- as.character(dfItems$year)
dfItems$item <- as.character(dfItems$item)

dfItemsYear <- aggregate(qty ~ item + year, dfItems, sum)
dfItemsYear <- dcast(dfItemsYear, item ~ year, sum)
dfItemsYear$total <- dfItemsYear$`2017` + dfItemsYear$`2018` + dfItemsYear$`2019`
dfItemsYear <- dfItemsYear[order(dfItemsYear$total, decreasing = T), ]
dfSelectedItems <- head(dfItemsYear, 10)
vItems <- dfSelectedItems$item

dfData <- dfData[dfData$item %in% vItems, ]

# Create a backup for the data with the items selected
write.csv(x = dfData, file = 'data_selected.csv', row.names = F)


### Exploratory data analysis

# Convert numberical categories to character
dfData <- read.csv('data_selected.csv', stringsAsFactors = F)

dfData$item_category <- as.character(dfData$item_category)
dfData$item <- as.character(dfData$item)
dfData$year <- as.character(dfData$year)
dfData$month <- as.character(dfData$month)
dfData$day <- as.character(dfData$day)
dfData$store <- as.character(dfData$store)

# Histogram to check the distribution of quantities
hist(dfData$qty)

# Visualise the correlation between Sold Quantity and the Weather
corrplot(
  cor(
    dfData[, c('unit_price','qty', 'prcp', 'tmin', 'tmax')]
    ), 
  method = 'number'
  )

# Visualise the total quantity sold by year and item category
dfData %>%
  ggplot(
    aes(x = year, y = qty, fill = item_category)
  ) +
  geom_bar(stat = "identity") +
  ggtitle('Total quantity sold by Year and Item Category') +
  xlab('Year') +
  ylab('Total Quantity Sold')

# Visualise the quantity sold by year on a boxplot
dfData %>%
  ggplot(
    aes(x = year, y = qty)
  ) +
  geom_boxplot(
    outlier.colour = 'black', 
    outlier.shape = 16,
    outlier.size = 2, 
    notch = F
  ) +
  ggtitle('Quantities sold by year') +
  xlab('Year') +
  ylab('Quantity Sold')

# Check if there is a difference between the Years considering Sold Quantity
# Linear regression
lm_year <- lm(qty ~ year, data = dfData)

# ANOVA
aov_year <- aov(lm_year)
summary(aov_year)

# Post-hoc Analysis (Tukey test)
tukey_year <- TukeyHSD(aov_year)
tukey_year
plot(tukey_year)


# Visualise the total quantity sold by Season and item category
dfData %>%
  ggplot(
    aes(x = season, y = qty, fill = item_category)
  ) +
  geom_bar(stat = "identity") +
  ggtitle('Total quantity sold by Season and Item Category') +
  xlab('Season') +
  ylab('Total Quantity Sold')

# Visualise the quantity sold by season on a boxplot
dfData %>%
  ggplot(
    aes(x = season, y = qty)) +
  geom_boxplot(
    outlier.colour = 'black', 
    outlier.shape = 16,
    outlier.size = 2, 
    notch = F
  ) +
  ggtitle('Quantities sold by Season') +
  xlab('Season') +
  ylab('Quantity Sold')
  
# Check if there is a difference between the Seasons considering Sold Quantity
# Linear regression
lm_season <- lm(qty ~ season, data = dfData)

# ANOVA
aov_season <- aov(lm_season)
summary(aov_season)

# Post-hoc Analysis (Tukey test)
tukey_season <- TukeyHSD(aov_season)
tukey_season
plot(tukey_season)


# Visualise the total quantity sold by month and item category
dfData %>%
  ggplot(
    aes(x = month, y = qty, fill = item_category)) +
  geom_bar(stat = "identity") +
  ggtitle('Total quantity sold by Month and Item Category') +
  xlab('Month (1 = January, 12 = December') +
  ylab('Total Quantity Sold')

# Visualise the quantities sold by month on a boxplot
dfData %>%
  ggplot(
    aes(x = month, y = qty)
  ) +
  geom_boxplot(
    outlier.colour = 'black', 
    outlier.shape = 16,
    outlier.size = 2, 
    notch = F
  ) +
  ggtitle('Quantities sold by Month') +
  xlab('Month (1 = January, 12 = December') +
  ylab('Quantities Sold')

# Check if there is a difference between the Months considering Sold Quantity
# Linear regression
lm_month <- lm(qty ~ month, data = dfData)

# ANOVA
aov_month <- aov(lm_month)
summary(aov_month)

# Post-hoc Analysis (Tukey test)
tukey_month <- TukeyHSD(aov_month)
tukey_month
plot(tukey_month)


# Visualise the quantity sold by weekday and item category
dfData %>%
  ggplot(
    aes(x = weekday, y = qty, fill = item_category)
  ) +
  geom_bar(stat = "identity") +
  ggtitle('Total Quantity sold by Weekday and Item Category') +
  xlab('Weekday') +
  ylab('Total Quantity Sold')

# Visualise the quantity sold by weekday on a boxplot
dfData %>%
  ggplot(
    aes(x = weekday, y = qty)
  ) +
  geom_boxplot(
    outlier.colour = 'black', 
    outlier.shape = 16,
    outlier.size = 2, 
    notch = F
  ) +
  ggtitle('Quantities sold by Weekday') +
  xlab('Weekday') +
  ylab('Quantities Sold')

# Check if there is a difference between the item_category considering Sold Quantity
# Linear regression
lm_weekday <- lm(qty ~ weekday, data = dfData)

# ANOVA
aov_weekday <- aov(lm_weekday)
summary(aov_weekday)

# Post-hoc Analysis (Tukey test)
tukey_weekday <- TukeyHSD(aov_weekday)
tukey_weekday
plot(tukey_weekday)


# Visualise the quantity sold by day and item category
dfData %>%
  ggplot(
    aes(x = day, y = qty, fill = item_category)
  ) +
  geom_bar(stat = "identity") +
  ggtitle('Total Quantity sold by Day and Item Category') +
  xlab('Day of Month') +
  ylab('Total Quantity Sold')

# Visualise the quantity sold by day on a boxplot
dfData %>%
  ggplot(
    aes(x = day, y = qty)
  ) +
  geom_boxplot(
    outlier.colour = 'black', 
    outlier.shape = 16,
    outlier.size = 2, 
    notch = F
  ) +
  ggtitle('Quantities sold by Day') +
  xlab('Day of Month') +
  ylab('Quantities Sold')

# Check if there is a difference between the days of month considering Sold Quantity
# Linear regression
lm_day <- lm(qty ~ day, data = dfData)

# ANOVA
aov_day <- aov(lm_day)
summary(aov_day)

# Post-hoc Analysis (Tukey test)
tukey_day <- TukeyHSD(aov_day)
tukey_day
plot(tukey_day)


# Visualise the quantity sold by store and item
dfData %>%
  ggplot(
    aes(x = store, y = qty, fill = item)
  ) +
  geom_bar(stat = "identity") +
  ggtitle('Total quantity sold by Store and Item') +
  xlab('Store') +
  ylab('Total Quantity Sold')

# Check if there is a difference between the item_category considering Sold Quantity
# Linear regression
lm_item <- lm(qty ~ item, data = dfData)

# ANOVA
aov_item <- aov(lm_item)
summary(aov_item)

# Post-hoc Analysis (Tukey test)
tukey_item <- TukeyHSD(aov_item)
tukey_item
plot(tukey_item)

# Visualise the quantity sold by store and item category
dfData %>%
  ggplot(
    aes(x = store, y = qty, fill = item_category)
  ) +
  geom_bar(stat = "identity") +
  ggtitle('Total quantity sold by Store and Item Category') +
  xlab('Store') +
  ylab('Total Quantity Sold')

# Check if there is a difference between the item_category considering Sold Quantity
# Linear regression
lm_item_category <- lm(qty ~ item_category, data = dfData)

# ANOVA
aov_item_category <- aov(lm_item_category)
summary(aov_item_category)

# Post-hoc Analysis (Tukey test)
tukey_item_category <- TukeyHSD(aov_item_category)
tukey_item_category
plot(tukey_item_category)


### Feature selection by importance
dfData <- read.csv('data_selected.csv', stringsAsFactors = F)

dfData$store <- as.factor(dfData$store)
dfData$item <- as.factor(dfData$item)
dfData$item_category <- as.factor(dfData$item_category)
dfData$holiday <- as.factor(dfData$holiday)
dfData$season <- as.factor(dfData$season)
dfData$weekday <- as.factor(dfData$weekday)

# Create the training and test datasets
dfData <- dfData[order(dfData$year, dfData$month, dfData$day), ]

dfTrain <- dfData[dfData$year != 2019, ]
dfTestX <- dfData[dfData$year == 2019, ]
dfTestY <- as.data.frame(x = dfTestX[, 'qty'])
names(dfTestY) <- 'qty'
dfTestX$qty <- NULL

# Train model with the significant variables
# All variables: Test RSE: 193.7
# Top 2 : 421.4
# Top 3 : 163
# Top 4 : 156.9
# Top 5: 168.6
# Top 6: 161.9
# Top 7: 161.1
# Top 8: 182.5

model_rf <- randomForest(
  qty ~ ., 
  data = dfTrain,
  xtest = dfTestX,
  ytest = dfTestY$qty,
  ntree = 25, 
  mtry = as.integer(sqrt(length(dfTestX))),
  replace = F,
  importance = T,
  keep.forest = T,
  do.trace = T
  )

# Plot the errors
plot(model_rf$test$mse)

# Plot Variable importance
varImpPlot(model_rf)

# Filter the dataset based on variable importance of the full model
dfVarImp <- data.frame(model_rf$importance)
dfVarImp <- dfVarImp[order(-dfVarImp$X.IncMSE), ]

# Select the must have variable names
vMust <- c('day', 
           'month', 
           'year', 
           'item', 
           'store',
           'qty'
           )

# Get the n most important variable names
vVars <- rownames(head(dfVarImp, 4))
vVars <- c(vVars, vMust)

# Reload the data for feature selection
dfData <- read.csv('data_selected.csv', stringsAsFactors = F)

dfData$store <- as.factor(dfData$store)
dfData$item <- as.factor(dfData$item)
dfData$item_category <- as.factor(dfData$item_category)
dfData$holiday <- as.factor(dfData$holiday)
dfData$season <- as.factor(dfData$season)
dfData$weekday <- as.factor(dfData$weekday)

# Create the training and test datasets & filter the variables
dfData <- dfData[order(dfData$year, dfData$month, dfData$day), ]
dfData <- dfData[, names(dfData) %in% vVars]

dfTrain <- dfData[dfData$year != 2019, ]
dfTestX <- dfData[dfData$year == 2019, ]
dfTestY <- as.data.frame(x = dfTestX[, 'qty'])
names(dfTestY) <- 'qty'
dfTestX$qty <- NULL

# Evaluate models with different subsets of features
# mtry = 7 - 177.1
# mtry = 6 - 176.9
# mtry = 5 - 168.9
# mtry = 4 - 163.4
# mtry = 3 - 156.9
# mtry = 2 - 184.4

model_rf_sub <- randomForest(
  qty ~ ., 
  data = dfTrain,
  xtest = dfTestX,
  ytest = dfTestY$qty,
  ntree = 25, 
  mtry = 3,
  replace = F,
  importance = T,
  keep.forest = T,
  do.trace = T
  )

# Create a dataframe for the error
dfError <- data.frame(
  Preds = predict(model_rf_sub, dfTestX),
  Test = dfTestY$qty
  )

rmse(dfError$Preds, dfError$Test)
mae(dfError$Preds, dfError$Test)


### Forecast for the next 30 days for each store and item pair

# Function to create a dataframe for the forecast with stores and dates
create_Forecast_Frame <- function(dfInput) {
  
  dfForecastData <- dfInput[, c('store', "max_date")]
  dfForecastData <- unique(dfForecastData)
  dfForecastData$max_date <- as.Date(dfForecastData$max_date)
  dfForecastData$forecast_end <- dfForecastData$max_date + 30
  
  dfForecast <- data.frame(matrix(ncol = 2, nrow = 0))
  colnames(dfForecast) <- c('store', 'date')
  
  for (i in 1:nrow(dfForecastData)) {
    dfTemp <- as.data.frame(seq.Date(
      from = dfForecastData$max_date[i] + 1, 
      to = dfForecastData$forecast_end[i], 
      by = 'day'
      )
    )
    
    dfTemp$store <- dfForecastData$store[i]
    names(dfTemp) <- c('date', 'store')
    
    dfForecast <- rbind(dfForecast, dfTemp)
    
  }
  
  return(dfForecast)
}

dfForecast <- create_Forecast_Frame(dfStations)

# Get the store and item pairs
dfStoreItem <- as.data.frame(unique(dfData[, c('store', 'item', 'longitude')]))

# Merge the two data frames
dfForecast <- merge(
  dfForecast, 
  dfStoreItem,
  by.x = 'store',
  by.y = 'store', 
  all.x = T
  )

# Create the input features needed for the model
dfForecast <- dfForecast[complete.cases(dfForecast), ]
dfForecast$date <- as.POSIXct(dfForecast$date, format = '%Y-%m-%d')
dfForecast$year <- year(dfForecast$date)
dfForecast$month <- month(dfForecast$date)
dfForecast$day <- day(dfForecast$date)
dfForecast$weekday <- weekdays(dfForecast$date)

vDates <- dfForecast$date
dfForecast$date <- NULL

dfForecast$store <- as.factor(dfForecast$store)
dfForecast$weekday <- as.factor(dfForecast$weekday)

# Predict and round the results to the nearest integer
dfForecast$qty <- round(predict(model_rf_sub, dfForecast))

# Create the final results 
dfForecast$date <- vDates
dfForecast$date <- as.character(as.POSIXct(dfForecast$date, format = '%Y-%m-%d'))

dfResults <- as.data.frame(aggregate(qty ~ date + item, dfForecast, sum))
names(dfResults) <- c('Date', 'Item', 'Quantity')

# Save it to a file
write.csv(x = dfResults, file = 'forecasts/result.csv', row.names = F)

# Create a chart and save it
dfResults$Date <- as.Date(dfResults$Date)

dfResults %>%
  ggplot(
    aes(x = Date, y = Quantity, fill = Item)
  ) +
  geom_bar(stat = "identity") +
  ggtitle('Forecasted Quantities sold by Date and Item') +
  xlab('Date') +
  ylab('Forecasted Quantity Sold')

ggsave('forecasts/result.png')
