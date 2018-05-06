#Turn off scientific notations
options(scipen = 999) 

#Import libraries
library(datasets)
library(corrplot)
library(ggplot2)

#Get data and filter out records with NA
df_data <- datasets::airquality
df_data <- df_data[complete.cases(df_data), ]

#Get an idea of the data
str(df_data)
head(df_data)

#Assumptions: We would like to predics the temperaute from the data
#Visualisitation is built based on the above assumption

#Get correlation of variables
corrplot(cor(df_data), "number")

#Factorise the variables for visualisation
df_data$Month <- as.factor(df_data$Month)
df_data$Day <- as.factor(df_data$Day)

#Visualise the data
ggplot(df_data, aes(x = Temp, y = Wind)) +
  geom_point(aes(col = Temp, size = Solar.R)) + 
  geom_smooth(method = glm, se = F, col = "steelblue") + 
  ggtitle("Temperature vs Ozone vs Wind vs Solar Radiation", subtitle = "From the airquality dataset") + 
  xlab("Temperature") + 
  ylab("Ozone levels") +
  scale_color_distiller(palette = "YlOrRd") + 
  scale_x_continuous(breaks = seq(50, 100, 5)) + 
  scale_y_continuous(breaks = seq(0, 25, 1))

ggplot(df_data, aes(x = Month, y = Temp)) + 
  geom_boxplot(fill = "lightblue") + 
  ggtitle("Box-plot: Temperature vs Month", subtitle = "From dataset airquality") + 
  xlab("Month of year") +
  ylab("Temperature")

ggplot(df_data, aes(x = Day, y = Temp)) + 
  geom_boxplot(fill = "lightblue") + 
  ggtitle("Box-plot: Temperature vs Day", subtitle = "From dataset airquality") + 
  xlab("Month of year") +
  ylab("Temperature")

ggplot(df_data, aes(x = Month, y = Day)) +
  geom_tile(aes(fill = Temp)) +
  scale_fill_gradient(low = "red", high = "yellow") + 
  ggtitle("Month vs Day vs Temperature vs missing values", subtitle = "From dataset airquality") +
  xlab("Month") +
  ylab("Day")

ggplot(df_data, aes(x = Month, y = Day)) +
  geom_tile(aes(fill = Ozone)) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") + 
  ggtitle("Month vs Day vs Ozone vs missing values", subtitle = "From dataset airquality") +
  xlab("Month") +
  ylab("Day")

ggplot(df_data, aes(x = Month, y = Day)) +
  geom_tile(aes(fill = Wind)) +
  scale_fill_gradient(low = "lightgreen", high = "darkgreen") + 
  ggtitle("Month vs Day vs wind vs missing values", subtitle = "From dataset airquality") +
  xlab("Month") +
  ylab("Day")
