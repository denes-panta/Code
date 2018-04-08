################################################################################
#Based on Practices from: https://rpubs.com/Ludovicbenistant/Customer_Analysis #
################################################################################

#Functions
quant <- function(df_c, p){
  quantile(df_c, p)
}

#Import data
set.seed(117)

df_cdnow <- read.table("F:/Code/Code/Advanced Analytics/R Samples/Customer Segmentation/CDNOW.txt")

df_cdnow <- as.data.frame(cbind(ID = df_cdnow[, 1], Date = df_cdnow[, 2], USD = df_cdnow[, 4]))
df_cdnow$Date <- as.Date(as.character(df_cdnow$Date), "%Y%m%d")

customer_n <- length(unique(df_cdnow$ID))

#Recency - x100 
#Frequenct - x10
#Monetary - x1

#Nested vs. Independent

#Outliers
boxplot(USD ~ Date, 
        data = df_cdnow, 
        main = "Outliers"
        )

out <- quant(df_cdnow$USD, c(0.90, 0.95, 0.99))
df_cdnow <- df_cdnow[df_cdnow$USD <= out[3], ]

boxplot(USD ~ Date, 
        data = df_cdnow, 
        main = "Outliers"
        )

#Create Customer Data for Analysis
library(plotly)
library(sqldf)

df_cdnow$Recency <- as.numeric(difftime(time1 = max(df_cdnow$Date)+1, time2 = df_cdnow$Date, units = "days"))

df_customer <- sqldf("SELECT ID,
                             MIN(Recency) AS 'Recency',
                             COUNT(ID) AS 'Frequency',
                             AVG(USD) AS 'Monetary'
                      FROM df_cdnow
                      GROUP BY ID"
                     )

hist(df_customer$Recency, breaks = c(100))
hist(df_customer$Frequency, breaks = c(100))
hist(df_customer$Monetary, breaks = c(100))

df_customer_s <- cbind(df_customer$ID, scale(df_customer[, c(-1)]))

cust_kmeans <- kmeans(df_customer_s, 
                      centers = 5,
                      iter.max = 100,
                      nstart = 20,
                      algorithm = "Lloyd"
                      )

df_customer <- cbind(df_customer, Clusters = cust_kmeans$cluster)

cluster_plot <- plot_ly(df_customer, 
                        x = df_customer$Recency, 
                        y = df_customer$Monetary, 
                        z = df_customer$Frequency, 
                        type = "scatter3d", 
                        mode = "markers", 
                        color = df_customer$Clusters
                        )
print(cluster_plot)
