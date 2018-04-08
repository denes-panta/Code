############################################################
#Based on Practices from: 'R Data Mining Blueprints (2016)'#
############################################################

set.seed(117)

#Functions
quant <- function(x){
  quantile(x, probs = c(0.95, 0.90, 0.99))
}

#Data Import and preprocessing
df_wholesale <- read.csv("F:/Code/Code/Advanced Analytics/R Samples/Customer Segmentation/Wholesalecustomers.csv")

par(mfrow = c(2,3))
apply(df_wholesale[, -c(1, 2)], 2, FUN = boxplot)

outliers <- sapply(df_wholesale[, c(-1, -2)], FUN = quant)

#Outlier Removal
df_wholesale$Fresh <- ifelse(df_wholesale$Fresh >= outliers[2, 1], 
                             outliers[2, 1], 
                             df_wholesale$Fresh
                             )

df_wholesale$Milk <- ifelse(df_wholesale$Milk >= outliers[2, 2], 
                            outliers[2, 2], 
                            df_wholesale$Milk
                            )

df_wholesale$Grocery <- ifelse(df_wholesale$Grocery >= outliers[2, 3], 
                               outliers[2, 3], 
                               df_wholesale$Grocery
                               )

df_wholesale$Frozen <- ifelse(df_wholesale$Frozen >= outliers[2, 4],
                               outliers[2, 4],
                               df_wholesale$Frozen
                               )

df_wholesale$Detergents_Paper <- ifelse(df_wholesale$Detergents_Paper >= outliers[2, 5], 
                                        outliers[2, 5], 
                                        df_wholesale$Detergents_Paper
                                        )

df_wholesale$Delicassen <- ifelse(df_wholesale$Delicassen >= outliers[2, 6], 
                                  outliers[2, 6], 
                                  df_wholesale$Delicassen
                                  )

apply(df_wholesale[, -c(1, 2)], 2, FUN = boxplot)

#Scaling
df_wholesale_scaled <- as.matrix(scale(df_wholesale[, -c(1, 2)]))


### K-Means ###

#Choosing Cluster number
library(Rcmdr)

sumsq <- NULL

par(mfrow=c(1,2))

#R Commander
for(i in 1:15) {
  sumsq[i] <- sum(KMeans(df_wholesale_scaled,
                         centers = i,
                         iter.max = 500, 
                         num.seeds = 50
                         )$withinss
                  )
}

plot(1:15, sumsq, type = "b", 
     xlab = "Number of Clusters", 
     ylab = "Within groups sum of squares",
     main = "Screeplot using Rcmdr"
     )

#Stats
for (i in 1:15) {
  sumsq[i] <- sum(kmeans(df_wholesale_scaled,
                         centers = i,
                         iter.max = 5000,
                         algorithm = "Forgy"
                         )$withinss
                  )
}

plot(1:15, sumsq, type = "b", 
     xlab = "Number of Clusters", 
     ylab = "Within groups sum of squares",
     main = "Screeplot using Stats"
     )

#Elbow rule: 4 clusters

#Clustering
ws_kmeans <- kmeans(df_wholesale_scaled, 
                    centers = 4, 
                    iter.max = 20, 
                    nstart = 10, 
                    algorithm = c("Hartigan-Wong"), 
                    trace = F
                    )

#Attach Cluster column
df_clustered <- cbind(df_wholesale, Membership = ws_kmeans$cluster)


### Hierarchical ###
library(cluster)

par(mfrow = c(1,1))

#Agglomerative
ws_hclust <- hclust(dist(df_wholesale_scaled, method = "euclidean"), method = "ward.D2")
plot(ws_hclust,hang = -0.005, cex = 0.7)

ws_hclust_c <- cutree(ws_hclust, k = 4)
plot(ws_hclust)
rect.hclust(ws_hclust,
            k = 4, 
            border = "red"
            )

#Divisive
ws_dclust <- diana(df_wholesale_scaled, 
                   diss = F, 
                   metric = "euclidean", 
                   stand = T,
                   keep.data = T
                   )

ws_dclust_c <- cutree(ws_dclust, k = 4)
plot(ws_dclust)


### Model-based Clustering ###

#Mclust
library(mclust)

ws_mclust <- Mclust(df_wholesale_scaled[, -c(1, 2)])
summary(ws_mclust)
plot(ws_mclust)(1)


### Self Organising Map ###
library(kohonen)

som_grid <- somgrid(xdim = 20, 
                    ydim = 20, 
                    topo = "hexagonal"
                    )

som_model <- som(df_wholesale_scaled,
                 grid = som_grid,
                 rlen = 500,
                 alpha = c(0.05, 0.01),
                 keep.data = T
                 )

plot(som_model, type = "changes", col = "blue")
plot(som_model, type = "count")
plot(som_model, type = "dist.neighbours")
plot(som_model, type = "codes")
