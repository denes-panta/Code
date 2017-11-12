#Bayesian A/B Test
#Source of implementation: https://cran.r-project.org/web/packages/bayesAB/vignettes/introduction.html

library(bayesAB)
set.seed(117)

#Creating random binomial distributions samples
New_layout <- rbinom(2000, 1, .275) #New Layout hypothetical conversionrate of 27.5%
Baseline <- rbinom(2000, 1, .25) #Baseline hypothetical conversion rate of 25%

#Choosing Prior Distribution
plotBeta(65, 200)

#Running the test
AB1 <- bayesTest(New_layout,
                 Baseline, 
                 priors = c('alpha' = 65, 'beta' = 200), 
                 n_samples = 1e5, 
                 distribution = 'bernoulli'
                 )

AB_s <- summary(AB1)
plot(AB1)

#Conclusion:
#The Probability that the New Layout is better than the Baseline is AB_s$probability.
#If the AB_s$probability is below the acceptable level, the sample sizes of the Test and Control Groups need to be increased.