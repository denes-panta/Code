# A/B Test with t-test
#
#Sources used:
#http://pragmati.st/2013/02/15/calculating-sample-sizes-for-ab-tests/
#http://20bits.com/article/statistical-analysis-and-ab-testing


#Set Seed
set.seed(117)


#Assumptions:
#Baseline conversion rate: 10%
#New Layout conversion rate: 15%
pb <- 0.1 #conversion rate of baseline layout 
pn <- 0.15 #conversion rate of new layout
N <- 685 #sample size


#Estimate size of Test and Control Group -> 1134.07 ~ 1150 / group
power.prop.test(n = NULL,
                p1 = pb,
                p2 = pn, 
                power = 0.80, 
                alternative = 'two.sided', 
                sig.level = 0.05
                )

#Generate random Groups
base <- rbinom(N, 1, pb)
new <- rbinom(N, 1, pn)


#Test Hypotesis
#H0 = There is no difference between the Baseline and New Layout, pn = pb -> X : pn-pb = 0
#H1 = There is a significant difference between the Baseline and New Layout, pn != pb -> X : pn-pb != 0


#Confidence intervals
binom.test(sum(base), N, pb, alternative = "two.sided")
binom.test(sum(new), N, pn, alternative = "two.sided")


#Z-Score 
z_X = (pn - pb) / sqrt(pb*(1-pb)/N + pn*(1-pn)/N)
print(z_X)


#Conclusion:
#Z-score is 2.80, which is above the defines 1.96 threshold.
#Therefore, we reject the H0, and accept the H1.
#And we can claim with 95% confidence, that the new Layout has a higher conversion rate.
