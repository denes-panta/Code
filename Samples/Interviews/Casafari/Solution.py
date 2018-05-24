#Read In and Data Munging
import pandas as pd
import pandasql as pdsql
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor as RFR
import seaborn as sb
#import nltk

#Data munging
dfData = pd.read_csv("F:\\Code\\Interviews\\Casafari\\assignment_data.csv")
print(dfData.isna().any())
dfData['title'] = dfData['title'].str.lower()
dfData['features'] = dfData['features'].str.lower()
dfData['features'] = dfData['features'].fillna("none")
dfData['living_area'] = dfData['living_area'].fillna(0)
dfData['plot_area'] = dfData['plot_area'].fillna(0)
dfData['total_area'] = dfData['total_area'].fillna(0)

dfData['location'] = dfData['title'].str.extract('(alenquer|quinta da marinha|nagüeles|nagueles|golden mile)', expand = False)
dfData.loc[dfData['location'] == "nagüeles", 'location'] = "nagueles"

dfData['type'] = dfData['title'].str.extract('(apartment|penthouse|duplex|house|villa|country estate|moradia|quinta|plot|land)', expand = False)
dfData['type'] = dfData['type'].fillna("unknown")

dfData.loc[dfData['type'].isin(['apartment', 'penthouse', 'duplex']), 'group'] = "apartments"
dfData.loc[dfData['type'].isin(['house', 'villa', 'country estate', 'moradia', 'quinta']), 'group'] = "houses"
dfData.loc[dfData['type'].isin(['plot', 'land']), 'group'] = "plots"
dfData.loc[dfData['type'].isin(['unknown']), 'group'] = "unknown"

#Experimenting with extracting the individual features
#sList = ' '.join(dfData['features'])
#allWords = nltk.tokenize.word_tokenize(sList)
#allWordDist = nltk.FreqDist(w.lower() for w in allWords)
#
#stopwords = nltk.corpus.stopwords.words('english')
#allWordExceptStopDist = nltk.FreqDist(w.lower() for w in allWords if w not in stopwords)

dfData['pool'] = dfData['features'].str.extract('(pool)', expand = False).notnull().astype(int)
dfData['sea view'] = dfData['features'].str.extract('(sea view)', expand = False).notnull().astype(int)
dfData['garage'] = dfData['features'].str.extract('(garage)', expand = False).notnull().astype(int)
print(dfData.isna().any())

#SQL challange
pysql = lambda q: pdsql.sqldf(q, globals())

#1
q = (
"""
SELECT *
FROM 
    dfData
WHERE 
    type IN ('quinta', 'house');
""")
df1 = pysql(q)

#2
q = (
"""
SELECT *
FROM 
    dfData
WHERE 
    pool = 1;
""")
df2 = pysql(q)

#3
q = (
"""
SELECT *
FROM dfData
WHERE 
    "group" != 'plot';
""")
df3 = pysql(q)

#4 The areas and prices that are zero have also been filtered out due to common sense
q = (
"""
SELECT 
    AVG(price / (CASE 
        WHEN "group" = 'plots' THEN "plot_area"
        WHEN "group" IN ('apartments', 'houses') and total_area > living_area THEN "total_area"
        WHEN "group" IN ('apartments', 'houses') and total_area <= living_area THEN "living_area"
    END)) AS "price_per_sqm"
FROM 
    dfData
WHERE 
    location = 'nagueles' AND 
    type = 'apartment' AND 
    price != 0 AND 
    living_area != 0 AND 
    total_area AND plot_area;
""")
df4 = pysql(q)

#Python Challenge
dfLinReg = dfData.loc[:, ("id", "location", "group", "living_area", "total_area", "plot_area", "pool", "sea view", "garage", "price")]
dfLinReg = dfLinReg.loc[dfLinReg["group"] != "unknown", :]

#Print the reqired file to csv
dfDelivery1 = dfData.loc[:, ("id", "location", "type", "title", "features", "pool", "sea view", "garage")]
dfDelivery1.to_csv("F:\\Code\\Interviews\\Casafari\\delivery1.csv", index = False)

dfLinReg["area"] = np.nan
dfLinReg.loc[dfLinReg["group"] == "plots", "area"] = dfLinReg["plot_area"]
dfLinReg.loc[(dfLinReg["group"].isin(['houses', 'apartments'])) & (dfLinReg["living_area"] >= dfLinReg["total_area"]), "area"] = dfLinReg["living_area"]
dfLinReg.loc[(dfLinReg["group"].isin(['houses', 'apartments'])) & (dfLinReg["living_area"] < dfLinReg["total_area"]), "area"] = dfLinReg["total_area"]

#Area with 0 and price with 0 are removed due to these values not making sense
dfLinReg = dfLinReg.join(pd.get_dummies(dfLinReg["location"], drop_first = True).astype(int)) #alenquer is the baseline
dfLinReg = dfLinReg.join(pd.get_dummies(dfLinReg["group"], drop_first = True).astype(int)) #apartments is the baseline
dfLinReg = dfLinReg.loc[(dfLinReg["area"] != 0), :]

del dfLinReg["living_area"]
del dfLinReg["total_area"]
del dfLinReg["plot_area"]

dfDelivery2 = dfLinReg.copy(deep = True)

del dfLinReg["id"]
del dfLinReg["location"]
del dfLinReg["group"]

dfRanFor = dfLinReg.copy(deep = True)
dfLinReg = dfLinReg.loc[(dfLinReg["price"] != 0), :]

plt.hist(dfLinReg["price"], 100)
plt.show()
dfLinReg["price"] = np.log(dfLinReg["price"])
plt.hist(dfLinReg["price"], 100)
plt.show()

plt.hist(dfLinReg["area"], 100)
plt.show()
dfLinReg["area"] = np.log(dfLinReg["area"])
plt.hist(dfLinReg["area"], 100)
plt.show()

#Overvalues/Undervalued
X = dfLinReg.loc[:, dfLinReg.columns.isin(["garage", "pool", "sea view", "price"]) == False]
y = dfLinReg.loc[:, dfLinReg.columns == "price"]
model_lm = sm.regression.linear_model.OLS(y, X).fit()
print(model_lm.summary())
dfOlList = model_lm.outlier_test()

#Outlier removal
X = X.loc[dfOlList["bonf(p)"] > 0.7, :]
y = y.loc[dfOlList["bonf(p)"] > 0.7, :]
model_lm = sm.regression.linear_model.OLS(y, X).fit()
print(model_lm.summary())

#Predicting if overvalued or undervalues
model_rf = RFR(n_estimators = 1000,
               max_features = 0.8,
               max_depth = 13,
               random_state = 0,
               oob_score = True
               )
model_rf.fit(X, y)
print(model_rf.oob_score_)

#Run prediction
del dfRanFor["pool"]
del dfRanFor["garage"]
del dfRanFor["sea view"]

dfRanFor["price"] = np.log1p(dfRanFor["price"])
dfRanFor["area"] = np.log1p(dfRanFor["area"])
X = dfRanFor.loc[:, dfRanFor.columns != "price"]
y = dfRanFor.loc[:, dfRanFor.columns == "price"]
dfResult = y.reset_index(drop = True)
preds = pd.DataFrame(model_rf.predict(X))
dfResult = dfResult.join(preds)
dfResult.columns = ["price", "preds"]
dfResult = dfResult.apply(np.exp)
dfResult["valuation"] = (dfResult["price"] / dfResult["preds"] - 1) * 100

#Scatterplot
dfResult["valued"] = 0
dfResult.loc[dfResult["valuation"] > 25, "valued"] = 0
dfResult.loc[dfResult["valuation"] < -25, "valued"] = 1
dfResult.loc[(dfResult["valuation"] >= -25) & (dfResult["valuation"] <= 25), "valued"] = 2

f = plt.figure(num=None, figsize=(10, 10), dpi=100, facecolor='w', edgecolor='k')
plt.scatter("valuation", "price", c = "valued", data = dfResult)
plt.title("Over- (Purple), Under- (Green)  and Normally (Yellow) valued ")
plt.xlabel("% Valuation")
plt.ylabel("Price")
plt.show()
f.savefig("F:\\Code\\Interviews\\Casafari\\output.pdf")

#Assumption:
#If the predicted price deviates more than 25% from the current price, it is considered over/under valued.
del dfDelivery2["pool"]
del dfDelivery2["garage"]
del dfDelivery2["sea view"]
dfDelivery2 = dfDelivery2.reset_index(drop = True)
dfDelivery2["over-valued"] = 0
dfDelivery2.loc[dfResult["valuation"] > 25, "over-valued"] = 1
dfDelivery2["under-valued"] = 0
dfDelivery2.loc[dfResult["valuation"] < -25, "under-valued"] = 1
dfDelivery2["normal"] = 0
dfDelivery2.loc[(dfResult["valuation"] >= -25) & (dfResult["valuation"] <= 25), "normal"] = 1
dfDelivery2 = dfDelivery2.loc[:, ("id", "location", "group", "area", "price", "over-valued", "under-valued", "normal")]
dfDelivery2.to_csv("F:\\Code\\Interviews\\Casafari\\delivery2.csv", index = False)

#The difference between an outlier and over/undervalued data, is that outlier were not used in the training of the model.

#Part 3 theoritical:
#Traps
#1. Some of the properties didn't have any given type in the title
#3. Some of the "nagüeles" were "nagueles"
#4. Some of the areas were 0
#5 The distribution was not normal

#Price listing
#I would ran a regression model just as below with the pool, garage, sea-view added:
X = dfLinReg.loc[:, dfLinReg.columns != "price"]
y = dfLinReg.loc[:, dfLinReg.columns == "price"]
model_lm = sm.regression.linear_model.OLS(y, X).fit()
print(model_lm.summary())
dfOlList = model_lm.outlier_test()

#Outlier removal
X = X.loc[dfOlList["bonf(p)"] > 0.7, :]
y = y.loc[dfOlList["bonf(p)"] > 0.7, :]
model_lm = sm.regression.linear_model.OLS(y, X).fit()
print(model_lm.summary())

#According to the regression model, neither the pool, nor sea view nor garage has significant effect on the price

#Missing values, outliers: delete them, as long as I have a big enough data sample
#Based on the picture provided, I would use image classification with CNN before running a model
#https://conferences.oreilly.com/strata/strata-eu/public/schedule/detail/65518
