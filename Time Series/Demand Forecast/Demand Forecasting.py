#Libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pylab import rcParams

sPath = "F:\\Code\\Code\\Time Series\\Demand Forecast\\"
dfData = pd.read_table(sPath + "Demand Forecast.csv", header = 0, sep = ';')

#Data Munging
#Filter out the NaN columns and rows
dfData = dfData.iloc[:, :-2]
dfData = dfData.dropna(thresh = len(dfData.columns))
#Mark the missing values with NaN
dfData = dfData.replace(-200, np.nan)
#Create new Date axis from Date and Time
dfData.index = pd.to_datetime(dfData["Date"] + " " + dfData["Time"])
#Check for missing Time Values
dfData = dfData.drop(columns = ["Date", "Time"], axis = 0)
idx = pd.date_range(min(dfData.index), max(dfData.index), freq='H')
dfData.index = pd.DatetimeIndex(dfData.index)
del idx

dfData["PT08.S1(CO)"] = dfData["PT08.S1(CO)"].interpolate("Time")

#Exploratory analysis
#Get correlation
plt.matshow(dfData.corr())
mCorr = dfData.iloc[:, 2:].corr()

#Get number of NaN per variables
mNaN = len(dfData) - dfData.count()

#Visualisation
rcParams['figure.figsize'] = 15, 10
dfData.loc[:, "CO(GT)"].plot()
plt.title("CO(GT)")
plt.show()
dfData.loc[:, "PT08.S1(CO)"].plot()
plt.title("PT08.S1(CO)")
plt.show()
dfData.loc[:, "NMHC(GT)"].plot()
plt.title("NMHC(GT)")
plt.show()
dfData.loc[:, "C6H6(GT)"].plot()
plt.title("C6H6(GT)")
plt.show()
dfData.loc[:, "PT08.S2(NMHC)"].plot()
plt.title("PT08.S2(NMHC)")
plt.show()
dfData.loc[:, "NOx(GT)"].plot()
plt.title("NOx(GT)")
plt.show()
dfData.loc[:, "PT08.S3(NOx)"].plot()
plt.title("PT08.S3(NOx)")
plt.show()
dfData.loc[:, "NO2(GT)"].plot()
plt.title("NO2(GT)")
plt.show()
dfData.loc[:, "PT08.S4(NO2)"].plot()
plt.title("PT08.S4(NO2)")
plt.show()
dfData.loc[:, "PT08.S5(O3)"].plot()
plt.title("PT08.S5(O3)")
plt.show()
dfData.loc[:, "T"].plot()
plt.title("T")
plt.show()
dfData.loc[:, "RH"].plot()
plt.title("RH")
plt.show()
dfData.loc[:, "AH"].plot()
plt.title("AH")
plt.show()
