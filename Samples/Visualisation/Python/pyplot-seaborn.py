from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

#Import data and create dataframe
df_data = pd.DataFrame(datasets.load_iris().data)
df_data.columns = datasets.load_iris().feature_names

df_data["class"] = datasets.load_iris().target

#Check the distribution of the variables
plt.hist("sepal length (cm)", data = df_data)
plt.title("Histogram - Sepal length (cm)")
plt.xlabel("Sepal length (cm)")
plt.ylabel("Count")
plt.show()

plt.hist("sepal width (cm)", data = df_data)
plt.title("Histogram - Sepal width (cm)")
plt.xlabel("Sepal width (cm)")
plt.ylabel("Count")
plt.show()

plt.hist("petal length (cm)", data = df_data)
plt.title("Histogram - Petal length (cm)")
plt.xlabel("Petal length (cm)")
plt.ylabel("Count")
plt.show()

plt.hist("petal width (cm)", data = df_data)
plt.title("Histogram - Petal width (cm)")
plt.xlabel("Petal width (cm)")
plt.ylabel("Count")
plt.show()

#Correclation plot
df_data.corr()

#Plot the correlated variables
plt.scatter("sepal length (cm)", "sepal width (cm)", c = "class", data = df_data)
plt.title("Sepal length vs Sepal width vs Class")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.show()

plt.scatter("petal length (cm)", "petal width (cm)", c = "class", data = df_data)
plt.title("Petal length vs Petal width vs Class")
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.show()

#predictor variables by classes
sb.swarmplot(x = "class", y = "sepal length (cm)", data = df_data)
sb.boxplot(x = "class", y = "sepal length (cm)", data = df_data)

sb.swarmplot(x = "class", y = "sepal width (cm)", data = df_data)
sb.boxplot(x = "class", y = "sepal width (cm)", data = df_data)

sb.swarmplot(x = "class", y = "petal length (cm)", data = df_data)
sb.boxplot(x = "class", y = "petal length (cm)", data = df_data)

sb.swarmplot(x = "class", y = "petal width (cm)", data = df_data)
sb.boxplot(x = "class", y = "petal width (cm)", data = df_data)
