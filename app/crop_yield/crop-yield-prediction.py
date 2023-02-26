"""
import numpy as np
import pandas as pd

Gathering and Cleaning CROP YIELDS DATA


df_yield = pd.read_csv('yield.csv')
df_yield.shape
# df_yield.head()
df_yield.tail(10)

# rename columns
df_yield = df_yield.rename(index=str, columns={"Value": "hg/ha_yield"})
# df_yield.head()

# drop unwanted columns for this model
df_yield = df_yield.drop(['Year Code', 'Element Code', 'Element', 'Year Code', 'Area Code', 'Domain Code', 'Domain', 'Unit', 'Item Code'], axis=1)
# df_yield.head()

# df_yield.describe()
# df_yield.info()

"""
# Gathering and Cleaning RAINFALL DATA
"""

df_rain = pd.read_csv('rainfall.csv')
df_rain.shape
# df_rain.head()
df_rain.tail()

# rename columns
df_rain = df_rain.rename(index=str, columns={" Area": 'Area'})

# Checking for datatypes
# df_rain.info()

# Convert average_rain_fall_mm_per_year from object to float
df_rain['average_rain_fall_mm_per_year'] = pd.to_numeric(df_rain['average_rain_fall_mm_per_year'], errors = 'coerce')
# df_rain.info()

# Dropping empty rows from dataset
df_rain = df_rain.dropna()
# df_rain.describe()

# Merge Yield dataframe with rain dataframe by year and area columns
yield_df = pd.merge(df_yield, df_rain, on=['Year', 'Area'])

# yield_df.head()
# yield_df.describe()


# Gathering and Cleaning PESTICIDES DATA

df_pes = pd.read_csv('pesticides.csv')
# df_pes.head()

# rename columns
df_pes = df_pes.rename(index=str, columns = {"Value": "pesticides_tonnes"})

# drop irrelevant columns
df_pes = df_pes.drop(['Element', 'Domain', 'Unit', 'Item'], axis=1)
# df_pes.head()
# df_pes.describe()
# df_pes.info()

# Merge Pesticides dataframe with yield dataframe
yield_df = pd.merge(yield_df, df_pes, on=['Year', 'Area'])
yield_df.shape
# yield_df.head()


# Gathering and Cleaning AVERAGE TEMPERATURE


df_avg_temp = pd.read_csv("temp.csv")
# df_avg_temp.head()
# df_avg_temp.describe()

# rename columns
df_avg_temp = df_avg_temp.rename(index=str, columns={"year": "Year", "country": 'Area'})
# df_avg_temp.head()

# Merge Average Temperature dataframe with yield dataframe
yield_df = pd.merge(yield_df, df_avg_temp, on=['Area', 'Year'])
# yield_df.head()

yield_df.shape
# yield_df.describe()
yield_df.isnull().sum()


# Data Exploration for yield_df

# yield_df is the final obtained dataframe to use in Data Exploration and Training

yield_df.groupby('Item').count()
# yield_df.describe()

# count number uniques
yield_df['Area'].nunique()

# ordering by highest yield production
yield_df.groupby(['Area'], sort=True)['hg/ha_yield'].sum().nlargest(10)

# group items by item and area
yield_df.groupby(['Item', 'Area'], sort=True)['hg/ha_yield'].sum().nlargest(10)

# visualizing correlation matrix as a heatmap to check correlations among columns

import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

correlation = yield_df.select_dtypes(include=[np.number]).corr()

mask = np.zeros_like(correlation, dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

# custom diverging colormap
cmap = sns.palette = "vlag"

# draw heatmap with mask and aspect ratio
sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=.3, center=0, square = True, linewidths=.5, cbar_kws={"shrink": .5})

# Data Preprocessing - converting raw data into clean data set


# yield_df.head()

# encoding categorical variables using one hot encoding

from sklearn.preprocessing import OneHotEncoder

yield_df_onehot = pd.get_dummies(yield_df, columns=['Area',"Item"], prefix = ['Country',"Item"])
features=yield_df_onehot.loc[:, yield_df_onehot.columns != 'hg/ha_yield']
label=yield_df['hg/ha_yield']
# features.head()

features = features.drop(['Year'], axis=1)
# features.info()
# features.head()
# print(features.col)
# quit()

# scaling featuress - bringing all features to the same level of magnitudes

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
features=scaler.fit_transform(features) 


# Training the Data

from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(features, label, test_size=0.25, random_state=42)

# write final df to csv file

yield_df.to_csv('yield_df.csv')

# Decision Tree Regressor Model

from sklearn.tree import DecisionTreeRegressor
clf=DecisionTreeRegressor()
model=clf.fit(train_data,train_labels)
"""
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split
yield_df = pd.read_csv('yield_df.csv')
features = yield_df[['average_rain_fall_mm_per_year','pesticides_tonnes','avg_temp']]
label = yield_df[['hg/ha_yield']]
print(label.head())
train_data, test_data, train_labels, test_labels = train_test_split(features, label, test_size=0.25, random_state=42)
clf=DecisionTreeRegressor()
model=clf.fit(train_data,train_labels)

# fig, ax = plt.subplots() 
# test_data["yield_predicted"]= model.predict(test_data)
# test_data["yield_actual"]=pd.DataFrame(test_labels)["hg/ha_yield"].tolist()
# ax.scatter(test_data["yield_actual"], test_data["yield_predicted"],edgecolors=(0, 0, 0))

# ax.set_xlabel('Actual')
# ax.set_ylabel('Predicted')
# ax.set_title("Actual vs Predicted")
# plt.show()

filename = "crop_yield_prediction.pickle"

# save model

pickle.dump(model, open(filename, "wb"))

# load model
model = pickle.load(open(filename, "rb"))

features = []



# ['average_rain_fall_mm_per_year', 'pesticides_tonnes' , 'avg_temp', Country_Albania, Country_Algeria, Country_Angola, Country_Argentina, Country_Armenia, Country_Australia, Country_Austria, Item_Cassava, Item_Maize, Item_Plantains and others, Item_Potatoes, Item_Rice, paddy, Item_Sorghum, Item_Soybeans, Item_Sweet, potatoes, Item_Wheat, Item_Yams]

# test_data["yield_predicted"]= model.predict(test_data)

entry = pd.DataFrame([[1485, 121, 16.37]],columns=['average_rain_fall_mm_per_year','pesticides_tonnes','avg_temp'])
result = model.predict(entry)
print(result)



