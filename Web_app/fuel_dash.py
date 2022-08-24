#Helper packages

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns
import math
import missingno as msn
import streamlit as st
import joblib

#Modelling packages

import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV




ind_list = []
for i in range(25421):
    ind_list.append(i)

#Importing the data

epa = pd.read_csv("/home/emumen/Documents/Misk-DSI-2022/Predicting-and-Analyzing-Fuel-Efficiency/data/EPAGreen.csv")

Columns = []
for column in epa:
    if epa[column].isna().sum() >= 0.6*26871:
#    if msn.bar(column) == 0:
        Columns.append(column)
    else:
        continue
epa.drop(Columns, axis = 1, inplace = True)
epa.drop(["Cert Region", "Stnd", "Stnd Description", "Underhood ID", "Unnamed: 0", "City_MPG", "Hwy_MPG"], axis = 1, inplace = True)
epa = epa.dropna()

#Correcting a spelling mistake for one of the rows in the column "Fuel"
spell_count = 0
for ind in epa.index:
    if epa["Fuel"][ind] == "Gasoline/Electricty":
        epa.Fuel[epa.Fuel == epa["Fuel"][ind]] = "Gasoline/Electricity"

#Comb CO2

a = 0
b = 0
g = 0
Las = []
for ind in epa.index:
    if isinstance(epa["Comb_CO2"][ind], str) == True and "/" in epa["Comb_CO2"][ind]:
        a = epa["Comb_CO2"][ind][0:3]
        e = int(a)
        b = epa["Comb_CO2"][ind][4:]
        f = int(b)
        epa.Comb_CO2[epa.Comb_CO2 == epa["Comb_CO2"][ind]] = (e + f)/2
for ind in epa.index:
    if isinstance(epa["Comb_CO2"][ind], str) == True:
        g = int(epa["Comb_CO2"][ind])
        epa.Comb_CO2[epa.Comb_CO2 == epa["Comb_CO2"][ind]] = g

for ind in epa.index:
    if isinstance(epa["Comb_CO2"][ind], float) == True:
        updt = round(epa["Comb_CO2"][ind])
        updt1 = int(updt)
        epa.Comb_CO2[epa.Comb_CO2 == epa["Comb_CO2"][ind]] = updt1

for ind in epa.index:
    Las.append(epa["Comb_CO2"][ind])


#for ind in epa.index:
#    if isinstance(epa['Comb_CO2'][ind], str) == True:
#        Las.append(epa['Comb_CO2'][ind])

#epa.info()
#print(Las)


#Cmb_MPG

a1 = 0
b1 = 0
g1 = 0
Las1 = []
for ind in epa.index:
    if isinstance(epa["Cmb_MPG"][ind], str) == True and "/" in epa["Cmb_MPG"][ind][0:2]:
        a1 = epa["Cmb_MPG"][ind][0]
        e1 = int(a1)
        b1 = epa["Cmb_MPG"][ind][2:]
        f1 = int(b1)
        epa.Cmb_MPG[epa.Cmb_MPG == epa["Cmb_MPG"][ind]] = (e1 + f1)/2
    elif isinstance(epa["Cmb_MPG"][ind], str) == True and "/" in epa["Cmb_MPG"][ind]:
        a1 = epa["Cmb_MPG"][ind][0:2]
        e1 = int(a1)
        b1 = epa["Cmb_MPG"][ind][3:]
        f1 = int(b1)
        epa.Cmb_MPG[epa.Cmb_MPG == epa["Cmb_MPG"][ind]] = (e1 + f1)/2
for ind in epa.index:
    if isinstance(epa["Cmb_MPG"][ind], str) == True:
        g1 = int(epa["Cmb_MPG"][ind])
        epa.Cmb_MPG[epa.Cmb_MPG == epa["Cmb_MPG"][ind]] = g1

for ind in epa.index:
    if isinstance(epa["Cmb_MPG"][ind], float) == True:
        updt2 = round(epa["Cmb_MPG"][ind])
        updt3 = int(updt2)
        epa.Cmb_MPG[epa.Cmb_MPG == epa["Cmb_MPG"][ind]] = updt3

for ind in epa.index:
    Las1.append(epa["Cmb_MPG"][ind])


#Dropping rows with the value "Mod" stored in their columns
l12 = []
for ind in epa.index:
    if epa["Air Pollution Score"][ind] == "Mod":
        epa = epa.drop(ind)

#Converting columns in the epa dataframe into their appropriate data type

epa['Comb_CO2'] = epa['Comb_CO2'].astype('int')
epa["Cmb_MPG"] = epa["Cmb_MPG"].astype('int')
#epa["City_MPG"] = epa["City_MPG"].astype('int')
epa["Greenhouse_Score"] = epa["Greenhouse_Score"].astype(int)
epa["Air Pollution Score"] = epa["Air Pollution Score"].astype(int)
epa["Drive"] = epa["Drive"].astype(str)
epa["Trans"] = epa["Trans"].astype(str)
epa["Fuel"] = epa["Fuel"].astype(str)
epa["Veh_Class"] = epa["Veh_Class"].astype(str)
epa["Model"] = epa["Model"].astype(str)
epa["SmartWay"] = epa["SmartWay"].astype(str)


#Splitting the data into training and testing segments

train,test = train_test_split(epa, train_size = 0.8, test_size = 0.2, random_state = 123)
#This is to separate the features or attribute from the target
X_train = train.drop("Cmb_MPG", axis=1)
y_train = train[["Cmb_MPG"]]

# One hot encode remaining nominal features
encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

# Combine into a pre-processing pipeline
preprocessor = ColumnTransformer(
  remainder="passthrough",
  transformers=[
   ("one-hot", encoder, selector(dtype_include="object")),
   ]
  )


New = epa["Displ", "Comb_CO2", "Fuel", "Veh_class"]
New = New.replace(["Gasoline", "Gasoline/Electricity", "Diesel", "Ethanol/Gas", "Ethanol", "CNG/Gasoline", "CNG"], [6, 5, 4, 3, 2, 1, 0])
New = New.replace(["small car", "small SUV", "midsize car", "standard SUV", "large car", "pickup", "station wagon", "special purpose", "minivan", "van"], [15, 14, 13, 12, 11, 10, 9, 8, 7])
Final = epa["Cmb_MPG"]

# GBM Estimator Model Object
pk = xgb.XGBRegressor()
pk.fit(New, Final)
#Loss Function
loss = 'neg_root_mean_squared_error'

#Creating a pipeline
model_pipeline = Pipeline(steps=[
  ("preprocessor", preprocessor),
  ("xgb_mod", pk),
])


kfold = KFold(n_splits=5, random_state=123, shuffle=True)

# Setting the Hyperparameters
hyper_grid = {
  'xgb_mod__n_estimators': [1000, 2500, 5000],
  'xgb_mod__learning_rate': [0.001, 0.01, 0.1],
  'xgb_mod__max_depth': [3, 5, 7, 9],
  'xgb_mod__min_child_weight': [1, 5, 15] 
}

# Implementing a five fold CV
results = cross_val_score(model_pipeline, X_train, y_train, cv=kfold, scoring=loss)

print('Gradient Boosting RMSE: ', np.abs(np.mean(results)), end='\n\n')


# create random search object
random_search = RandomizedSearchCV(
    model_pipeline, 
    param_distributions=hyper_grid, 
    n_iter=20, 
    cv=kfold, 
    scoring=loss, 
    n_jobs=-1, 
    random_state=13
)

# Execute random search
random_search_results = random_search.fit(X_train, y_train)

# Best model score
print('Gradient Boosting random search RMSE: ', np.abs(random_search_results.best_score_), end='\n\n')

# Optimal penalty parameter in grid search
print('Optimal hyperparameter')
print("xgb_mod__n_estimators: ", random_search_results.best_estimator_.get_params().get('xgb_mod__n_estimators'))
print("xgb_mod__learning_rate: ", random_search_results.best_estimator_.get_params().get('xgb_mod__learning_rate'))
print("xgb_mod__max_depth: ", random_search_results.best_estimator_.get_params().get('xgb_mod__max_depth'))
print("xgb_mod__min_child_weight: ", random_search_results.best_estimator_.get_params().get('xgb_mod__min_child_weight'), end="\n\n")


# %%

import joblib
joblib.dump(pk, "pk.pkl")