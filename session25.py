# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 21:33:43 2018

@author: avatash.rathore
"""

# Core Libraries to load (for data manipulation and analysis)
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Import stats module from scipy for Statistical analysis
import scipy.stats as stats

# Core Libraries to load - Machine Learning
import sklearn

## Import LinearRegression Module - Modelling
from sklearn.linear_model import LinearRegression

## Import RandomForestRegressor Module - Modelling
from sklearn.ensemble import RandomForestRegressor

## Import StandardScaler - Model Data Preprocessing
from sklearn.preprocessing import StandardScaler

## Import train_test_split Module
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, cross_val_score,KFold 

## Importing mean_squared_error and r2_score from sklearn.metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

## Import boston dataset from sklearn.datasets
from sklearn.datasets import load_boston

### Load Boston dataset into a variable    
boston = load_boston()

print("\n LETS GET A GRASP OF DATASET\n\n")
# Get keys of boston dataset dictionary
print(boston.keys())
# Get the number of rows and columns in the dataset
boston.data.shape
# Get the column names in the dataset
print(boston.feature_names)
# Get description of the column names in the dataset
print(boston.DESCR)
print(" Create dataframe from boston.data and target information in variables")

Data = pd.DataFrame(boston.data, columns=boston.feature_names)
targets = boston.target
print(Data.head())
print("Get datset info\n\n")
print(Data.info())
# Getting basic statistical information about the columns
print(Data.describe()) # Only numerical columns)

print("\n Plotting the histograms of numerical columns to understand their distribution\n\n")
Data.hist(bins=50, figsize=(10,10), layout=(8,2)) 
plt.show()
print(" Checking for correlations using HEATMAP\n\n")
plt.figure(figsize=(20,20))
sns.heatmap(Data.corr(), cmap="PRGn", annot= True)

print("\n")
X = Data
Y = targets

# We will be using 80:20 split for train and test datasets
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.20, random_state = 100)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
print("\n CREATE A MODEL")
lm = LinearRegression()
model = lm.fit(x_train, y_train) # Sklearn already considers the intercepts for linear regression
print("Estimated Beta Coefficients: \n", model.coef_)

y_test_pred = model.predict(x_test)

print("\nLinear Regression - Base", "\n\t R2-Score:", model.score(x_test, y_test),
                         "\n\t RMSE:", math.sqrt(mean_squared_error(y_test_pred, y_test)),"\n")

rf_reg = RandomForestRegressor(n_estimators=100)
rf_model= rf_reg.fit(x_train, y_train)
y_test_pred = rf_model.predict(x_test)

print("## RandomForest Regressor - Unscaled Data", "\n\t R2-Score:", rf_model.score(x_test, y_test),
                         "\n\t RMSE:", math.sqrt(mean_squared_error(y_test_pred, y_test)),"\n")


print("Feature Selection ")
importance = pd.DataFrame.from_dict({'cols':x_train.columns, 'importance': rf_reg.feature_importances_})
importance = importance.sort_values(by='importance', ascending=False)
plt.figure(figsize=(20,15))
sns.barplot(importance.cols, importance.importance)
plt.xticks(rotation=90)
print("\n Lets use only important columns for Model")
imp_cols = importance[importance.importance >= 0.01].cols.values
imp_cols
# Fitting models with columns where feature importance>=0.01

x_train, x_test, y_train, y_test = train_test_split(X[imp_cols],Y,test_size=0.20, random_state = 100)

rf_reg = RandomForestRegressor(n_estimators=100)
rf_model= rf_reg.fit(x_train, y_train)
y_test_pred = rf_model.predict(x_test)

print("## RandomForest Regressor - Unscaled Data", "\n\t R2-Score:", rf_model.score(x_test, y_test),
                         "\n\t RMSE:", math.sqrt(mean_squared_error(y_test_pred, y_test)),"\n")

imp_cols = importance[importance.importance >= 0.002].cols.values
imp_cols
# Fitting models with columns where feature importance>=0.005

x_train, x_test, y_train, y_test = train_test_split(X[imp_cols],Y,test_size=0.20, random_state = 100)

rf_reg = RandomForestRegressor(n_estimators=100)
rf_model= rf_reg.fit(x_train, y_train)
y_test_pred = rf_model.predict(x_test)

print("## RandomForest Regressor - Unscaled Data", "\n\t R2-Score:", rf_model.score(x_test, y_test),
                         "\n\t RMSE:", math.sqrt(mean_squared_error(y_test_pred, y_test)),"\n")


# Cross validating the model created with columns whose feature importances >= 0.002, with k-fold(10-fold) cross validation
scoring = 'neg_mean_squared_error'
kfold = KFold(n_splits=10, random_state=100)

cv_results = cross_val_score(model, x_train,y_train, cv=kfold, scoring=scoring)
print("## Linear Regression","\n\t CV-Mean:", cv_results.mean(),
                                                 "\n\t CV-Std. Dev:",  cv_results.std(),"\n")

cv_results = cross_val_score(rf_model, x_train,y_train, cv=kfold, scoring=scoring)
print("## RandomForestRegressor - Unscaled data","\n\t CV-Mean:", cv_results.mean(),
                                                 "\n\t CV-Std. Dev:",  cv_results.std(),"\n")

cv_results = cross_val_score(rf_model_scaled, x_scaled_train,y_scaled_train, cv=kfold, scoring=scoring)
print("## RandomForestRegressor - Scaled data","\n\t CV-Mean:", cv_results.mean(),
                                                 "\n\t CV-Std. Dev:",  cv_results.std(),"\n")

print("\n\n Model Tuning")
RF_Regressor =  RandomForestRegressor(n_estimators=100, n_jobs = -1, random_state = 100)

CV = ShuffleSplit(test_size=0.20, random_state=100)

param_grid = {"max_depth": [5, None],
              "n_estimators": [50, 100, 150, 200],
              "min_samples_split": [2, 4, 5],
              "min_samples_leaf": [2, 4, 6]
             }

print("\n\n")
rscv_grid1 = GridSearchCV(RF_Regressor, param_grid=param_grid, verbose=1)
rscv_grid1.fit(x_train, y_train)
rscv_grid1.best_params_
# Best Estimator - Unscaled
rf_model = rscv_grid1.best_estimator_
rf_model.fit(x_train, y_train)
rf_model.score(x_test, y_test)
