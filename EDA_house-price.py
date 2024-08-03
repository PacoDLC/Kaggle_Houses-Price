# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:53:26 2024

@author: Francisco De La Cruz
"""

"""*************************************************************************"""
""">>>>>>>>>>>>>>>>>>>>>> EXPLORATORY DATA ANALYSIS <<<<<<<<<<<<<<<<<<<<<<<<"""
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

""">>>>>>>>>>>>>>>> Analysis of target variable "SalePrice" <<<<<<<<<<<<<<<"""

df_train = pd.read_csv('train.csv' , index_col = "Id")

df_train["SalePrice"].describe()

sns.displot(df_train["SalePrice"])

SP_skew = df_train["SalePrice"].skew()
SP_kurt = df_train["SalePrice"].kurt()
print(f'Skewness of SalePrice variable: {SP_skew}')
print(f'Kurtosis of SalePrice variable: {SP_kurt}')

# Visualize relation between "SalePrice" with "GrLivAre" and "TotalBsmtSF" variables

data_01 = pd.concat([df_train["SalePrice"], df_train["GrLivArea"]], axis = 1)
data_01.plot.scatter(x = "GrLivArea", y = "SalePrice")

data_02 = pd.concat([df_train["SalePrice"], df_train["TotalBsmtSF"]], axis = 1)
data_02.plot.scatter(x = "TotalBsmtSF", y = "SalePrice")

# Relationship with categorical features

data_03 = pd.concat([df_train["SalePrice"], df_train["OverallQual"]], axis = 1)
f, ax = plt.subplots(figsize = (8,6))
fig = sns.boxplot(x = "OverallQual", y = "SalePrice", data = data_03)

data_04 = pd.concat([df_train["SalePrice"], df_train["YearBuilt"]], axis = 1)
f, ax = plt.subplots(figsize = (20,8))
fig = sns.boxplot(x = "YearBuilt", y = "SalePrice", data = data_04)
plt.xticks(rotation = 90)

# Elaborate a correlation matrix between the numerical variables involved.

corr_matx = df_train.select_dtypes(include = "number").corr()
f, ax = plt.subplots(figsize = (12,9))
sns.heatmap(corr_matx)

cols = corr_matx.nlargest(10, "SalePrice")["SalePrice"].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale = 1.25)
hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = ".2f",\
                 annot_kws = {"size":10,}, yticklabels = cols.values,\
                     xticklabels = cols.values)
plt.show()

sns.set()
cols = ["SalePrice", "OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF",\
        "FullBath", "YearBuilt"]
sns.pairplot(df_train[cols], size = 2.5)
plt.show()