# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 21:54:32 2024

@author: Francisco De La Cruz
"""

import pandas as pd
import numpy as np
import seaborn as sns
# import matplotlib.pyplot as plt

"""*************************************************************************"""
""">>>>>>>>>>>>>>>>>>>>>>>>>> DATA PREPROCESING <<<<<<<<<<<<<<<<<<<<<<<<<<<<"""
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

train_set = pd.read_csv('train.csv' , index_col = "Id")
test_set = pd.read_csv('test.csv' , index_col = "Id")

print("Full train dataset shape is {}".format(train_set.shape))
print("Full test dataset shape is {}".format(test_set.shape))

descript_train_set = train_set.describe(include = "object").T
descript_test_set = test_set.describe(include = "object").T

train_set.info()
test_set.info()

# OBSERVATION: The array "test_set" doesn't contain the column "SalePrice"

X_train = train_set.iloc[: , :-1]
Y_train = train_set.iloc[: , -1]

X_test = test_set.iloc[: , :]
# Y_test = test_set.iloc[: , -1]

# Checking the number of null values in each column

N_Nulls_train = train_set.isna().sum()
N_Nulls_test = test_set.isna().sum()

# Delete columns with a large number of null values
 
X_train = X_train.drop(["Alley", "MasVnrType", "FireplaceQu", "PoolQC", \
                        "Fence", "MiscFeature"], axis = 1)
X_test = X_test.drop(["Alley", "MasVnrType", "FireplaceQu", "PoolQC", \
                        "Fence", "MiscFeature"], axis = 1)

# Identify the numerical data colums

numerical_data = ["LotFrontage", "LotArea", "YearBuilt", "YearRemodAdd", \
                  "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", \
                  "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", \
                  "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", \
                  "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", \
                  "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF", \
                  "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", \
                  "PoolArea", "MiscVal", "MoSold", "YrSold"]
    
# Identify the categorical data colums

categorical_nom_data = ["MSSubClass", "MSZoning", "Street", "Alley" ,"LotShape", \
                        "LandContour", "Utilities", "LotConfig", "LandSlope",\
                        "Neighborhood", "Condition1", "Condition2", \
                        "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", \
                        "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation", \
                        "Heating", "CentralAir", "Electrical", "Functional", \
                        "GarageType", "PavedDrive", "SaleType"] 
categorical_ord_data = ["OverallQual", "OverallCond", "ExterQual", \
                        "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure", \
                        "BsmtFinType1", "BsmtFinType2", "HeatingQC", \
                        "KitchenQual", "FireplaceQu","GarageFinish", "GarageQual", \
                        "GarageCond", "PoolQC", "Fence", "SaleCondition"]

# NaN treatment

from sklearn.impute import SimpleImputer

imputer_num = SimpleImputer(missing_values = np.nan, strategy = "mean")

for i in numerical_data:
    j = X_train.columns.get_loc(i)
    imputer_num = imputer_num.fit(X_train.iloc[:, j].values.reshape(-1, 1))
    X_train.iloc[:,j] = imputer_num.transform(X_train.iloc[:,j].values.\
                                              reshape(-1,1))
    j = X_test.columns.get_loc(i)
    imputer_num = imputer_num.fit(X_test.iloc[:, j].values.reshape(-1, 1))
    X_test.iloc[:,j] = imputer_num.transform(X_test.iloc[:,j].values.\
                                              reshape(-1,1))

imputer_nom = SimpleImputer(missing_values = np.nan, strategy = "most_frequent")

for m in categorical_nom_data:
    n = X_train.columns.get_loc(m)
    imputer_nom = imputer_nom.fit(X_train.iloc[:, n].values.reshape(-1, 1))
    X_train.iloc[:,n] = imputer_nom.transform(X_train.iloc[:,n].values.\
                                              reshape(-1,1))
    n = X_test.columns.get_loc(m)
    imputer_nom = imputer_nom.fit(X_test.iloc[:, n].values.reshape(-1, 1))
    X_test.iloc[:,n] = imputer_nom.transform(X_test.iloc[:,n].values.\
                                              reshape(-1,1))
        
imputer_ord = SimpleImputer(missing_values = np.nan, strategy = "most_frequent")

for p in categorical_ord_data:
    q = X_train.columns.get_loc(p)
    imputer_ord = imputer_ord.fit(X_train.iloc[:, q].values.reshape(-1, 1))
    X_train.iloc[:,q] = imputer_ord.transform(X_train.iloc[:,q].values.\
                                              reshape(-1,1))
for p in categorical_ord_data:
    if p not in X_test:
        continue
    q = X_test.columns.get_loc(p)
    imputer_ord = imputer_ord.fit(X_test.iloc[:, q].values.reshape(-1, 1))
    X_test.iloc[:,q] = imputer_ord.transform(X_test.iloc[:,q].values.\
                                              reshape(-1,1))
        
N_Nulls_train = X_train.isna().sum()
N_Nulls_test = X_test.isna().sum()

# Checking correlations for numerical data

correlation_num_data = train_set.select_dtypes(include = "number").corr()
sns.heatmap(correlation_num_data)

# Encoding ordinal and nominal catogorical data

from sklearn import preprocessing

labelencoder = preprocessing.LabelEncoder()

for m in categorical_nom_data:
    n = X_train.columns.get_loc(m)
    X_train.iloc[:,n] = labelencoder.fit_transform(X_train.iloc[:,n])
    X_test.iloc[:,n] = labelencoder.fit_transform(X_test.iloc[:,n])
    
for p in categorical_ord_data:
    q = X_train.columns.get_loc(p)
    X_train.iloc[:,q] = labelencoder.fit_transform(X_train.iloc[:,q])
    if p not in X_test:
        continue
    X_test.iloc[:,q] = labelencoder.fit_transform(X_test.iloc[:,q])
    
# Dummyfication of nominal categorical data

categorical_nom_index = []

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

for m in categorical_nom_data:
    categorical_nom_index.append(X_train.columns.get_loc(m))

# print(categorical_nom_index)
    
columntransformer = ColumnTransformer([('one_hot_encoder', \
                                         OneHotEncoder(categories = 'auto'),\
                                         categorical_nom_index)],\
                                         remainder = 'passthrough')

# X_train = np.array(columntransformer.fit_transform(X_train), dtype = float)
# X_test = np.array(columntransformer.fit_transform(X_test), dtype = float)

''' 
Let's use the Training Sets X_train and Y_train to split them into Training 
and Testing Subsets in order to test the Machine Learning models below.
'''

from sklearn.model_selection import train_test_split

X_subtrain, X_subtest, Y_subtrain, Y_subtest = train_test_split(
    X_train, Y_train, test_size = 0.2, random_state = 0)

"""*************************************************************************"""
""">>>>>>>>>>>>>>>>>>>>>>>>>> REGRESSION MODELS <<<<<<<<<<<<<<<<<<<<<<<<<<<<"""
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

'''********************* Multiple Linear Regression ************************'''

# Fit the Multiple Linear Regression model with the Training Subset

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

regressor_MLR = LinearRegression()
regressor_MLR.fit(X_subtrain, Y_subtrain)

Y_pred_MLR = regressor_MLR.predict(X_subtest)

mse = mean_squared_error(Y_subtest, Y_pred_MLR)
mae = mean_absolute_error(Y_subtest, Y_pred_MLR)
r2 = r2_score(Y_subtest, Y_pred_MLR)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R^2 Score: {r2}')


