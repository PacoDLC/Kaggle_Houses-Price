# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 18:36:10 2024

@author: Francisco De La Cruz

Descritpion: This Python script is a complement of the 'houses-price_v01.py' 
script. Specifically, this one takes the both 'train.csv' and 'test.csv' 
datasets in order to a the XGBoost regression model with the 'train.csv' set 
to predict the 'test.csv' set. 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

"""*************************************************************************"""
""">>>>>>>>>>>>>>>>>>>> ANALYSIS AND DATA PREPROSECING <<<<<<<<<<<<<<<<<<<<<"""
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

dataset_train = pd.read_csv('train.csv', index_col = "Id")
dataset_test = pd.read_csv('test.csv', index_col = "Id")

print("Full train dataset shape is {}".format(dataset_train.shape))
print("Full test dataset shape is {}".format(dataset_test.shape))

descript_dataset_train = dataset_train.describe(include = "object").T
descript_dataset_test = dataset_test.describe(include = "object").T

dataset_train.info()
dataset_test.info()

# Identify numerical and categorical data according to data_description.txt file

# Identify the numerical data colums

numerical_data = ["MSSubClass", "LotFrontage", "LotArea", "OverallQual", "OverallCond", \
                  "YearBuilt", "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", \
                  "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", \
                  "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", \
                  "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", \
                  "KitchenAbvGr", "TotRmsAbvGrd", "GarageYrBlt", \
                  "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", \
                  "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", \
                  "MiscVal", "MoSold", "YrSold"]
    
# Identify the categorical data colums

categorical_nom_data = ["MSZoning", "Street", "Alley" ,"LotShape", \
                        "LandContour", "Utilities", "LotConfig", "LandSlope",\
                        "Neighborhood", "Condition1", "Condition2", \
                        "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", \
                        "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation", \
                        "Heating", "CentralAir", "Electrical", "Functional", \
                        "GarageType", "PavedDrive", "SaleType"] 
categorical_ord_data = ["ExterQual", \
                        "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure", \
                        "BsmtFinType1", "BsmtFinType2", "HeatingQC", \
                        "KitchenQual", "FireplaceQu","GarageFinish", "GarageQual", \
                        "GarageCond", "PoolQC", "Fence", "SaleCondition"]

"""***************************** NaN Treatment *****************************"""

# Checking the general number of Null values in each column

Gen_N_Nulls_train = dataset_train.isna().sum()
Gen_N_Nulls_test = dataset_test.isna().sum()

""" 
OBSERVATIONS: Variables such as 'Alley', 'MasVnrType', 'FireplaceQu', 
'PoolQC', 'Fence', 'MiscFeature' has a huge number of Null values. Let's 
apply a special NaN Treatment for these variables.
"""

dataset_train = dataset_train.drop(["MiscFeature"], axis = 1)
dataset_test = dataset_test.drop(["MiscFeature"], axis = 1)

print("Full dataset train shape is {}".format(dataset_train.shape))
print("Full dataset test shape is {}".format(dataset_test.shape))

# NaN Treatment on the variables mentioned above

from sklearn.impute import SimpleImputer

imputer_spcl = SimpleImputer(missing_values = np.nan, strategy = "constant" \
                              , fill_value = "Miss")

Alley_Col_train = dataset_train.columns.get_loc("Alley")

imputer_Alley = imputer_spcl.fit(dataset_train.iloc[:, Alley_Col_train].values.reshape(-1, 1))

dataset_train.iloc[:, Alley_Col_train] = imputer_Alley.transform(dataset_train.iloc[:,Alley_Col_train].values.\
                                          reshape(-1,1))

Alley_NaN_train = dataset_train["Alley"].isna().sum()

Alley_Col_test = dataset_test.columns.get_loc("Alley")

imputer_Alley = imputer_spcl.fit(dataset_test.iloc[:, Alley_Col_test].values.reshape(-1, 1))

dataset_test.iloc[:, Alley_Col_test] = imputer_Alley.transform(dataset_test.iloc[:,Alley_Col_test].values.\
                                          reshape(-1,1))

Alley_NaN_test = dataset_test["Alley"].isna().sum()

MasVnrType_Col_train = dataset_train.columns.get_loc("MasVnrType")

imputer_spcl = imputer_spcl.fit(dataset_train.iloc[:, MasVnrType_Col_train].values.reshape(-1, 1))

dataset_train.iloc[:, MasVnrType_Col_train] = imputer_spcl.transform(dataset_train.iloc[:, MasVnrType_Col_train].values.\
                                          reshape(-1,1))

MasVnrType_NaN_train = dataset_train["MasVnrType"].isna().sum()

MasVnrType_Col_test = dataset_test.columns.get_loc("MasVnrType")

imputer_spcl = imputer_spcl.fit(dataset_test.iloc[:, MasVnrType_Col_test].values.reshape(-1, 1))

dataset_test.iloc[:, MasVnrType_Col_test] = imputer_spcl.transform(dataset_test.iloc[:, MasVnrType_Col_test].values.\
                                          reshape(-1,1))

MasVnrType_NaN_test = dataset_test["MasVnrType"].isna().sum()

FireplaceQu_Col_train = dataset_train.columns.get_loc("FireplaceQu")

imputer_spcl = imputer_spcl.fit(dataset_train.iloc[:, FireplaceQu_Col_train].values.reshape(-1, 1))

dataset_train.iloc[:, FireplaceQu_Col_train] = imputer_spcl.transform(dataset_train.iloc[:, FireplaceQu_Col_train].values.\
                                          reshape(-1,1))

FireplaceQu_NaN_train = dataset_train["FireplaceQu"].isna().sum()

FireplaceQu_Col_test = dataset_test.columns.get_loc("FireplaceQu")

imputer_spcl = imputer_spcl.fit(dataset_test.iloc[:, FireplaceQu_Col_test].values.reshape(-1, 1))

dataset_test.iloc[:, FireplaceQu_Col_test] = imputer_spcl.transform(dataset_test.iloc[:, FireplaceQu_Col_test].values.\
                                          reshape(-1,1))

FireplaceQu_NaN_test = dataset_test["FireplaceQu"].isna().sum()

PoolQC_Col_train = dataset_train.columns.get_loc("PoolQC")

imputer_spcl = imputer_spcl.fit(dataset_train.iloc[:, PoolQC_Col_train].values.reshape(-1, 1))

dataset_train.iloc[:, PoolQC_Col_train] = imputer_spcl.transform(dataset_train.iloc[:, PoolQC_Col_train].values.\
                                          reshape(-1,1))

PoolQC_NaN_train = dataset_train["PoolQC"].isna().sum()

PoolQC_Col_test = dataset_test.columns.get_loc("PoolQC")

imputer_spcl = imputer_spcl.fit(dataset_test.iloc[:, PoolQC_Col_test].values.reshape(-1, 1))

dataset_test.iloc[:, PoolQC_Col_test] = imputer_spcl.transform(dataset_test.iloc[:, PoolQC_Col_test].values.\
                                          reshape(-1,1))

PoolQC_NaN_test = dataset_test["PoolQC"].isna().sum()

Fence_Col_train = dataset_train.columns.get_loc("Fence")

imputer_spcl = imputer_spcl.fit(dataset_train.iloc[:, Fence_Col_train].values.reshape(-1, 1))

dataset_train.iloc[:, Fence_Col_train] = imputer_spcl.transform(dataset_train.iloc[:, Fence_Col_train].values.\
                                          reshape(-1,1))

Fence_NaN_train = dataset_train["Fence"].isna().sum()

Fence_Col_test = dataset_test.columns.get_loc("Fence")

imputer_spcl = imputer_spcl.fit(dataset_test.iloc[:, Fence_Col_test].values.reshape(-1, 1))

dataset_test.iloc[:, Fence_Col_test] = imputer_spcl.transform(dataset_test.iloc[:, Fence_Col_test].values.\
                                          reshape(-1,1))

Fence_NaN_test = dataset_test["Fence"].isna().sum()

# General NaN treatment

from sklearn.impute import SimpleImputer

# Checking the number of Null values in numerical data columns
    
Num_N_Nulls_train = dataset_train[numerical_data].isna().sum()
Num_N_Nulls_test = dataset_test[numerical_data].isna().sum()

imputer_num = SimpleImputer(missing_values = np.nan, strategy = "mean")

i = dataset_train.columns.get_loc("LotFrontage")
imputer = imputer_num.fit(dataset_train.iloc[:, i].values.reshape(-1, 1))
dataset_train.iloc[:,i] = imputer_num.transform(dataset_train.iloc[:, i].values.\
                                          reshape(-1,1))
dataset_test.iloc[:,i] = imputer_num.transform(dataset_test.iloc[:, i].values.\
                                          reshape(-1,1))
dataset_train["LotFrontage"].isna().sum()
dataset_test["LotFrontage"].isna().sum()
Num_N_Nulls_train = dataset_train[numerical_data].isna().sum()
Num_N_Nulls_test = dataset_test[numerical_data].isna().sum()

i = dataset_train.columns.get_loc("MasVnrArea")
imputer = imputer_num.fit(dataset_train.iloc[:, i].values.reshape(-1, 1))
dataset_train.iloc[:,i] = imputer_num.transform(dataset_train.iloc[:, i].values.\
                                          reshape(-1,1))
dataset_test.iloc[:,i] = imputer_num.transform(dataset_test.iloc[:, i].values.\
                                          reshape(-1,1))
dataset_train["MasVnrArea"].isna().sum()
dataset_test["MasVnrArea"].isna().sum()
Num_N_Nulls_train = dataset_train[numerical_data].isna().sum()
Num_N_Nulls_test = dataset_test[numerical_data].isna().sum()

imputer_num = SimpleImputer(missing_values = np.nan, strategy = "most_frequent")

i = dataset_train.columns.get_loc("GarageYrBlt")
imputer = imputer_num.fit(dataset_train.iloc[:, i].values.reshape(-1, 1))
dataset_train.iloc[:,i] = imputer_num.transform(dataset_train.iloc[:, i].values.\
                                          reshape(-1,1))
dataset_test.iloc[:,i] = imputer_num.transform(dataset_test.iloc[:, i].values.\
                                          reshape(-1,1))
dataset_train["GarageYrBlt"].isna().sum()
dataset_test["GarageYrBlt"].isna().sum()
Num_N_Nulls_train = dataset_train[numerical_data].isna().sum()
Num_N_Nulls_test = dataset_test[numerical_data].isna().sum()

imputer_num = SimpleImputer(missing_values = np.nan, strategy = "most_frequent")

for i in numerical_data:
    j = dataset_test.columns.get_loc(i)
    imputer_num = imputer_num.fit(dataset_test.iloc[:, j].values.reshape(-1, 1))
    dataset_test.iloc[:,j] = imputer_num.transform(dataset_test.iloc[:,j].values.\
                                              reshape(-1,1))

Num_N_Nulls_test = dataset_test[numerical_data].isna().sum()

# Checking the number of Null values in nominal categorical data columns

Nom_N_Nulls_train = dataset_train[categorical_nom_data].isna().sum()
Nom_N_Nulls_test = dataset_test[categorical_nom_data].isna().sum()

imputer_nom = SimpleImputer(missing_values = np.nan, strategy = "most_frequent")

n = dataset_train.columns.get_loc("Electrical")
imputer_nom = imputer_nom.fit(dataset_train.iloc[:, n].values.reshape(-1, 1))
dataset_train.iloc[:, n] = imputer_nom.transform(dataset_train.iloc[:, n].values.\
                                          reshape(-1,1))
dataset_test.iloc[:, n] = imputer_nom.transform(dataset_test.iloc[:, n].values.\
                                          reshape(-1,1))
dataset_train["Electrical"].isna().sum()
dataset_test["Electrical"].isna().sum()
Nom_N_Nulls_train = dataset_train[categorical_nom_data].isna().sum()
Nom_N_Nulls_test = dataset_test[categorical_nom_data].isna().sum()

imputer_nom = SimpleImputer(missing_values = np.nan, strategy = "constant", \
                            fill_value = "Miss")

n = dataset_train.columns.get_loc("GarageType")
imputer_nom = imputer_nom.fit(dataset_train.iloc[:, n].values.reshape(-1, 1))
dataset_train.iloc[:, n] = imputer_nom.transform(dataset_train.iloc[:, n].values.\
                                          reshape(-1,1))
dataset_test.iloc[:, n] = imputer_nom.transform(dataset_test.iloc[:, n].values.\
                                          reshape(-1,1))
dataset_train["GarageType"].isna().sum()
dataset_test["GarageType"].isna().sum()
Nom_N_Nulls_train = dataset_train[categorical_nom_data].isna().sum()
Nom_N_Nulls_test = dataset_test[categorical_nom_data].isna().sum()

imputer_nom = SimpleImputer(missing_values = np.nan, strategy = "most_frequent")

for m in categorical_nom_data:
    n = dataset_test.columns.get_loc(m)
    imputer_nom = imputer_nom.fit(dataset_test.iloc[:, n].values.reshape(-1, 1))
    dataset_test.iloc[:,n] = imputer_nom.transform(dataset_test.iloc[:,n].values.\
                                              reshape(-1,1))
        
Nom_N_Nulls_test = dataset_test[categorical_nom_data].isna().sum()

# Checking the number of Null values in ordinal categorical data columns

imputer_nom = SimpleImputer(missing_values = np.nan, strategy = "most_frequent")

n = dataset_train.columns.get_loc("KitchenQual")
imputer_nom = imputer_nom.fit(dataset_train.iloc[:, n].values.reshape(-1, 1))
dataset_train.iloc[:, n] = imputer_nom.transform(dataset_train.iloc[:, n].values.\
                                          reshape(-1,1))
dataset_test.iloc[:, n] = imputer_nom.transform(dataset_test.iloc[:, n].values.\
                                          reshape(-1,1))

"""
Miss: BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, GarageFinish,
 GarageQual, GarageCond
"""

Ord_N_Nulls_train = dataset_train[categorical_ord_data].isna().sum()
Ord_N_Nulls_test = dataset_test[categorical_ord_data].isna().sum()

imputer_ord = SimpleImputer(missing_values = np.nan, strategy = "constant", \
                            fill_value = "Miss")
    
for p in categorical_ord_data:
    q = dataset_train.columns.get_loc(p)
    # dataset[p] = dataset[p].astype(str)
    imputer_ord = imputer_ord.fit(dataset_train.iloc[:, q].values.reshape(-1, 1))
    dataset_train.iloc[:,q] = imputer_ord.transform(dataset_train.iloc[:,q].values.\
                                              reshape(-1,1))
    imputer_ord = imputer_ord.fit(dataset_test.iloc[:, q].values.reshape(-1, 1))
    dataset_test.iloc[:,q] = imputer_ord.transform(dataset_test.iloc[:,q].values.\
                                              reshape(-1,1))

Ord_N_Nulls_train = dataset_train[categorical_ord_data].isna().sum()
Ord_N_Nulls_test = dataset_test[categorical_ord_data].isna().sum()

"""###################### Numerical Data Analysis ##########################"""

# Checking correlations for numerical data

correlation_num_data = dataset_train.select_dtypes(include = "number").corr()

# Identify the relevant numerical columns in descending order

rel_num_cols = correlation_num_data.nlargest(len(numerical_data), \
                                             "SalePrice")["SalePrice"].index

cm = np.corrcoef(dataset_train[rel_num_cols].values.T)

# Identify the independent variables with high correlation between them

high_corr_threshold = 0.8
high_corr_pairs = []

for i in range(len(correlation_num_data.columns)):
    for j in range(i):
        if correlation_num_data.iloc[i, j] > high_corr_threshold:
            colname = correlation_num_data.columns[i]
            rowname = correlation_num_data.columns[j]
            high_corr_pairs.append((rowname, colname))

print(f"Pairs of variables with an absolute correlation greater than \
      {high_corr_threshold}:")
print(high_corr_pairs)

high_corr_cols = ["TotalBsmtSF", "GrLivArea", "GarageCars"]

for i in high_corr_cols:
    # j = dataset.columns.get_loc(i)
    numerical_data.remove(i)
    dataset_train = dataset_train.drop([i], axis = 1, errors = "ignore")
    dataset_test = dataset_test.drop([i], axis = 1, errors = "ignore")
    correlation_num_data = correlation_num_data.drop([i], axis = 0, errors = "ignore")
    correlation_num_data = correlation_num_data.drop([i], axis = 1, errors = "ignore")
    
print("Full train dataset shape is {}".format(dataset_train.shape))
print("Full test dataset shape is {}".format(dataset_test.shape))
print("Full correlation_num_data shape is {}".format(correlation_num_data.shape))

# Identify the first 10 relevant numerical columns

rel_num_cols = correlation_num_data.nlargest(10, "SalePrice")["SalePrice"].index

cm = np.corrcoef(dataset_train[rel_num_cols].values.T)

# Identify the last 10 relevant numerical columns

irrel_num_cols = correlation_num_data.nsmallest(10, "SalePrice")["SalePrice"].index

cm = np.corrcoef(dataset_train[irrel_num_cols].values.T)

# Identify the viariables with correlation between (-0.1, 0.1) with "SalePrice"

low_corr_vars = correlation_num_data['SalePrice'][(correlation_num_data['SalePrice'] \
                > -0.1) & (correlation_num_data['SalePrice'] < 0.1)]
    
for i in low_corr_vars.index:
    numerical_data.remove(i)
    if i not in dataset_train.columns:
        continue
    dataset_train = dataset_train.drop([i], axis = 1)
    dataset_test = dataset_test.drop([i], axis = 1)
    correlation_num_data = correlation_num_data.drop([i], axis = 0, errors = "ignore")
    correlation_num_data = correlation_num_data.drop([i], axis = 1, errors = "ignore")

print("Full train dataset shape is {}".format(dataset_train.shape))
print("Full test dataset shape is {}".format(dataset_test.shape))
print("Full correlation_num_data shape is {}".format(correlation_num_data.shape))

"""##################### Categorical Data Analysis #########################"""

single_nom_values_train = {column: dataset_train[column].unique() for column in categorical_nom_data}
single_nom_values_test = {column: dataset_train[column].unique() for column in categorical_nom_data}

count_train = 0

print("Unique Nominal Values")
for column, unique_value in single_nom_values_train.items():
    print(f' {column}: {unique_value}')
    count_train += len(unique_value)
    
count_test = 0
    
print("Unique Nominal Values")
for column, unique_value in single_nom_values_test.items():
    print(f' {column}: {unique_value}')
    count_test += len(unique_value)

# Encoding ordinal categorical features

dict_cat_ord_data = {"ExterQual": ["Fa", "TA", "Gd", "Ex"],
                     "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
                     "BsmtQual": ["Miss", "Fa", "TA", "Gd", "Ex"],
                     "BsmtCond": ["Miss", "Po", "Fa", "TA", "Gd"],
                     "BsmtExposure": ["Miss", "No", "Mn", "Av", "Gd"],
                     "BsmtFinType1": ["Miss", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
                     "BsmtFinType2": ["Miss", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
                     "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
                     "KitchenQual": ["Fa", "TA", "Gd", "Ex"],
                     "FireplaceQu": ["Miss", "Po", "Fa", "TA", "Gd", "Ex"],
                     "GarageFinish": ["Miss", "Unf", "RFn", "Fin"],
                     "GarageQual": ["Miss", "Po", "Fa", "TA", "Gd", "Ex"],
                     "GarageCond": ["Miss", "Po", "Fa", "TA", "Gd", "Ex"],
                     "PoolQC": ["Miss", "Fa", "Gd", "Ex"],
                     "Fence": ["Miss", "MnWw", "GdWo", "MnPrv", "GdPrv"],
                     "SaleCondition": ["Partial", "Family", "Alloca", "AdjLand", "Abnorml", "Normal"]}

from sklearn.preprocessing import OrdinalEncoder

for i, j in dict_cat_ord_data.items():
    ord_encoder = OrdinalEncoder(categories = [j])
    dataset_train[i] = ord_encoder.fit_transform(dataset_train[i].values.reshape(-1,1))
    dataset_test[i] = ord_encoder.fit_transform(dataset_test[i].values.reshape(-1,1))

# Encoding nominal categorical features

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
    
nom_data_transformer_train = ColumnTransformer([('one_hot_encoder', \
                                         OneHotEncoder(categories = 'auto'),\
                                         categorical_nom_data)],\
                                         remainder = 'passthrough')

nom_data_transformer_test = ColumnTransformer([('one_hot_encoder', \
                                         OneHotEncoder(categories = 'auto'),\
                                         categorical_nom_data)],\
                                         remainder = 'passthrough')

encoded_col_train = nom_data_transformer_train.fit_transform(dataset_train[categorical_nom_data]).toarray()
encoded_col_test = nom_data_transformer_test.fit_transform(dataset_test[categorical_nom_data]).toarray()

dataset_num_train = dataset_train[numerical_data]
dataset_num_test = dataset_test[numerical_data]

dataset_ord_train = dataset_train[categorical_ord_data]
dataset_ord_test = dataset_test[categorical_ord_data]

dataset_nom_train = dataset_train[categorical_nom_data]
dataset_nom_test = dataset_test[categorical_nom_data]

dataset_encoded_train = pd.DataFrame(encoded_col_train, \
            columns = nom_data_transformer_train.get_feature_names_out(categorical_nom_data))
dataset_encoded_test = pd.DataFrame(encoded_col_test, \
            columns = nom_data_transformer_test.get_feature_names_out(categorical_nom_data))
    
# Checks that the number of rows of a DataFrame to concatenate is the same 
    
dataset_encoded_train = dataset_encoded_train.reset_index(drop = True)
dataset_num_train = dataset_num_train.reset_index(drop = True)
dataset_ord_train = dataset_ord_train.reset_index(drop = True)
    
dataset_encoded_train = pd.concat([dataset_encoded_train, dataset_num_train, dataset_ord_train], axis = 1)

dataset_encoded_test = dataset_encoded_test.reset_index(drop = True)
dataset_num_test = dataset_num_test.reset_index(drop = True)
dataset_ord_test = dataset_ord_test.reset_index(drop = True)
    
dataset_encoded_test = pd.concat([dataset_encoded_test, dataset_num_test, dataset_ord_test], axis = 1)

missing_columns = set(dataset_encoded_train.columns) - set(dataset_encoded_test.columns)

for col in missing_columns:
    dataset_encoded_test[col] = 0.0
    
dataset_encoded_test = dataset_encoded_test[dataset_encoded_train.columns]

# Scaling of numerical data

# from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

dataset_encoded_train[numerical_data] = scaler.fit_transform(dataset_encoded_train[numerical_data])
dataset_encoded_test[numerical_data] = scaler.fit_transform(dataset_encoded_test[numerical_data])

X_train = dataset_encoded_train
y_train = dataset_train["SalePrice"]
X_test = dataset_encoded_test

# Fit a XGBoost Regression model with the Training set

import xgboost as xgb

regressor_XGB = xgb.XGBRegressor(objective = 'reg:squarederror',\
                             n_estimators = 1400, learning_rate = 0.1,\
                             max_depth = 4)
regressor_XGB.fit(X_train, y_train)

y_pred_XGB = regressor_XGB.predict(X_test)

# Comparing predictions with "sample_submission.csv" file

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sample_df = pd.read_csv("sample_submission.csv", index_col = "Id")

y_test = sample_df["SalePrice"]

mse_XGB = mean_squared_error(y_test, y_pred_XGB)
mae_XGB = mean_absolute_error(y_test, y_pred_XGB)
r2_XGB = r2_score(y_test, y_pred_XGB)

print(f'Mean Squared Error (MSE) XGB model: {mse_XGB:.5g}')
print(f'Mean Absolute Error (MAE) XGB model: {mae_XGB:.5g}')
print(f'R^2 Score XGB model: {r2_XGB:.5g}')

# Visualizing the results of the XGBoost Regression model

index = np.arange(len(y_pred_XGB))

plt.figure(figsize=(10, 6))
plt.plot(index, y_test, label = 'Test Set', color = 'red', marker = 'o')
plt.plot(index, y_pred_XGB, label = 'Predictions', color = 'blue', marker = 'x')
plt.ylim(y_test.min() - 1e+05, y_test.max() + 1e+05)
plt.xlabel('Index')
plt.ylabel('SalePrice')
plt.title('XGBoost Regression Model (RFR) Predictions')
plt.legend()
plt.show()

# Import predictions

id_column = np.arange(1461, 2920)

assert len(y_pred_XGB) == len(id_column)

predictions_df = pd.DataFrame({
    'Id': id_column,
    'SalePrice': y_pred_XGB
})

predictions_df.to_csv('SalePricePredictions.csv', index = False)

# Fit a Support Vector Regression model with the Training set

from sklearn.svm import SVR

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
y_train_scaled = y_train.to_numpy().reshape(-1,1)
y_train_scaled = scaler.fit_transform(y_train_scaled)

regressor_SVR = SVR(kernel = 'linear')
regressor_SVR.fit(X_train_scaled, y_train_scaled.ravel())

y_pred_scaled_SVR = regressor_SVR.predict(X_test_scaled)
y_pred_scaled_SVR = y_pred_scaled_SVR.reshape(len(y_pred_scaled_SVR), 1)
y_pred_SVR = scaler.inverse_transform(y_pred_scaled_SVR)

mse_SVR = mean_squared_error(y_test, y_pred_SVR)
mae_SVR = mean_absolute_error(y_test, y_pred_SVR)
r2_SVR = r2_score(y_test, y_pred_SVR)

print(f'Mean Squared Error (MSE) SVR model: {mse_SVR:.5g}')
print(f'Mean Absolute Error (MAE) SVR model: {mae_SVR:.5g}')
print(f'R^2 Score SVR model: {r2_SVR:.5g}')

# Visualizing the results of the Support Vector Regression model

plt.scatter(y_test, y_pred_SVR, color = 'blue', edgecolor = 'w', \
            label = 'Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], \
         color = 'red', linestyle = '--', lw = 2, label = 'Perfect Fit')
plt.xlabel('Real Values')
plt.ylabel('SVR Predictions')
plt.title('Support Vector Regression Model (SVR) Predictions')
plt.legend()
plt.show()

index = np.arange(len(y_test))

plt.figure(figsize=(10, 6))
plt.plot(index, y_test, label = 'Test Set', color = 'red', marker = 'o')
plt.plot(index, y_pred_SVR, label = 'Predictions', color = 'blue', marker = 'x')
plt.ylim(y_test.min() - 1e+05, y_test.max() + 1e+05)
plt.xlabel('Index')
plt.ylabel('SalePrice')
plt.title('Support Vector Regression Model (SVR) Predictions')
plt.legend()
plt.show()

# Import predictions

id_column = np.arange(1461, 2920)

assert len(y_pred_SVR) == len(id_column)

predictions_df = pd.DataFrame({
    'Id': id_column,
    'SalePrice': y_pred_SVR.ravel()
})

predictions_df.to_csv('SalePricePredictionsv01.csv', index = False)

# Fit a Random Forest Regression model with the Training set

from sklearn.ensemble import RandomForestRegressor

regressor_RFR = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor_RFR.fit(X_train, y_train.ravel())

y_pred_RFR = regressor_RFR.predict(X_test)

mse_RFR = mean_squared_error(y_test, y_pred_RFR)
mae_RFR = mean_absolute_error(y_test, y_pred_RFR)
r2_RFR = r2_score(y_test, y_pred_RFR)

print(f'Mean Squared Error (MSE) RFR model: {mse_RFR:.5g}')
print(f'Mean Absolute Error (MAE) RFR model: {mae_RFR:.5g}')
print(f'R^2 Score RFR model: {r2_RFR:.5g}')

# Visualizing the results of the Random Forest Regression model

plt.scatter(y_test, y_pred_RFR, color = 'blue', edgecolor = 'w', \
            label = 'Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], \
         color = 'red', linestyle = '--', lw = 2, label = 'Perfect Fit')
plt.xlabel('Real Values')
plt.ylabel('RFR Predictions')
plt.title('Random Forest Regression Model (RFR) Predictions')
plt.legend()
plt.show()

index = np.arange(len(y_test))

plt.figure(figsize=(10, 6))
plt.plot(index, y_test, label = 'Test Set', color = 'red', marker = 'o')
plt.plot(index, y_pred_RFR, label = 'Predictions', color = 'blue', marker = 'x')
plt.ylim(y_test.min() - 1e+05, y_test.max() + 1e+05)
plt.xlabel('Index')
plt.ylabel('SalePrice')
plt.title('Random Forest Regression Model (RFR) Predictions')
plt.legend()
plt.show()

# Import predictions

id_column = np.arange(1461, 2920)

assert len(y_pred_RFR) == len(id_column)

predictions_df = pd.DataFrame({
    'Id': id_column,
    'SalePrice': y_pred_RFR.ravel()
})

predictions_df.to_csv('SalePricePredictionsv02.csv', index = False)

# Fit a K-Nearest Neighbors Regression model with the Training set

from sklearn.neighbors import KNeighborsRegressor

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

regressor_KNN = KNeighborsRegressor(n_neighbors = 200)
regressor_KNN.fit(X_train_scaled, y_train)

y_pred_KNN = regressor_KNN.predict(X_test_scaled)

mse_KNN = mean_squared_error(y_test, y_pred_KNN)
mae_KNN = mean_absolute_error(y_test, y_pred_KNN)
r2_KNN = r2_score(y_test, y_pred_KNN)

print(f'Mean Squared Error (MSE) KNN model: {mse_KNN:.5g}')
print(f'Mean Absolute Error (MAE) KNN model: {mae_KNN:.5g}')
print(f'R^2 Score KNN model: {r2_KNN:.5g}')

# Visualizing the results of the K-Nearest Neighbors Regression model

plt.scatter(y_test, y_pred_KNN, color = 'blue', edgecolor = 'w', \
            label = 'Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], \
         color = 'red', linestyle = '--', lw = 2, label = 'Perfect Fit')
plt.xlabel('Real Values')
plt.ylabel('KNN Predictions')
plt.title('K-Nearest Neighbors Regression Model (XGB) Predictions')
plt.legend()
plt.show()

index = np.arange(len(y_test))

plt.figure(figsize=(10, 6))
plt.plot(index, y_test, label = 'Test Set', color = 'red', marker = 'o')
plt.plot(index, y_pred_KNN, label = 'Predictions', color = 'blue', marker = 'x')
plt.ylim(y_test.min() - 1e+05, y_test.max() + 1e+05)
plt.xlabel('Index')
plt.ylabel('SalePrice')
plt.title('K-Nearest Neighbors Regression Model (RFR) Predictions')
plt.legend()
plt.show()

# Import predictions

id_column = np.arange(1461, 2920)

assert len(y_pred_KNN) == len(id_column)

predictions_df = pd.DataFrame({
    'Id': id_column,
    'SalePrice': y_pred_KNN.ravel()
})

predictions_df.to_csv('SalePricePredictionsv03.csv', index = False)
