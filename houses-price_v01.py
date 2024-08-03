# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 20:09:11 2024

@author: Francisco De La Cruz

Description: This Python script takes the "train.csv" file due the "test.csv"
file does not contain the 'SalePrice' variabe that is our predictible variable. 
In order to test the regresion models we'll apply we are going to use the 
"train.csv" file as the whole Dataset we'll split into 'Training' and 'Testing'
subsets. 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

# %matplotlib inline

"""*************************************************************************"""
""">>>>>>>>>>>>>>>>>>>> ANALYSIS AND DATA PREPROSECING <<<<<<<<<<<<<<<<<<<<<"""
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

dataset = pd.read_csv('train.csv', index_col = "Id")

print("Full dataset shape is {}".format(dataset.shape))

descript_dataset = dataset.describe(include = "object").T

dataset.info()

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

Gen_N_Nulls = dataset.isna().sum()

""" 
OBSERVATIONS: Variables such as 'Alley', 'MasVnrType', 'FireplaceQu', 
'PoolQC', 'Fence', 'MiscFeature' has a huge number of Null values. Let's 
apply a special NaN Treatment for these variables.
"""

Alley_NaN = Gen_N_Nulls['Alley']
MasVnrType_NaN = Gen_N_Nulls['MasVnrType']
FireplaceQu_NaN = Gen_N_Nulls['FireplaceQu']
PoolQC_NaN = Gen_N_Nulls['PoolQC']
Fence_NaN = Gen_N_Nulls['Fence']
MiscFeature_NaN = Gen_N_Nulls['MiscFeature']

print(f'No. of Null values ("Alley"): {Alley_NaN}')
print(f'No. of Null values ("MasVnrType"): {MasVnrType_NaN}')
print(f'No. of Null values ("FireplaceQu"): {FireplaceQu_NaN}')
print(f'No. of Null values ("PoolQC"): {PoolQC_NaN}')
print(f'No. of Null values ("Fence"): {Fence_NaN}')
print(f'No. of Null values ("MiscFeature"): {MiscFeature_NaN}')

dataset = dataset.drop(["MiscFeature"], axis = 1)

print("Full dataset shape is {}".format(dataset.shape))

# NaN Treatment on the variables mentioned above

from sklearn.impute import SimpleImputer

imputer_spcl = SimpleImputer(missing_values = np.nan, strategy = "constant" \
                              , fill_value = "Miss")

Alley_Col = dataset.columns.get_loc("Alley")

imputer_Alley = imputer_spcl.fit(dataset.iloc[:, Alley_Col].values.reshape(-1, 1))

dataset.iloc[:, Alley_Col] = imputer_Alley.transform(dataset.iloc[:,Alley_Col].values.\
                                          reshape(-1,1))

Alley_NaN = dataset["Alley"].isna().sum()

MasVnrType_Col = dataset.columns.get_loc("MasVnrType")

imputer_spcl = imputer_spcl.fit(dataset.iloc[:, MasVnrType_Col].values.reshape(-1, 1))

dataset.iloc[:, MasVnrType_Col] = imputer_spcl.transform(dataset.iloc[:, MasVnrType_Col].values.\
                                          reshape(-1,1))

MasVnrType_NaN = dataset["MasVnrType"].isna().sum()

FireplaceQu_Col = dataset.columns.get_loc("FireplaceQu")

imputer_spcl = imputer_spcl.fit(dataset.iloc[:, FireplaceQu_Col].values.reshape(-1, 1))

dataset.iloc[:, FireplaceQu_Col] = imputer_spcl.transform(dataset.iloc[:, FireplaceQu_Col].values.\
                                          reshape(-1,1))

FireplaceQu_NaN = dataset["FireplaceQu"].isna().sum()

PoolQC_Col = dataset.columns.get_loc("PoolQC")

imputer_spcl = imputer_spcl.fit(dataset.iloc[:, PoolQC_Col].values.reshape(-1, 1))

dataset.iloc[:, PoolQC_Col] = imputer_spcl.transform(dataset.iloc[:, PoolQC_Col].values.\
                                          reshape(-1,1))

PoolQC_NaN = dataset["PoolQC"].isna().sum()

Fence_Col = dataset.columns.get_loc("Fence")

imputer_spcl = imputer_spcl.fit(dataset.iloc[:, Fence_Col].values.reshape(-1, 1))

dataset.iloc[:, Fence_Col] = imputer_spcl.transform(dataset.iloc[:, Fence_Col].values.\
                                          reshape(-1,1))

Fence_NaN = dataset["Fence"].isna().sum()

print(f'No. of Null values ("Alley"): {Alley_NaN}')
print(f'No. of Null values ("MasVnrType"): {MasVnrType_NaN}')
print(f'No. of Null values ("FireplaceQu"): {FireplaceQu_NaN}')
print(f'No. of Null values ("PoolQC"): {PoolQC_NaN}')
print(f'No. of Null values ("Fence"): {Fence_NaN}')

# General NaN treatment

from sklearn.impute import SimpleImputer

# Checking the number of Null values in numerical data columns
    
Num_N_Nulls = dataset[numerical_data].isna().sum()

imputer_num = SimpleImputer(missing_values = np.nan, strategy = "mean")

i = dataset.columns.get_loc("LotFrontage")
imputer = imputer_num.fit(dataset.iloc[:, i].values.reshape(-1, 1))
dataset.iloc[:,i] = imputer_num.transform(dataset.iloc[:, i].values.\
                                          reshape(-1,1))
dataset["LotFrontage"].isna().sum()
Num_N_Nulls = dataset[numerical_data].isna().sum()

i = dataset.columns.get_loc("MasVnrArea")
imputer = imputer_num.fit(dataset.iloc[:, i].values.reshape(-1, 1))
dataset.iloc[:,i] = imputer_num.transform(dataset.iloc[:, i].values.\
                                          reshape(-1,1))
dataset["MasVnrArea"].isna().sum()
Num_N_Nulls = dataset[numerical_data].isna().sum()

imputer_num = SimpleImputer(missing_values = np.nan, strategy = "most_frequent")

i = dataset.columns.get_loc("GarageYrBlt")
imputer = imputer_num.fit(dataset.iloc[:, i].values.reshape(-1, 1))
dataset.iloc[:,i] = imputer_num.transform(dataset.iloc[:, i].values.\
                                          reshape(-1,1))
dataset["GarageYrBlt"].isna().sum()
Num_N_Nulls = dataset[numerical_data].isna().sum()

# Checking the number of Null values in nominal categorical data columns

Nom_N_Nulls = dataset[categorical_nom_data].isna().sum()

imputer_nom = SimpleImputer(missing_values = np.nan, strategy = "most_frequent")

n = dataset.columns.get_loc("Electrical")
imputer_nom = imputer_nom.fit(dataset.iloc[:, n].values.reshape(-1, 1))
dataset.iloc[:, n] = imputer_nom.transform(dataset.iloc[:, n].values.\
                                          reshape(-1,1))
dataset["Electrical"].isna().sum()
Nom_N_Nulls = dataset[categorical_nom_data].isna().sum()

imputer_nom = SimpleImputer(missing_values = np.nan, strategy = "constant", \
                            fill_value = "Miss")

n = dataset.columns.get_loc("GarageType")
imputer_nom = imputer_nom.fit(dataset.iloc[:, n].values.reshape(-1, 1))
dataset.iloc[:, n] = imputer_nom.transform(dataset.iloc[:, n].values.\
                                          reshape(-1,1))
dataset["GarageType"].isna().sum()
Nom_N_Nulls = dataset[categorical_nom_data].isna().sum()

# Checking the number of Null values in ordinal categorical data columns

"""
Miss: BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, GarageFinish,
 GarageQual, GarageCond
"""

Ord_N_Nulls = dataset[categorical_ord_data].isna().sum()

imputer_ord = SimpleImputer(missing_values = np.nan, strategy = "constant", \
                            fill_value = "Miss")
    
for p in categorical_ord_data:
    q = dataset.columns.get_loc(p)
    # dataset[p] = dataset[p].astype(str)
    imputer_ord = imputer_ord.fit(dataset.iloc[:, q].values.reshape(-1, 1))
    dataset.iloc[:,q] = imputer_ord.transform(dataset.iloc[:,q].values.\
                                              reshape(-1,1))
Ord_N_Nulls = dataset[categorical_ord_data].isna().sum()

"""###################### Numerical Data Analysis ##########################"""

# Checking correlations for numerical data

correlation_num_data = dataset.select_dtypes(include = "number").corr()
f, ax = plt.subplots(figsize = (12,10))
hm_num = sns.heatmap(correlation_num_data)
plt.title("Heatmap of Numerical Data", fontsize=18, weight='bold')
plt.show()

# Identify the relevant numerical columns in descending order

rel_num_cols = correlation_num_data.nlargest(len(numerical_data), \
                                             "SalePrice")["SalePrice"].index

cm = np.corrcoef(dataset[rel_num_cols].values.T)

sns.set(font_scale = 1.25)
f, ax = plt.subplots(figsize = (22,20))
hm_rel_num = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = ".2f",\
                 annot_kws = {"size":10,}, yticklabels = rel_num_cols.values,\
                     xticklabels = rel_num_cols.values)
plt.title("Initial Heatmap of Numerical Data", fontsize=18, weight='bold')
plt.text(0.5, 1.05, "Correlations with 'SalePrice", ha='center', va='center', \
         transform=hm_num.transAxes, fontsize=12)
plt.show()

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
    dataset = dataset.drop([i], axis = 1, errors = "ignore")
    correlation_num_data = correlation_num_data.drop([i], axis = 0, errors = "ignore")
    correlation_num_data = correlation_num_data.drop([i], axis = 1, errors = "ignore")
    
print("Full dataset shape is {}".format(dataset.shape))
print("Full correlation_num_data shape is {}".format(correlation_num_data.shape))

# Identify the first 10 relevant numerical columns

rel_num_cols = correlation_num_data.nlargest(10, "SalePrice")["SalePrice"].index

cm = np.corrcoef(dataset[rel_num_cols].values.T)

sns.set(font_scale = 1.25)
f, ax = plt.subplots(figsize = (10,10))
hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = ".2f",\
                 annot_kws = {"size":10,}, yticklabels = rel_num_cols.values,\
                     xticklabels = rel_num_cols.values)
plt.title("Heatmap of Numerical Data", fontsize=18, weight='bold')
plt.text(0.5, 1.05, "Most relevant correlations with 'SalePrice", ha='center', va='center', \
         transform=hm_num.transAxes, fontsize=12)
plt.show()

# Identify the last 10 relevant numerical columns

irrel_num_cols = correlation_num_data.nsmallest(10, "SalePrice")["SalePrice"].index

cm = np.corrcoef(dataset[irrel_num_cols].values.T)

sns.set(font_scale = 1.25)
f, ax = plt.subplots(figsize = (10,10))
hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = ".2f",\
                 annot_kws = {"size":10,}, yticklabels = irrel_num_cols.values,\
                     xticklabels = irrel_num_cols.values)
plt.title("Heatmap of Numerical Data", fontsize=18, weight='bold')
plt.text(0.5, 1.05, "Most irrelevant correlations with 'SalePrice", ha='center', va='center', \
         transform=hm_num.transAxes, fontsize=12)
plt.show()

# Identify the viariables with correlation between (-0.1, 0.1) with "SalePrice"

low_corr_vars = correlation_num_data['SalePrice'][(correlation_num_data['SalePrice'] \
                > -0.1) & (correlation_num_data['SalePrice'] < 0.1)]
    
for i in low_corr_vars.index:
    numerical_data.remove(i)
    if i not in dataset.columns:
        continue
    dataset = dataset.drop([i], axis = 1)
    correlation_num_data = correlation_num_data.drop([i], axis = 0, errors = "ignore")
    correlation_num_data = correlation_num_data.drop([i], axis = 1, errors = "ignore")

print("Full dataset shape is {}".format(dataset.shape))
print("Full correlation_num_data shape is {}".format(correlation_num_data.shape))

# Final HeatMap of correlations bewteen "SalePrice" and numerical data

rel_num_cols = correlation_num_data.nlargest(len(numerical_data), \
                                             "SalePrice")["SalePrice"].index

cm = np.corrcoef(dataset[rel_num_cols].values.T)

sns.set(font_scale = 1.25)
f, ax = plt.subplots(figsize = (20,18))
hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = ".2f",\
                 annot_kws = {"size":10,}, yticklabels = rel_num_cols.values,\
                     xticklabels = rel_num_cols.values)
plt.title("Final Heatmap of Numerical Data", fontsize=18, weight='bold')
plt.show()

"""##################### Categorical Data Analysis #########################"""

# Relationship with nominal categorical features

for m in categorical_nom_data:
    if m not in dataset.columns:
        continue
    else: 
        n = dataset.columns.get_loc(m)
        data_nom_i = pd.concat([dataset["SalePrice"], dataset[m]], axis = 1)
        f, ax = plt.subplots(figsize = (8,6))
        fig = sns.boxplot(x = str(m), y = "SalePrice", data = data_nom_i)
        plt.title("Categorical Nominal Data", fontsize=18, weight='bold')
        plt.xticks(rotation = 90)
        plt.show()
        
len(categorical_nom_data)

# Relationship with ordinal categorical features

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

for i, j in dict_cat_ord_data.items():
    if i not in dataset.columns:
        continue
    else:
        cat_dtype = pd.CategoricalDtype(categories = j, ordered = True)
        dataset[i] = dataset[i].astype(cat_dtype)
        data_nom_i = pd.concat([dataset["SalePrice"], dataset[i]], axis = 1)
        f, ax = plt.subplots(figsize = (8,6))
        fig = sns.boxplot(x = str(i), y = "SalePrice", data = data_nom_i)
        plt.title("Categorical Ordinal Data", fontsize=18, weight='bold')
        plt.xticks(rotation = 90)
        plt.show()
        
len(categorical_ord_data)

# Let's see the unique values in categorical data columns

single_nom_values = {column: dataset[column].unique() for column in categorical_nom_data}

count = 0

print("Unique Nominal Values")
for column, unique_value in single_nom_values.items():
    print(f' {column}: {unique_value}')
    count += len(unique_value)
    
print(f' No. Unique Nominal Values: {count}')
    
single_ord_values = {column: dataset[column].unique() for column in categorical_ord_data}

print("Unique Ordinal Values")
for column, unique_value in single_ord_values.items():
    print(f' {column}: {unique_value}')
    
# Encoding ordinal categorical features

from sklearn.preprocessing import OrdinalEncoder

for i, j in dict_cat_ord_data.items():
    ord_encoder = OrdinalEncoder(categories = [j])
    dataset[i] = ord_encoder.fit_transform(dataset[i].values.reshape(-1,1))
    
# Verify the encoded values

single_ord_values = {column: dataset[column].unique() for column in categorical_ord_data}

print("Unique Ordinal Values")
for column, unique_value in single_ord_values.items():
    print(f' {column}: {unique_value}')

# Encoding nominal categorical features

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
    
nom_data_transformer = ColumnTransformer([('one_hot_encoder', \
                                         OneHotEncoder(categories = 'auto'),\
                                         categorical_nom_data)],\
                                         remainder = 'passthrough')

encoded_col = nom_data_transformer.fit_transform(dataset[categorical_nom_data]).toarray()

dataset_num = dataset[numerical_data]

dataset_ord = dataset[categorical_ord_data]

dataset_encoded = pd.DataFrame(encoded_col, \
            columns = nom_data_transformer.get_feature_names_out(categorical_nom_data))
    
# Checks that the number of rows of a DataFrame to concatenate is the same 
    
dataset_encoded = dataset_encoded.reset_index(drop = True)
dataset_num = dataset_num.reset_index(drop = True)
dataset_ord = dataset_ord.reset_index(drop = True)
    
dataset_encoded = pd.concat([dataset_encoded, dataset_num, dataset_ord], axis = 1)
    
# Scaling of numerical data

# from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

dataset_encoded[numerical_data] = scaler.fit_transform(dataset_encoded[numerical_data])

dataset_encoded = dataset_encoded.drop(dataset_encoded.index[-1])    
dataset = dataset.drop(dataset.index[-1])

# Analysis of target variable "SalePrice"

dataset["SalePrice"].describe()

sns.displot(dataset["SalePrice"])

SP_skew = dataset["SalePrice"].skew()
SP_kurt = dataset["SalePrice"].kurt()
print(f'Skewness of SalePrice variable: {SP_skew}')
print(f'Kurtosis of SalePrice variable: {SP_kurt}')

sns.set()
cols = ["SalePrice", "OverallQual", "FullBath", "YearBuilt"]
sns.pairplot(dataset[cols], size = 2.5)
plt.show()

# Splitting the dataset into Training and Test sets

X = dataset_encoded.iloc[:, :].values
y = dataset.iloc[:, -1].values

y = y.reshape(-1,1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2,\
                                                    random_state = 0)

nan_counts = dataset_encoded.isnull().sum()

metrics_models = pd.DataFrame(columns = ["Model", "MSE", "MAE", "R^2"])

"""*************************************************************************"""
""">>>>>>>>>>>>>>>>>>>>>>>>>> REGRESSION MODELS <<<<<<<<<<<<<<<<<<<<<<<<<<<<"""
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

# Fit the Multiple Linear Regression model with the Training set

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

regressor_MLR = LinearRegression()
regressor_MLR.fit(X_train_scaled, y_train)

y_pred_MLR = regressor_MLR.predict(X_test_scaled)

mse_MLR = mean_squared_error(y_test, y_pred_MLR)
mae_MLR = mean_absolute_error(y_test, y_pred_MLR)
r2_MLR = r2_score(y_test, y_pred_MLR)

print(f'Mean Squared Error (MSE) MLR model: {mse_MLR:.5g}')
print(f'Mean Absolute Error (MAE) MLR model: {mae_MLR:.5g}')
print(f'R^2 Score MLR model: {r2_MLR:.5g}')

metric_MLR = ["MLR", f"{mse_MLR:.5g}", f"{mae_MLR:.5g}", f"{r2_MLR:.5g}"]

metrics_models.loc[0] = metric_MLR

# Visualizing the results of the Multiple Linear Regression model

plt.scatter(y_test, y_pred_MLR, color = 'blue', edgecolor = 'w', \
            label = 'Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], \
         color = 'red', linestyle = '--', lw = 2, label = 'Perfect Fit')
plt.xlabel('Real Values')
plt.ylabel('RLM Predictions')
plt.title('Multiple Linear Regression Model (MLR) Predictions')
plt.legend()
plt.show()

index = np.arange(len(y_test))

plt.figure(figsize=(10, 6))
plt.plot(index, y_test, label = 'Test Set', color = 'red', marker = 'o')
plt.plot(index, y_pred_MLR, label = 'Predictions', color = 'blue', marker = 'x')
plt.ylim(y_test.min() - 1e+05, y_test.max() + 1e+05)
plt.xlabel('Index')
plt.ylabel('SalePrice')
plt.title('Multiple Linear Regression Model (MLR) Predictions')
plt.legend()
plt.show()

# Fit a Support Vector Regression model with the Training set

from sklearn.svm import SVR

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
y_train_scaled = scaler.fit_transform(y_train.reshape(len(y_train), 1))

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

metric_SVR = ["SVR", f"{mse_SVR:.5g}", f"{mae_SVR:.5g}", f"{r2_SVR:.5g}"]

metrics_models.loc[1] = metric_SVR

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

# Fit a Decision Tree Regression model with the Training set

from sklearn.tree import DecisionTreeRegressor

regressor_DTR = DecisionTreeRegressor(criterion = 'absolute_error', \
                                       random_state = 3)
regressor_DTR.fit(X_train, y_train)

y_pred_DTR = regressor_DTR.predict(X_test)

mse_DTR = mean_squared_error(y_test, y_pred_DTR)
mae_DTR = mean_absolute_error(y_test, y_pred_DTR)
r2_DTR = r2_score(y_test, y_pred_DTR)

print(f'Mean Squared Error (MSE) DTR model: {mse_DTR:.5g}')
print(f'Mean Absolute Error (MAE) DTR model: {mae_DTR:.5g}')
print(f'R^2 Score DTR model: {r2_DTR:.5g}')

metric_DTR = ["DTR", f"{mse_DTR:.5g}", f"{mae_DTR:.5g}", f"{r2_DTR:.5g}"]

metrics_models.loc[2] = metric_DTR

# Visualizing the results of the Decision Tree Regression model

plt.scatter(y_test, y_pred_DTR, color = 'blue', edgecolor = 'w', \
            label = 'Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], \
         color = 'red', linestyle = '--', lw = 2, label = 'Perfect Fit')
plt.xlabel('Real Values')
plt.ylabel('DTR Predictions')
plt.title('Decision Tree Regression Model (DTR) Predictions')
plt.legend()
plt.show()

index = np.arange(len(y_test))

plt.figure(figsize=(10, 6))
plt.plot(index, y_test, label = 'Test Set', color = 'red', marker = 'o')
plt.plot(index, y_pred_DTR, label = 'Predictions', color = 'blue', marker = 'x')
plt.ylim(y_test.min() - 1e+05, y_test.max() + 1e+05)
plt.xlabel('Index')
plt.ylabel('SalePrice')
plt.title('Decision Tree Regression Model (DTR) Predictions')
plt.legend()
plt.show()

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

metric_RFR = ["RFR", f"{mse_RFR:.5g}", f"{mae_RFR:.5g}", f"{r2_RFR:.5g}"]

metrics_models.loc[3] = metric_RFR

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

# Fit a XGBoost Regression model with the Training set

import xgboost as xgb

regressor_XGB = xgb.XGBRegressor(objective = 'reg:squarederror',\
                             n_estimators = 1400, learning_rate = 0.1,\
                             max_depth = 4)
regressor_XGB.fit(X_train, y_train)

y_pred_XGB = regressor_XGB.predict(X_test)

mse_XGB = mean_squared_error(y_test, y_pred_XGB)
mae_XGB = mean_absolute_error(y_test, y_pred_XGB)
r2_XGB = r2_score(y_test, y_pred_XGB)

print(f'Mean Squared Error (MSE) XGB model: {mse_XGB:.5g}')
print(f'Mean Absolute Error (MAE) XGB model: {mae_XGB:.5g}')
print(f'R^2 Score XGB model: {r2_XGB:.5g}')

metric_XGB = ["XGB", f"{mse_XGB:.5g}", f"{mae_XGB:.5g}", f"{r2_XGB:.5g}"]

metrics_models.loc[4] = metric_XGB

# Visualizing the results of the XGBoost Regression model

plt.scatter(y_test, y_pred_XGB, color = 'blue', edgecolor = 'w', \
            label = 'Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], \
         color = 'red', linestyle = '--', lw = 2, label = 'Perfect Fit')
plt.xlabel('Real Values')
plt.ylabel('XGB Predictions')
plt.title('XGBoost Regression Model (XGB) Predictions')
plt.legend()
plt.show()

index = np.arange(len(y_test))

plt.figure(figsize=(10, 6))
plt.plot(index, y_test, label = 'Test Set', color = 'red', marker = 'o')
plt.plot(index, y_pred_XGB, label = 'Predictions', color = 'blue', marker = 'x')
plt.ylim(y_test.min() - 1e+05, y_test.max() + 1e+05)
plt.xlabel('Index')
plt.ylabel('SalePrice')
plt.title('XGBoost Regression Model (RFR) Predictions')
plt.legend()
plt.show()

# Fit a K-Nearest Neighbors Regression model with the Training set

from sklearn.neighbors import KNeighborsRegressor

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

regressor_KNN = KNeighborsRegressor(n_neighbors = 7)
regressor_KNN.fit(X_train_scaled, y_train)

y_pred_KNN = regressor_KNN.predict(X_test_scaled)

mse_KNN = mean_squared_error(y_test, y_pred_KNN)
mae_KNN = mean_absolute_error(y_test, y_pred_KNN)
r2_KNN = r2_score(y_test, y_pred_KNN)

print(f'Mean Squared Error (MSE) KNN model: {mse_KNN:.5g}')
print(f'Mean Absolute Error (MAE) KNN model: {mae_KNN:.5g}')
print(f'R^2 Score KNN model: {r2_KNN:.5g}')

metric_KNN = ["KNN", f"{mse_KNN:.5g}", f"{mae_KNN:.5g}", f"{r2_KNN:.5g}"]

metrics_models.loc[5] = metric_KNN

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

# Fit a ElasticNet Regression model with the Training set

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {'alpha': [0.1, 1.0, 10.0],
              'l1_ratio': [0.1, 0.5, 0.9]}

# regressor_ENR = ElasticNet(alpha = 1.0, l1_ratio = 0.5, random_state = 42)
regressor_ENR = ElasticNet(random_state = 42)
grid_search = GridSearchCV(estimator = regressor_ENR,\
                           param_grid = param_grid, cv = 5,\
                           scoring = 'neg_mean_squared_error')
regressor_ENR.fit(X_train_scaled, y_train)

y_pred_ENR = regressor_ENR.predict(X_test_scaled)

mse_ENR = mean_squared_error(y_test, y_pred_ENR)
mae_ENR = mean_absolute_error(y_test, y_pred_ENR)
r2_ENR = r2_score(y_test, y_pred_ENR)

print(f'Mean Squared Error (MSE) ENR model: {mse_ENR:.5g}')
print(f'Mean Absolute Error (MAE) ENR model: {mae_ENR:.5g}')
print(f'R^2 Score ENR model: {r2_ENR:.5g}')

metric_ENR = ["ENR", f"{mse_ENR:.5g}", f"{mae_ENR:.5g}", f"{r2_ENR:.5g}"]

metrics_models.loc[6] = metric_ENR

# Visualizing the results of the ElasticNet Regression model

plt.scatter(y_test, y_pred_ENR, color = 'blue', edgecolor = 'w', \
            label = 'Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], \
         color = 'red', linestyle = '--', lw = 2, label = 'Perfect Fit')
plt.xlabel('Real Values')
plt.ylabel('ENR Predictions')
plt.title('ElasticNet Regression Model (ENR) Predictions')
plt.legend()
plt.show()

index = np.arange(len(y_test))

plt.figure(figsize=(10, 6))
plt.plot(index, y_test, label = 'Test Set', color = 'red', marker = 'o')
plt.plot(index, y_pred_ENR, label = 'Predictions', color = 'blue', marker = 'x')
plt.ylim(y_test.min() - 1e+05, y_test.max() + 1e+05)
plt.xlabel('Index')
plt.ylabel('SalePrice')
plt.title('ElasticNet Regression Model (ENR) Predictions')
plt.legend()
plt.show()

# Fit a Bayesian Ridge Regression model with the Training set

from sklearn.linear_model import BayesianRidge

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

regressor_BRR = BayesianRidge()
regressor_BRR.fit(X_train_scaled, y_train.ravel())

y_pred_BRR = regressor_BRR.predict(X_test_scaled)

mse_BRR = mean_squared_error(y_test, y_pred_BRR)
mae_BRR = mean_absolute_error(y_test, y_pred_BRR)
r2_BRR = r2_score(y_test, y_pred_BRR)

print(f'Mean Squared Error (MSE) BRR model: {mse_BRR:.5g}')
print(f'Mean Absolute Error (MAE) BRR model: {mae_BRR:.5g}')
print(f'R^2 Score ENR model: {r2_BRR:.5g}')

metric_BRR = ["BRR", f"{mse_BRR:.5g}", f"{mae_BRR:.5g}", f"{r2_BRR:.5g}"]

metrics_models.loc[7] = metric_BRR

# Visualizing the results of the Bayesian Ridge Regression model

plt.scatter(y_test, y_pred_BRR, color = 'blue', edgecolor = 'w', \
            label = 'Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], \
         color = 'red', linestyle = '--', lw = 2, label = 'Perfect Fit')
plt.xlabel('Real Values')
plt.ylabel('BRR Predictions')
plt.title('Bayesian Ridge Regression Model (BRR) Predictions')
plt.legend()
plt.show()

index = np.arange(len(y_test))

plt.figure(figsize=(10, 6))
plt.plot(index, y_test, label = 'Test Set', color = 'red', marker = 'o')
plt.plot(index, y_pred_BRR, label = 'Predictions', color = 'blue', marker = 'x')
plt.ylim(y_test.min() - 1e+05, y_test.max() + 1e+05)
plt.xlabel('Index')
plt.ylabel('SalePrice')
plt.title('Bayesian Ridge Regression Model (BRR) Predictions')
plt.legend()
plt.show()

# Fit a Extremely Randomized Trees Regression (ERT) model with the Training set

from sklearn.ensemble import ExtraTreesRegressor

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

regressor_ERT = ExtraTreesRegressor(n_estimators = 400, random_state = 42)
regressor_ERT.fit(X_train_scaled, y_train.ravel())

y_pred_ERT = regressor_ERT.predict(X_test_scaled)

mse_ERT = mean_squared_error(y_test, y_pred_ERT)
mae_ERT = mean_absolute_error(y_test, y_pred_ERT)
r2_ERT = r2_score(y_test, y_pred_ERT)

print(f'Mean Squared Error (MSE) ERT model: {mse_ERT:.5g}')
print(f'Mean Absolute Error (MAE) ERT model: {mae_ERT:.5g}')
print(f'R^2 Score ERT model: {r2_ERT:.5g}')

metric_ERT = ["ERT", f"{mse_ERT:.5g}", f"{mae_ERT:.5g}", f"{r2_ERT:.5g}"]

metrics_models.loc[8] = metric_ERT

# Visualizing the results of the Bayesian Ridge Regression model

plt.scatter(y_test, y_pred_ERT, color = 'blue', edgecolor = 'w', \
            label = 'Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], \
         color = 'red', linestyle = '--', lw = 2, label = 'Perfect Fit')
plt.xlabel('Real Values')
plt.ylabel('ERT Predictions')
plt.title('Extremely Randomized Trees Regression Model (ERT) Predictions')
plt.legend()
plt.show()

index = np.arange(len(y_test))

plt.figure(figsize=(10, 6))
plt.plot(index, y_test, label = 'Test Set', color = 'red', marker = 'o')
plt.plot(index, y_pred_ERT, label = 'Predictions', color = 'blue', marker = 'x')
plt.ylim(y_test.min() - 1e+05, y_test.max() + 1e+05)
plt.xlabel('Index')
plt.ylabel('SalePrice')
plt.title('Extremely Randomized Trees Regression Model (ERT) Predictions')
plt.legend()
plt.show()