#default import on Kaggle
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax   
from sklearn.linear_model import LinearRegression, ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegresso
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
import os
warnings.filterwarnings('ignore')


#import data
train_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

#concat data
train_ID = train_data["Id"]
test_ID = test_data['Id']

train_data.drop(['Id'], axis=1, inplace=True)
test_data.drop(['Id'], axis=1, inplace=True)

y = train_data['SalePrice'].reset_index(drop=True)
train_merge = train_data.drop(['SalePrice'], axis=1)
data_all = pd.concat((train_merge, test_data), 
                     sort=False).reset_index(drop=True)

# numeric feature describe
num_data=data_all.select_dtypes(['int64','float64'])
des=data_all.describe()
    
# KDE
for i in list(num_data.columns):
    sns.distplot(num_data[i].dropna()).set_title(i)
    plt.show()

#select categorical data
cat_data=data_all.select_dtypes(['object'])

##################################################################

#Remove outliers

##################################################################

numerical_cols=num_data.columns.tolist()
categorical_cols=cat_data.columns.tolist()

print('Numarical Data：',len(numerical_cols))
print('Categorical Data：',len(categorical_cols))


#collect numnerical data to trensfer to categorical data
transform_cols = []

for col in numerical_cols:
    if len(data_all[col].unique())<20:
        transform_cols.append(col)
        
#Transfer features - 'MSSubClass'、'YrSold'、'MoSold' to string
data_all['MSSubClass'] = data_all['MSSubClass'].apply(str)
data_all['YrSold'] = data_all['YrSold'].astype(str)
data_all['MoSold'] = data_all['MoSold'].astype(str)

#Add to related data type
numerical_cols.remove('MSSubClass')
numerical_cols.remove('YrSold')
numerical_cols.remove('MoSold')
categorical_cols.append('MSSubClass')
categorical_cols.append('YrSold')
categorical_cols.append('MoSold')

#draw boxplot for numerical data
fig = plt.figure(figsize=(80,60),dpi=120)
for i in range(len(numerical_cols)):
    plt.subplot(6, 6, i+1)
    sns.boxplot(train_data[numerical_cols[i]], orient='v', width=0.5)
    plt.ylabel(numerical_cols[i], fontsize=36)
plt.show()
fig.savefig('1.png')

#draw bar chart for categorical data
for i in range(len(categorical_cols)):
    fig,ax=plt.subplots(1,1) 
    ax.hist(train_data[categorical_cols[i]].dropna())
    ax.set_title(categorical_cols[i],fontsize=12,color='Black')
    plt.show()
    
#the relationship between GrLivArea and SalePrice
fig = plt.figure(figsize=(6,5))
plt.axvline(x=4600, color='r', linestyle='--')
sns.scatterplot(x='GrLivArea',y='SalePrice',data=train_data, alpha=0.6)
train_data.GrLivArea.sort_values(ascending=False)[:2]

#the relationship between LotArea and SalePrice
fig = plt.figure(figsize=(6,5))
#plt.axvline(x=200000, color='r', linestyle='--')
sns.scatterplot(x='LotArea',y='SalePrice',data=train_data, alpha=0.6)
train_data['LotArea'].sort_values(ascending=False)[:3]

#the relationship between TotalBsmtSF and SalePrice
fig = plt.figure(figsize=(6,5))
#plt.axvline(x=200000, color='r', linestyle='--')
sns.scatterplot(x='TotalBsmtSF',y='SalePrice',data=train_data, alpha=0.6)
train_data['TotalBsmtSF'].sort_values(ascending=False)[:3]

#the relationship between 1stFlrSF and SalePrice
fig = plt.figure(figsize=(6,5))
#plt.axvline(x=200000, color='r', linestyle='--')
sns.scatterplot(x='1stFlrSF',y='SalePrice',data=train_data, alpha=0.6)
train_data['1stFlrSF'].sort_values(ascending=False)[:3]

#the relationship between MasVnrArea and SalePrice
fig = plt.figure(figsize=(6,5))
#plt.axvline(x=1500, color='r', linestyle='--')
sns.scatterplot(x='MasVnrArea',y='SalePrice',data=train_data, alpha=0.6)

#the relationship between LotFrontage and SalePrice
fig = plt.figure(figsize=(6,5))
#plt.axvline(x=1500, color='r', linestyle='--')
sns.scatterplot(x='LotFrontage',y='SalePrice',data=train_data, alpha=0.6)
train_data['LotFrontage'].sort_values(ascending=False)[:3]

#the relationship between BsmtFinSF1 and SalePrice
fig = plt.figure(figsize=(6,5))
#plt.axvline(x=1500, color='r', linestyle='--')
sns.scatterplot(x='BsmtFinSF1',y='SalePrice',data=train_data, alpha=0.6)
train_data['BsmtFinSF1'].sort_values(ascending=False)[:3]

#Exclude outlier and reorder the dataset
train_data = train_data[train_data.GrLivArea < 4600]
#train_data = train_data[train_data['1stFlrSF'] < 4000]
train_data.reset_index(drop=True, inplace=True)
train_data.shape

######################################################

#missing value

######################################################

(data_all[categorical_cols].isna().sum()/data_all.shape[0]).sort_values(ascending=False)[:25]

for col in ('PoolQC', 'MiscFeature','Alley', 'Fence', 'FireplaceQu', 'MasVnrType', 'Utilities',
            'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType', 
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    data_all[col] = data_all[col].fillna('None')
    
(data_all[categorical_cols].isna().sum()/data_all.shape[0]).sort_values(ascending=False)[:10]

#Use mode to fill the rest categorical data
for col in ('Functional', 'SaleType', 'Electrical', 'Exterior2nd', 'Exterior1st', 'KitchenQual','MSZoning'):
    data_all[col] = data_all[col].fillna(data_all[col].mode()[0])

(data_all[categorical_cols].isna().sum()/data_all.shape[0]).sort_values(ascending=False)[:3]

#cleaning numerical data
(data_all[numerical_cols].isna().sum()/data_all.shape[0]).sort_values(ascending=False)[:12]

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea',
            'BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF'):
    data_all[col] = data_all[col].fillna(0)
    
(data_all[numerical_cols].isna().sum()/data_all.shape[0]).sort_values(ascending=False)[:3]
data_all['LotFrontage'] = data_all['LotFrontage'].fillna(data_all['LotFrontage'].median())

##################################################

#Transfer to logarithmic form (Due to RMSE)

##################################################

#draw Q-Q plot and histogram for 'SalePrice'
from scipy import stats
plt.figure(figsize=(10,5))
ax_1 = plt.subplot(1,2,1)
sns.distplot(train_data["SalePrice"],fit=stats.norm)
ax_2 = plt.subplot(1,2,2)
res = stats.probplot(train_data["SalePrice"],plot=plt)

train_data["SalePrice"] = np.log1p(train_data["SalePrice"])
plt.figure(figsize=(10,5))
ax_1 = plt.subplot(1,2,1)
sns.distplot(train_data["SalePrice"],fit=stats.norm)
ax_2 = plt.subplot(1,2,2)
res = stats.probplot(train_data["SalePrice"],plot=plt)
