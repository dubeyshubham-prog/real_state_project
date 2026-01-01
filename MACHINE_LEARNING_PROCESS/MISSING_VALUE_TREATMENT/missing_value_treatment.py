#REQUIRED LIBRARIES=>
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import re

#IMPORTANT CODE SNIPPET TO PRINT THE ENTIRE DATASET:
#SHOWS ALL ROWS
pd.set_option('display.max_rows',None)
#SHOWS ALL COLUMNS
pd.set_option('display.max_columns',None)
#PREVENT LINE WRAPPING
pd.set_option('display.width',None)
#SHOW FULL CONTENT IN EACH CELL
pd.set_option('display.max_colwidth',None)

#IMPORTANT CODE SNIPPET TO IGNORE THE WARNINGS:
warnings.filterwarnings('ignore')

#READING THE DATASET:
df = pd.read_csv('gurgaon_properties_outlier_treated.csv')
# print(df.isnull().sum())
# print(df.head())
# print(df.shape)

#COLUMN-->built_up_area
# sns.scatterplot(x=df['built_up_area'],y=df['super_built_up_area'])
# sns.scatterplot(x=df['built_up_area'],y=df['carpet_area'])
# print(((df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull())))

all_present_df = df[~((df['super_built_up_area'].isnull()) | (df['built_up_area'].isnull()) | (df['carpet_area'].isnull()))]
# print(all_present_df.shape)

#CALCULATING RATIO'S
super_to_built_up_ratio = (all_present_df['super_built_up_area']/all_present_df['built_up_area']).median()
carpet_to_built_up_ratio = (all_present_df['carpet_area']/all_present_df['built_up_area']).median()
# print(super_to_built_up_ratio, carpet_to_built_up_ratio)

#BOTH PRESENT built_up_area NULL:
sbc_df = df[~(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & ~(df['carpet_area'].isnull())]
# print(sbc_df.head())
# print(sbc_df.shape)
sbc_df['built_up_area'].fillna(round(((sbc_df['super_built_up_area']/1.105) + (sbc_df['carpet_area']/0.9))/2),inplace=True)
df.update(sbc_df)
# print(df.isnull().sum())

#ONLY super_built_up_area PRESENT BOTH NULL:
sb_df = df[~(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull())]
sb_df['built_up_area'].fillna(round(sb_df['super_built_up_area']/1.105),inplace=True)
df.update(sb_df)
# print(df.isnull().sum())

#ONLY carpet_area IS PRESENT BOTH NULL:
c_df = df[(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & ~(df['carpet_area'].isnull())]
c_df['built_up_area'].fillna(round(c_df['carpet_area']/0.9),inplace=True)
df.update(c_df)
# print(df.isnull().sum())
# sns.scatterplot(x=df['built_up_area'],y=df['price'])

anamoly_df = df[(df['built_up_area'] < 2000) & (df['price'] > 2.5)][['price','area','built_up_area']]
# print(anamoly_df.sample(5))

#DEROP UNNECESSARY COLUMNS FROM THE DATASET:
df.drop(columns=['area','areaWithType','super_built_up_area','carpet_area','area_room_ratio'],inplace=True)
# print(df.shape)
# print(df.isnull().sum())

#COLUMN-->floorNum
# print(df[df['floorNum'].isnull()])
# print(df[df['property_type']=='house']['floorNum'].median())
df['floorNum'].fillna(0.2, inplace=True)

#COLUMN-->facing
'''
NOTE:
    THE MISSING DATA IS TOO MUCH AND NOW DROPPING THIS COLUMN
    IS THE BEST POSSIBLE WAY TO DEAL WITH THIS COLUMN BECAUSE 
    IT IS NOT THAT USEFUL.OUT PROJECT 
'''
df.drop(columns=['facing'],inplace=True)
# print(df.shape)
# print(df.isnull().sum())

#COLUMN-->society
#ONLY ONE MISSING DATA THERE DROPPING THAT PARTICULAR INDEX
df.drop(index=[567],inplace=True)
# print(df.isnull().sum())
# print(df.shape)

#COLUMN-->agePossession
# print(df['agePossession'].value_counts())
# print(df[df['agePossession'] == 'Undefined'])

#CREATING A FUNCTION TO FILL THE UNDEFINED ROWS:
def mode_based_imputation(row):
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['sector'] == row['sector']) & (df['property_type'] == row['property_type'])]['agePossession'].mode()
        # If mode_value is empty (no mode found), return NaN, otherwise return the mode
        if not mode_value.empty:
            return mode_value.iloc[0]
        else:
            return np.nan
    else:
        return row['agePossession']

df['agePossession'] = df.apply(mode_based_imputation,axis=1)

def mode_based_imputation2(row):
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['sector'] == row['sector'])]['agePossession'].mode()
        # If mode_value is empty (no mode found), return NaN, otherwise return the mode
        if not mode_value.empty:
            return mode_value.iloc[0]
        else:
            return np.nan
    else:
        return row['agePossession']
df['agePossession'] = df.apply(mode_based_imputation2,axis=1)

def mode_based_imputation3(row):
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['property_type'] == row['property_type'])]['agePossession'].mode()
        # If mode_value is empty (no mode found), return NaN, otherwise return the mode
        if not mode_value.empty:
            return mode_value.iloc[0]
        else:
            return np.nan
    else:
        return row['agePossession']
df['agePossession'] = df.apply(mode_based_imputation3,axis=1)
print(df.isnull().sum())

#SAVING THE DATA AS A CSV FILE:
df.to_csv('gurgaon_properties_missing_value_imputation.csv',index=False)
print(df.shape)



















plt.show()