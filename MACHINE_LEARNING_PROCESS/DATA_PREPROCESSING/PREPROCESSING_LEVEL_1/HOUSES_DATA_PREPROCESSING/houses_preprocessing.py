#ALL REQUIRED LIBRARIES=>
import numpy as pandas
import pandas as pd
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

#READING DATA IN PANDAS FILE:
df = pd.read_csv('houses.csv')

#OBSERVING THE DATASET:
# print(df.head())
# print(df.shape)
# print(df.info())

#EXPLORATORY DATA ANALYSIS=>
#CHECK FOR DUPLICATES:
# print(df.duplicated().sum())
df = df.drop_duplicates()
# print(df.shape)

#COLUMNS TO DROP-->property_name,link,property_id
df.drop(columns=['link','property_id'],inplace=True)
# print(df.shape)

#RENAMING COLUMNS FROM rate-->price_per_sqft
df.rename(columns={'rate':'price_per_sqft'},inplace=True)
# print(df.head())

#UNIVARIATE ANALYSIS:
#COLUMN-->society
# print(df['society'].value_counts().shape)
df['society'] = df['society'].apply(lambda name: re.sub(r'\d+(\.\d+)?\s?★', '', str(name)).strip()).str.lower()
# print(df['society'].value_counts().shape)
# print(df['society'].value_counts().shape)
df['society'] = df['society'].str.replace('nan','independent')

#COLUMN-->PRICE:
# print(df['price'].value_counts())
# print(df['price'].unique())
df = df[df['price']!='Price on Request']
# print(df.shape)
#THIS FUNCTION CREATE A STANDARD UNIT FOR PRICE COL:
def treat_price(x):
    if type(x)==float:
        return x
    else:
        if x[1]=='Lac':
            return round(float(x[0])/100,2)
        else:
            return float(x[0])
df['price'] = df['price'].str.split(' ').apply(treat_price)
# print(df.head())

#COLUMN-->price_per_sqft
df['price_per_sqft'].value_counts()
df['price_per_sqft'] = df['price_per_sqft'].str.split('/').str.get(0).str.replace('₹','').str.replace(',','').str.strip().astype('float')
# print(df.head())

#COLUMN-->bedroom
# print(df['bedRoom'].value_counts())
# print(df[df['bedRoom'].isnull()].shape)
df = df[~df['bedRoom'].isnull()]
df['bedRoom'] = df['bedRoom'].str.split(' ').str.get(0).astype('int')
# print(df.head())

#COLUMN-->bathrooms
# print(df['bathroom'].value_counts())
# print(df['bathroom'].isnull().sum())
df['bathroom'] = df['bathroom'].str.split(' ').str.get(0).astype('int')
# print(df.head())

#COLUMN-->balcony
# print(df['balcony'].value_counts())
df['balcony'] = df['balcony'].str.split(' ').str.get(0).str.replace('No','0')
# print(df.head())

#COLUMN-->additionalRoom
# print(df['additionalRoom'].value_counts())
# print(df['additionalRoom'].isnull().sum())
df['additionalRoom'].fillna('not available', inplace=True)
df['additionalRoom'] = df['additionalRoom'].str.lower()
# print(df.head())

#COLUMN-->floor
# print(df['noOfFloor'].value_counts())
# print(df['noOfFloor'].isnull().sum())
df['noOfFloor'] = df['noOfFloor'].str.split(' ').str.get(0)
df.rename(columns={'noOfFloor':'floorNum'},inplace=True)
# print(df.head())

#COLUMN-->facing
# print(df['facing'].isnull().sum())
df['facing'].fillna('NA',inplace=True)

#COLUMN-->area
df['area'] = round((df['price']*10000000)/df['price_per_sqft'])

#INSERTING A NEW COLUMN CALLED property_type
df.insert(loc=1,column='property_type',value='house')
print(df.head())

#SAVING THE DATA AS A CSV FILE NAMED:houses_cleaned
df.to_csv('houses_cleaned.csv', index=False)