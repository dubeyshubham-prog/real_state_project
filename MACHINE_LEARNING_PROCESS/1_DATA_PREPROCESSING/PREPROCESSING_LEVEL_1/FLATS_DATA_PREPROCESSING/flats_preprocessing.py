#ALL REQUIRED LIBRARIES=>
import pandas as pd
import numpy as np
import re
import warnings

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
df = pd.read_csv('flats.csv')
# print(df.shape)

#EXPLORATORY DATA ANALYSIS IS BEEN DONE HERE=>
#OBSERVING THE DATASET:
'''
print(df.head())
print(df.shape)
print(df.info())
print(df.duplicated().sum())
print(df.isnull().sum())
'''

#COLUMNS TO BE DROP:-->'property_name','link','property_id'
df.drop(columns = ['link','property_id'],inplace=True)
# print(df.shape)

#RENAMING THE AREA COLUMN:
df.rename(columns={'area':'price_per_sqft'},inplace=True)
# print(df.head())


#UNIVARIATE DATA ANALYSIS=>
#COLUMN-->SOCIETY:
# print(df['society'].value_counts())
# print(df['society'].value_counts().shape)
# print(df.head())
df['society'] = df['society'].apply(lambda name: re.sub(r'\d+(\.\d+)?\s?★',
                                                        '',
                                                             str(name)).strip()).str.lower()
# print(df['society'].value_counts().shape)
# print(df.head())
# print(df.shape)

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

#COLUMN-->'price_per_sqft'
# print(df['price_per_sqft'].value_counts())
# print(df['price_per_sqft'].head())
df['price_per_sqft'] = df['price_per_sqft'].str.split('/').str.get(0).str.replace('₹','').str.replace(',','').str.strip().astype('float')
# print(df.head())

#COLUMN-->bedRoom
# print(df['bedRoom'].value_counts())
# print(df['bedRoom'].isnull().sum())
df['bedRoom'] = df['bedRoom'].str.split(' ').str.get(0).astype('int')

#COLUMN-->bathroom
# print(df['bathroom'].value_counts())
# print(df['bathroom'].isnull().sum())
df['bathroom'] = df['bathroom'].str.split(' ').str.get(0).astype('int')

#COLUMN-->balcony
# print(df['balcony'].value_counts())
df['balcony'] = df['balcony'].str.split(' ').str.get(0).str.replace('No','0')

#COLUMN-->additionalRoom
# print(df['additionalRoom'].value_counts())
# print(df['additionalRoom'].isnull().sum())
df['additionalRoom'].fillna('Not Available',inplace=True)
# print(df['additionalRoom'].isnull().sum())
df['additionalRoom'] = df['additionalRoom'].str.lower()
#print(df.head())

#COLUMN-->floorNum
# print(df['floorNum'].value_counts())
# print(df['floorNum'].isnull().sum())
df['floorNum'] = df['floorNum'].str.split(' ').str.get(0).str.replace('Ground','0').str.replace('Basement','-1').str.replace('Lower','0').str.extract(r'(\d+)')
# print(df['floorNum'])
# print(df.head())

#COLUMN-->facing
print(df['facing'].value_counts())
# print(df['facing'].isnull().sum())
df.fillna('NA',inplace=True)
# print(df['facing'].isnull().sum())

#INSERTING A NEW COLUMN AREA AT LOCATION 4:
print(df.columns)
df['price']=pd.to_numeric(df['price'],errors='coerce')
df['price_per_sqft']=pd.to_numeric(df['price_per_sqft'],errors='coerce')
df.insert(loc=4,column='area',value=round((df['price']*10000000)/df['price_per_sqft']))
df.insert(loc=1,column='property_type',value='flat')
print(df.head())

#SAVING THE CLEANED DATA AS CSV FILE=>
df.to_csv('flats_cleaned.csv',index=False)
