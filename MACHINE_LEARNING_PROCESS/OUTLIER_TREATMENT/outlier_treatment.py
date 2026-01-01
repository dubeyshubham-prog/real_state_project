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
df = pd.read_csv('gurgaon_properties_cleaned_v2.csv')
# print(df.head())
# print(df.shape)
df = df.reset_index(drop=True)

#OUTLIER ON THE BASIS OF PRICE COLUMN:
# sns.displot(df['price'],kde=True)
# sns.boxplot(x=df['price'])

#CALCULATE THE IQR FOR THE PRICE COLUMN:
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

#DEFINE BOUNDS FOR THE OUTLIER:
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR
#IDENTIFY OUTLIERS:
outliers = df[(df['price']<lower_bound) | (df['price']>upper_bound)]

#DISPLAYING THE NUMBER OF OUTLIERS AND SOME STATISTICS:
# num_outliers = outliers.shape
# outlier_price_status = outliers['price'].describe()
# print(num_outliers, outlier_price_status)
# print(outliers.sort_values('price', ascending=False).head(20))

'''
OBSERVATION:
        ON THE BASIS OF PRICE COL WE CAN SAY THAT THERE ARE SOME GENUINE OUTLIERS BUT THERE
        ARE SOME DATA ERRORS AS WELL
'''

#COLUMN-->Price_per_sqft
# sns.displot(df['price_per_sqft'], kde=True)
# sns.boxplot(x= df['price_per_sqft'])

#CALCULATE IQR FOR price_per_sqft COLUMN:
Q1 = df['price_per_sqft'].quantile(0.25)
Q3 = df['price_per_sqft'].quantile(0.75)
IQR = Q3 - Q1

#DEFINE BOUNDS FOR OUTLIERS:
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

#IDENTIFY OUTLIERS
outliers_sqft = df[(df['price_per_sqft'] < lower_bound) | (df['price_per_sqft'] > upper_bound)]

#DISPLAYING THE NUMBER OF OUTLIERS AND SOME STATISTICS:
num_outliers = outliers_sqft.shape[0]
outliers_sqft_stats = outliers_sqft['price_per_sqft'].describe()
# print(num_outliers, outliers_sqft_stats)

outliers_sqft['area'] = outliers_sqft['area'].apply(lambda x:x*9 if x<1000 else x)
outliers_sqft['price_per_sqft'] = round((outliers_sqft['price']*10000000)/outliers_sqft['area'])
# print(outliers_sqft['price_per_sqft'].describe())
df.update(outliers_sqft)
# sns.boxplot(x=df['price_per_sqft'])
df = df[df['price_per_sqft'] <= 50000]
# sns.boxplot(x=df['price_per_sqft'])

#COLUMN-->area:
# sns.distplot(df['area'])
# sns.boxplot(x=df['area'])
# print(df['area'].describe())
# print(df[df['area'] > 100000])
df = df[df['area'] < 100000]
# sns.distplot(df['area'])
# sns.boxplot(x=df['area'])

# print(df[df['area'] > 10000].sort_values('area',ascending=False))
#2679,3488,181,3430,1144,1324,3071,2513,853,589,495,545,1474,1842,3061,
#1838,3135,1103,2791,1157,
df.drop(index=[2679,3488,181,3430,1144,1324,3071,2513,853,589,495,545,1474,1842,3061,1838,3135,1103,], inplace=True)
# print(df.shape)
# sns.distplot(df['area'])
# sns.boxplot(x=df['area'])
# print(df['area'].describe())

#COLUMN-->Bedroom
# sns.distplot(df['bedRoom'])
# sns.boxplot(x=df['bedRoom'])
# print(df['bedRoom'].describe())
# print(df[df['bedRoom']>10].sort_values('bedRoom', ascending=False))
df = df[df['bedRoom'] <= 10]
# print(df.shape)
# sns.distplot(df['bedRoom'])
# sns.boxplot(x=df['bedRoom'])
# print(df['bedRoom'].describe())

#COLUMN-->Bathroom
# sns.distplot(df['bathroom'])
# sns.boxplot(x=df['bathroom'])
# print(df[df['bathroom'] > 10].sort_values('bathroom',ascending=False))

#COLUMN-->super built up area
# sns.distplot(df['super_built_up_area'])
# sns.boxplot(x=df['super_built_up_area'])
# print(df['super_built_up_area'].describe())

#COLUMN-->built up area
# sns.distplot(df['built_up_area'])
# sns.boxplot(x=df['built_up_area'])
# print(df[df['built_up_area'] > 10000])

#COLUMN-->carpet area
# sns.distplot(df['carpet_area'])
# sns.boxplot(x=df['carpet_area'])

#COLUMN-->luxury_score
# sns.distplot(df['luxury_score'])
# sns.boxplot(x=df['luxury_score'])

df['price_per_sqft'] = round((df['price']*10000000)/df['area'])
# sns.distplot(df['price_per_sqft'])

#COLUMN-->area and bedRoom RELATION
x = df[df['price_per_sqft'] <= 20000]
# print((x['area']/x['bedRoom']).quantile(0.02))

# print(df[(df['area']/df['bedRoom'])<193])

# sns.lmplot(data=df, x='area', y='bedRoom')

#FINDING AREA AND BEDROOM RATIO:
df['area_room_ratio'] = df['area']/df['bedRoom']
# print(df.shape)
# print((df[df['area_room_ratio']<250])['bedRoom'].value_counts())
df = df[df['area_room_ratio']>100]
# print(df.shape)

#PERFORMING A TRANSFORMATION ON OUTLIERS:
outliers_df = df[(df['area_room_ratio']<250) & (df['bedRoom']>3)]
# print(outliers_df[['area_room_ratio', 'bedRoom']])

outliers_df['bedRoom'] = round(outliers_df['bedRoom']/outliers_df['floorNum'])

#UPDATING THE df DATASET:
df.update(outliers_df)

df['area_room_ratio'] = df['area']/df['bedRoom']
# print(df[(df['area_room_ratio']<250) & (df['bedRoom']>4)].shape)
#REMOVE THERE DATASETS FROM THE DATA:
print(df.shape)
df = df[~((df['area_room_ratio']<250) & (df['bedRoom']>4))]
print(df[(df['area_room_ratio']<250) & (df['bedRoom']>4)].shape)

df.to_csv('gurgaon_properties_outlier_treated.csv', index=False)



















plt.show()