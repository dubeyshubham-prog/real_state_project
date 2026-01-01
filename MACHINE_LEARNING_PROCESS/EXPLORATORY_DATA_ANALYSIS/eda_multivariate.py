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
df = pd.read_csv('gurgaon_properties_cleaned_v2.csv').drop_duplicates()
# print(df.head())
# print(df.shape)
# print(df.info())
#DROPING A COLUMN-->address
df.drop(columns=['address'], inplace=True)
# print(df.shape)

#COLUMN-->property_type vs price
'''
sns.barplot(x=df['property_type'],y=df['price'],estimator=np.median)
sns.boxplot(x=df['property_type'],y=df['price'])
'''

#COLUMN-->property_type vs area
'''
sns.barplot(x=df['property_type'], y=df['built_up_area'], estimator=np.median)
sns.boxplot(x=df['property_type'], y=df['built_up_area'])
#REMOVING THAT CRAZY OUTLIER:
df = df[df['built_up_area'] != 737147]
sns.boxplot(x=df['property_type'], y=df['built_up_area'])
'''

#COLUMN-->property_type vs price_per_sqft
'''
sns.barplot(x=df['property_type'], y=df['price_per_sqft'], estimator=np.median)
sns.boxplot(x=df['property_type'], y=df['price_per_sqft'])
'''

#COLUMN-->property_type vs bedRoom
'''
sns.heatmap(pd.crosstab(df['property_type'],df['bedRoom']))
#CHECKING FOR OUTLIERS
print(df[df['bedRoom'] >= 10].shape)
'''

#COLUMN-->property_type vs floorNum
'''
sns.barplot(x=df['property_type'],y=df['floorNum'])
sns.boxplot(x=df['property_type'],y=df['floorNum'])
#CHECKING FOR OUTLIERS:
print(df[(df['property_type'] == 'house') & (df['floorNum'] > 10)])
'''

#COLUMN-->property_type vs agePossession
# sns.heatmap(pd.crosstab(df['property_type'],df['agePossession']))
'''sns.heatmap(pd.pivot_table(df,index='property_type',
                           columns='agePossession',
                           values='price',
                           aggfunc='mean'),annot=True)

plt.figure(figsize=(15,4))
sns.heatmap(pd.pivot_table(df,index='property_type',
                           columns='bedRoom',
                           values='price',
                           aggfunc='mean'),annot=True)
'''

#COLUMN-->property_type vs luxury_score
'''
sns.barplot(x=df['property_type'],y=df['luxury_score'])
sns.boxplot(x=df['property_type'],y=df['luxury_score'])
'''

#COLUMN-->property_type vs sector analysis
'''
plt.figure(figsize=(15,6))
sns.heatmap(pd.crosstab(df['property_type'],df['sector'].sort_index()))
'''

#GROUP BY 'SECTOR' AND CALCULATE THE AVERAGE PRICE
avg_price_per_sector = df.groupby('sector')['price'].mean().reset_index()
# print(avg_price_per_sector)
# print(df[df['sector'] == 'dwarka expressway']['price'].mean())

#FUNCTION TO EXTRACT SECTOR NUMBER:
def extract_sector_number(sector_name):
    match = re.search(r'\d+', sector_name)
    if match:
        return int(match.group())
    else:
        return float('inf')#RETURN A LARGE NUMBER FOR NON-NUMBERED SECTORS

avg_price_per_sector['sector_number'] = avg_price_per_sector['sector'].apply(extract_sector_number)
#SORT BY SECTOR NUMBER:
avg_price_per_sector_sorted_by_sector = avg_price_per_sector.sort_values(by='sector_number')

#PLOT THE HEATMAP:
'''
plt.figure(figsize=(5, 25))
sns.heatmap(avg_price_per_sector_sorted_by_sector.set_index('sector')[['price']], annot=True, fmt=".2f", linewidths=.5)
plt.title('Average Price per Sector (Sorted by Sector Number)')
plt.xlabel('Average Price')
plt.ylabel('Sector')
plt.show()
'''

luxury_score = df.groupby('sector')['luxury_score'].mean().reset_index()

luxury_score['sector_number'] = luxury_score['sector'].apply(extract_sector_number)

# Sort by sector number
luxury_score_sector = luxury_score.sort_values(by='sector_number')

# Plot the heatmap
'''
plt.figure(figsize=(5, 25))
sns.heatmap(luxury_score_sector.set_index('sector')[['luxury_score']], annot=True, fmt=".2f", linewidths=.5)
plt.title('Sector (Sorted by Sector Number)')
plt.xlabel('Average Price per sqft')
plt.ylabel('Sector')
plt.show()
'''

#COLUMN-->price
'''
plt.figure(figsize=(12,8))
sns.scatterplot(x=df[df['area']<10000]['area'],y=df['price'],hue=df['bedRoom'])
plt.figure(figsize=(12,8))
sns.scatterplot(x=df[df['area']<10000]['area'],y=df['price'],hue=df['agePossession'])
'''
# sns.barplot(x=df['bedRoom'],y=df['price'],estimator=np.median)
sns.barplot(x=df['agePossession'],y=df['price'],estimator=np.median)
plt.xticks(rotation='vertical')
plt.show()












plt.show()