#REQUIRED LIBRARIES=>
import numpy as np
import pandas as pd

#IMPORTANT CODE SNIPPET TO PRINT THE ENTIRE DATASET:
#SHOWS ALL ROWS
pd.set_option('display.max_rows',None)
#SHOWS ALL COLUMNS
pd.set_option('display.max_columns',None)
#PREVENT LINE WRAPPING
pd.set_option('display.width',None)
#SHOW FULL CONTENT IN EACH CELL
pd.set_option('display.max_colwidth',None)

#READING THE DATASET:
flats = pd.read_csv('FLATS_DATA_PREPROCESSING/flats_cleaned.csv')
# print(flats.shape)
houses = pd.read_csv('HOUSES_DATA_PREPROCESSING/houses_cleaned.csv')
# print(houses.shape)

#CONCATINATING BOTH THA DATASETS:
df = pd.concat([flats,houses],ignore_index=True)
# print(df.head())

df = df.sample(df.shape[0],ignore_index=True)
# print(df.head())
print(df.shape)
df.to_csv('gurgaon_properties.csv',index=False)