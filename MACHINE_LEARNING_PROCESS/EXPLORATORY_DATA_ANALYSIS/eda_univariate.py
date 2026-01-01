#REQUIRED LIBRARIES=>
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
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

#READING THE DATASET:
df = pd.read_csv('gurgaon_properties_cleaned_v2.csv')
# print(df.head())
# print(df.shape)
# print(df.info())
#DROPING A COLUMN-->address
df.drop(columns=['address'], inplace=True)
# print(df.shape)

#SMALL PREPROCESSING ON THE DATASET:
# print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
# print(df.duplicated().sum())

#COLUMN-->property_type
# df['property_type'].value_counts().plot(kind='bar')
'''
OBSERVATION:
            1.FLATS ARE IN MAJORITY(75 PERCENT) AND THERE ARE LESS NUMBER OF HOUSES(~25 PERCENT)
            2.NO MISSING VALUES
'''

#COLUMN-->society
# print(df['society'].value_counts().sum())
# print(df['society'].info())
# print(df['society'].isnull().sum())
# print(df['society'].value_counts())

df[df['society'] != 'independent']['society'].value_counts(normalize=True).cumsum().head(75)

society_counts = df['society'].value_counts()
# print(society_counts)
#FREQUENCY DISTRIBUTION FOR SOCIETY:
frequency_bins = {
    "Very High (>100)": (society_counts > 100).sum(),
    "High (50-100)": ((society_counts >= 50) & (society_counts <= 100)).sum(),
    "Average (10-49)": ((society_counts >= 10) & (society_counts < 50)).sum(),
    "Low (2-9)": ((society_counts > 1) & (society_counts < 10)).sum(),
    "Very Low (1)": (society_counts == 1).sum()
}
# print(frequency_bins)
# df[df['society']!='independent']['society'].value_counts().head(10).plot(kind='bar')
'''
OBSERVATION:
            1.AROUND 13% PROPERTIES COMES UNDER INDEPENDENT TAG.
            2.THERE ARE 675 SOCIETIES.
            3.THE TOP 75 SOCIETIES HAVE 50% OF THE PROPERTIES AND THE REST 50% OF THE 
            PROPERTIES COME UNDER THE REMAINING 600 SOCIETIES 
            4.VERY HIGH(>100): ONLY 1 SOCIETY HAS MORE THAN 100 LISTINGS.
            5.HIGH(50-100):2 SOCIETIES HAVE BETWEEN 50 TO 100 LISTINGS.
            6.AVERAGE(10-49):92 SOCIETIES FALL IN THIS RANGE WITH 10 TO 49 LISTINGS EACH.
            7.LOW(2-9): 273 SOCIETIES HAVE BETWEEN 2 TO 9 LISTINGS.
            8.VERY LOW(1):A SIGNIFICANT NUMBER, 308 SOCIETIES, HAVE ONLY 1 LISTING.
            9.1 MISSING VALUES
'''

#COLUMN-->sector
# print(df['sector'].value_counts())
#TOP 10 SECTORS:
# df['sector'].value_counts().head(10).plot(kind='bar')

#FREQUENCY DISTRIBUTION FOR SECTOR COUNT:
sector_counts = df['sector'].value_counts()

frequency_bin = {
    "Very High (>100)": (sector_counts > 100).sum(),
    "High (50-100)": ((sector_counts >= 50) & (sector_counts <= 100)).sum(),
    "Average (10-49)": ((sector_counts >= 10) & (sector_counts < 50)).sum(),
    "Low (2-9)": ((sector_counts > 1) & (sector_counts < 10)).sum(),
    "Very Low (1)": (sector_counts == 1).sum()
}
# print(frequency_bin)
# print(df['sector'].nunique())
'''
OBSERVATION:
            1.THERE ARE TOTAL OF 115 UNIQUE sectors IN THE DATASET.
            2.FREQUENCY DISTRIBUTION SECTOR.
            3.Very High (>100):3 SECTORS HAVE MORE THAT HUNDRED LISTINGS
            4.High (50-100):25 SECTORS HAVE BETWEEN 50 TO 100 LISTINGS
            5.Average (10-49):A MAJORITY, 63 SECTORS, FALL IN THIS RANGE WITH 10 TO 49 LISTINGS EACH.
            6.Low (2-9):23 SECTORS HAVE BETWEEN 2 TO 9 LISTINGS
            7.Very Low (1):INTERESTINGLY, THERE ARE NO SECTORS WITH ONLY 1 LISTINGS 
'''

#COLUMN-->price
# print(df['price'].isnull().sum())
# print(df['price'].describe())
# sns.histplot(df['price'], kde=True, bins=50)
# sns.boxplot(x= df['price'], color='lightgreen')
# plt.grid()
'''
OBSERVATION:
            DESCRIPTIVE STATISTICS:
                    COUNT:THERE ARE 3,660 NON-MISSING PRICE ENTRIES.
                    MEAN PRICE:THE AVERAGE PRICE IS APPROXIMATELY 2.53 CRORES.
                    MEDIAN PRICE:THE MEDIAN(OR 50TH PERCENTILE) PRICE IS 1.52 CRORES.
                    STANDARD DEVIATION:THE PRICES HAVE A STANDARD DEVIATION OF 2.98, INDICATING VARIABILITY
                    IN THE PRICES.
                    RANGE:PRICES RANGE FROM A MINIMUM OF 0.07 CRORES TO A MAXIMUM OF 31.5 CRORES.
                    IQR:THE INTER QUARTILE RANGE(DIFFERENCE BETWEEN 75TH AND 25TH PERCENTILE) IS FROM 0.95
                    CRORES TO 2.75 CRORES.
            
            VISUALIZATION:
                    DISTRIBUTION:THE HISTOGRAM INDICATES THAT MOST PROPERTIES ARE PRICED IN THE LOWER
                    RANGE (BELOW 5 CRORES), WITH A FEW PROPERTIES GOING BEYOND 10 CRORES.
                    BOX PLOT:THE BOX PLOT SHOWCASES THE SPREAD OF THE DATA AND POTENTIAL OUTLIERS.
                    PROPERTIES PRICED ABOVE APPROXIMATELY 10 CRORES MIGHT BE CONSIDERED OUTLIERS AS
                    THEY LIE BEYOND THE UPPER WHISKER OF THE BOX PLOT.
                    MISSING VALUES:THERE ARE 17 MISSING VALUES IN THE PRICE COLUMN.
'''

#SKEWNESS AND KURTOSIS:
# skewness = df['price'].skew()
# kurtosis = df['price'].kurt()
# print(skewness, kurtosis)
'''
OBSERVATION:
        SKEWNESS:
            THE PRICE DISTRIBUTION HAS A SKEWNESS OF APPROXIMATELY 3.28,
            INDICATING A POSITIVE SKEW. THIS MEANS THAT THE DISTRIBUTION
            TAIL IS SKEWED TO THE RIGHT, WHICH ALIGN WITH OUR OBSERVATION
            FROM THE HISTOGRAM WHERE MOST PROPERTIES HAVE PRICES ON THE LOWER
            END WITH A FEW HIGH-PRICED PROPERTIES.
        
        KURTOSIS:
            THE KURTOSIS VALUE IS APPROXIMATELY 14.93. A KURTOSIS VALUE GREATER
            THAN 3 INDICATES A DISTRIBUTION WITH HEAVIER TAILS AND MORE OUTLIERS
            COMPARED TO A NORMAL DISTRIBUTION.
'''

#QUANTILE ANALYSIS:
quantiles = df['price'].quantile([0.01, 0.05, 0.95, 0.99])
# print(quantiles)

'''
OBSERVATION:
        1% QUANTILE: ONLY 1% OF THE PROPERTIES ARE PRICED BELOW 0.25 CRORES.
        5% QUANTILE: ONLY 5% OF THE PROPERTIES ARE PRICED BELOW 0.37 CRORES.
        95% QUANTILE: 95% OF THE PROPERTIES ARE PRICED BELOW 8.5 CRORES.
        99% QUANTILE: 99% OF THE PROPERTIES ARE PRICED BELOW 15.26 CRORES,
        INDICATING THAT VERY FEW PROPERTIES ARE PRICED ABOVE THIS VALUE.
'''

#IDENTIFYING THE POTENTIAL OUTLIERS USING IQR METHOD:
Q1 = df['price'].describe()['25%']
Q3 = df['price'].describe()['75%']

IQR = Q3-Q1
# print(IQR)
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR

# print(lower_bound, upper_bound)

outliers = df[(df['price']<lower_bound) | (df['price']>upper_bound)]
# print(outliers.shape)
# print(outliers['price'].describe())
'''
OBSERVATION:
        1.BASED ON THE IQR METHOD, THERE ARE 425 PROPERTIES CONSIDERED AS OUTLIERS.
        2.THESE OUTLIERS HAVE AN AVERAGE PRICE OF APPROXIMATELY 9.24 CRORES.
        3.THE RANGE FOR THESE OUTLIERS IS FROM 5.46 CRORES TO 31.5 CRORES.
'''

#PRICE BINNING:
bins = [0, 1, 2, 3, 5, 10, 20, 50]
bin_labels = ["0-1", "1-2", "2-3", "3-5", "5-10", "10-20", "20-50"]
# pd.cut(df['price'], bins=bins, labels=bin_labels, right=False).value_counts().plot(kind='bar')
'''
OBSERVATION:
        1.THE MAJORITY OF PROPERTIES ARE PRICED IN THE "1-2 CRORES" AND "2-3 CRORES" RANGES.
        2.THERE'S A SIGNIFICANT DROP IN THE NUMBER OF PROPERTIES PRICED ABOVE "5 CRORES."
'''

#ECDF PLOT:
'''
ecdf = df['price'].value_counts().sort_index().cumsum() / len(df['price'])
plt.plot(ecdf.index, ecdf, marker='.', linestyle='none')
plt.grid()
plt.figure(figsize=(15,6))
'''

#DISTRIBUTION PLOT WITHOUT LOG TRANSFORMATION:
'''
plt.subplot(1,2,1)
sns.histplot(df['price'], kde=True, bins=50, color='skyblue')
plt.title("Distribution of the price(Original)")
plt.xlabel("Log(Price)")
plt.ylabel("Frequency")
'''

#DISTRIBUTION PLOT WITH LOG TRANSFORMATION
'''
plt.subplot(1, 2, 2)
sns.histplot(np.log1p(df['price']), kde=True, bins=50, color='lightgreen')
plt.title('Distribution of Prices (Log Transformed)')
plt.xlabel('Log(Price)')
plt.ylabel('Frequency')
plt.tight_layout()
'''

'''
OBSERVATION:
        1.NP.LOG1P(X): THIS FUNCTION COMPUTES THE NATURAL LOGARITHM OF 1+X.
         IT'S DESIGNED TO PROVIDE MORE ACCURATE RESULTS FOR VALUES OF X THAT 
         ARE VERY CLOSE TO ZERO.

        2.USING NP.LOG1P HELPS IN TRANSFORMING THE PRICE COLUMN WHILE ENSURING
         THAT ANY VALUE (INCLUDING ZERO, IF PRESENT) IS HANDLED APPROPRIATELY.
         WHEN WE NEED TO REVERSE THE TRANSFORMATION, WE CAN USE NP.EXPM1 WHICH
         COMPUTES E^X − 1.
'''

skewness = np.log1p(df['price']).skew()
kurtosis = np.log1p(df['price']).kurt()
# print(skewness,kurtosis)

#DISTRIBUTION PLOT WITHOUT LOG TRANSFORMATION:
'''
plt.subplot(1, 2, 1)
sns.boxplot(x=df['price'], color='skyblue')
plt.title('Distribution of Prices (Original)')
plt.xlabel('Price (in Crores)')
plt.ylabel('Frequency')
'''

# DISTRIBUTION PLOT WITH LOG TRANSFORMATION:
'''
plt.subplot(1, 2, 2)
sns.boxplot(x=np.log1p(df['price']), color='lightgreen')
plt.title('Distribution of Prices (Log Transformed)')
plt.xlabel('Log(Price)')
plt.ylabel('Frequency')
'''

#COLUMN-->price_per_sqft
# print(df['price_per_sqft'].isnull().sum())
# print(df['price_per_sqft'].describe())

# sns.histplot(df['price_per_sqft'], bins=50, color='skyblue', kde=True)
'''
OBSERVATION:
        MOST PROPERTIES HAVE A PRICE_PER_SQFT RANGING BETWEEN APPROXIMATELY ₹0 AND ₹40,000.
        THERE IS A SIGNIFICANT CONCENTRATION IN THE LOWER RANGE, WITH A FEW PROPERTIES
        HAVING EXCEPTIONALLY HIGH PRICE_PER_SQFT.
'''
# sns.boxplot(x=df['price_per_sqft'], color='lightgreen')

'''
OBSERVATION:
        “THE BOX PLOT CLEARLY SHOWS SEVERAL OUTLIERS, ESPECIALLY ON THE HIGHER SIDE.
         THE INTER QUARTILE RANGE (IQR) IS RELATIVELY COMPACT, BUT THERE ARE MANY DATA 
         POINTS BEYOND THE ‘WHISKERS’ OF THE BOX PLOT, INDICATING POTENTIAL OUTLIERS.”
         
         1.POTENTIAL OUTLIERS.
         2.RIGHT SKEWED
         3.17 MISSING VALUES
'''

#COLUMN-->bedRoom
'''
print(df['bedRoom'].isnull().sum())
df['bedRoom'].value_counts().sort_index().plot(kind='bar')
df['bedRoom'].value_counts(normalize=True).head().plot(kind='pie',autopct='%0.2f%%')
'''

#COLUMN-->bathroom
'''
print(df['bathroom'].isnull().sum())
df['bathroom'].value_counts().sort_index().plot(kind='bar')
df['bathroom'].value_counts(normalize=True).head().plot(kind='pie',autopct='%0.2f%%')
'''
# print(df.shape)

#COLUMN-->balcony
'''
print(df['balcony'].isnull().sum())
df['balcony'].value_counts().plot(kind='bar')
df['balcony'].value_counts(normalize=True).head().plot(kind='pie',autopct='%0.2f%%')
'''

#COLUMN-->floorNum
'''
print(df['floorNum'].isnull().sum())
print(df['floorNum'].describe())
df['floorNum'].value_counts().sort_index().plot(kind='bar')
sns.boxplot(x=df['floorNum'], color='lightgreen')
'''

'''
OBSERVATION:
        1.THE MAJORITY OF THE PROPERTIES LIE BETWEEN THE GROUND FLOOR (0) AND THE 25TH FLOOR.
        2.FLOORS 1 TO 4 ARE PARTICULARLY COMMON, WITH THE 3RD FLOOR BEING THE MOST FREQUENT.
        3.THERE ARE A FEW PROPERTIES LOCATED AT HIGHER FLOORS, BUT THEIR FREQUENCY IS MUCH LOWER.
        4.THE BOX PLOT REVEALS THAT THE MAJORITY OF THE PROPERTIES ARE CONCENTRATED AROUND THE LOWER
         FLOORS.
        5.THE INTER QUARTILE RANGE (IQR) LIES BETWEEN APPROXIMATELY THE 2ND AND 10TH FLOORS.
        6.DATA POINTS BEYOND THE "WHISKERS" OF THE BOX PLOT, ESPECIALLY ON THE HIGHER SIDE, INDICATE 
        POTENTIAL OUTLIERS.
'''

#COLUMN-->facing
'''
print(df['facing'].isnull().sum())
df.fillna('NA', inplace=True)
print(df['facing'].value_counts())
'''

#COLUMN-->agePossession
'''
print(df['agePossession'].isnull().sum())
print(df['agePossession'].value_counts())
'''

#COLUMN-->area
#1.super built-up area
'''
print(df['super_built_up_area'].isnull().sum())
print(df['super_built_up_area'].describe())
sns.histplot(df['super_built_up_area'].dropna(), bins=50, color='skyblue', kde=True)
plt.subplot(1,2,1)
sns.boxplot(x=df['super_built_up_area'])
plt.subplot(1,2,2)
sns.boxplot(x=df['super_built_up_area'].dropna(), color='lightgreen')
'''
'''
OBSERVATION:
        1.MOST PROPERTIES HAVE A SUPER BUILT-UP AREA RANGING BETWEEN APPROXIMATELY
        1,000 SQ.FT AND 2,500 SQ.FT.
        2.THERE ARE A FEW PROPERTIES WITH A SIGNIFICANTLY LARGER AREA, LEADING TO A
        RIGHT-SKEWED DISTRIBUTION.
        3.THE INTER QUARTILE RANGE (IQR) LIES BETWEEN ROUGHLY 1,480 SQ.FT AND 2,215 SQ.FT,
        INDICATING THAT THE MIDDLE 50% OF THE PROPERTIES FALL WITHIN THIS RANGE.
        4.THERE ARE SEVERAL DATA POINTS BEYOND THE UPPER "WHISKER" OF THE BOX PLOT,
        INDICATING POTENTIAL OUTLIERS. THESE ARE PROPERTIES WITH AN UNUSUALLY LARGE
        SUPER BUILT-UP AREA.
'''

#2.built-up area
'''
print(df['built_up_area'].isnull().sum())
sns.histplot(df['built_up_area'].dropna(), bins=50, color='skyblue', kde=False)
print(df['built_up_area'].describe())
print(df['built_up_area'].value_counts())
sns.boxplot(x=df['built_up_area'].dropna(), color='lightgreen')
df = df.drop(df[df['built_up_area']>700000].index)
sns.boxplot(x=df['built_up_area'].dropna(), color='lightgreen')
sns.histplot(df['built_up_area'].dropna(), bins=50, color='skyblue', kde=False)
'''

#3.carpet area
'''
print(df['carpet_area'].isnull().sum())
print(df['carpet_area'].describe())
sns.histplot(df['carpet_area'].dropna(), bins=50, color='skyblue', kde=False)
sns.boxplot(df['carpet_area'].dropna(), color='lightgreen')
'''

#COLUMN-->furnishing_type
print(df['furnishing_type'].value_counts())
df['furnishing_type'].value_counts().plot(kind='pie',autopct='%0.2f%%')

plt.show()
