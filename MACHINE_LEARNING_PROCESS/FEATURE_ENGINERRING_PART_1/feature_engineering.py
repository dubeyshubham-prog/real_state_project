#ALL THE REQUIRED LIBRARIES=>
import numpy as np
import pandas as pd
import re
import warnings
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles
from scipy.cluster.vq import kmeans
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
import ast

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
df = pd.read_csv('gurgaon_properties_cleaned_v1.csv')
# print(df.head(1))
# print(df.duplicated().sum())

#FOCU IS ON-->areaWithType, additionalRoom, agePossession, furnishDetails, features

#COLUMN--> areaWidthType
# print(df.sample(5)[['area','price','areaWithType']])
#CREATE FUNCTIONS TO CREATE BUILT-UP AREA, SUPER-BUILT-UP AREA AND CARPET AREA

#FUNCTION_1:
#THIS FUNCTION WILL EXTRACT THE CONTENT OF SUPER-BUILT-UP AREA:
def super_built_up_area(text):
    match = re.search(r'Super Built up area (\d+\.?\d*)',text)
    if match:
        return float(match.group(1))
    else:
        return None

#FUNCTION_2:
#THIS FUNCTION EXTRACT THE BUILD-UP AREA AND CARPET AREA:
def get_area(text,area_type):
    match = re.search(area_type + r'\s*:\s*(\d+\.?\d*)', text)
    if match:
        return float(match.group(1))
    return None

#FUNCTION_3:
#THIS FUNCTION CHECKS IF THE AREA IS PROVIDED IN sq.m. AND CONVERTS IT TO sqft IF NEEDED:
def convert_to_sqft(text, area_value):
    if area_value is None:
        return None
    match = re.search(r'{} \((\d+\.?\d*) sq.m.\)'.format(area_value),text)
    if match:
        sq_m_value = float(match.group(1))
        return sq_m_value*10.7639#CONVERSION FACTOR FROM sq.m. to sqft
    return area_value

#EXCTRACT SUPER-BUILT-UP AREA AND CONVERT IT TO sqft IF NEEDED:
df['super_built_up_area'] = df['areaWithType'].apply(super_built_up_area)
df['super_built_up_area'] = df.apply(lambda x: convert_to_sqft(x['areaWithType'],
                                                               x['super_built_up_area']), axis=1)

#EXTRACT BUILT_UP AREA AND CONVERT  TO sqft IF NEEDED
df['built_up_area'] = df['areaWithType'].apply(lambda x: get_area(x, 'Built Up area'))
df['built_up_area'] = df.apply(lambda x: convert_to_sqft(x['areaWithType'], x['built_up_area']), axis=1)

#EXTRACT CARPET AREA AND CONVERT TO sqft IF NEEDED
df['carpet_area'] = df['areaWithType'].apply(lambda x: get_area(x, 'Carpet area'))
df['carpet_area'] = df.apply(lambda x: convert_to_sqft(x['areaWithType'], x['carpet_area']), axis=1)
# print(df.head())

# print(df[['price','property_type','area','areaWithType','super_built_up_area','built_up_area','carpet_area']].sample(5))
# print(df.duplicated().sum())

# print(df[~((df['super_built_up_area'].isnull()) | (df['built_up_area'].isnull()) | (df['carpet_area'].isnull()))][['price','property_type','area','areaWithType','super_built_up_area','built_up_area','carpet_area']])
# print(df[df['areaWithType'].str.contains('Plot')][['price','property_type','area','areaWithType','super_built_up_area','built_up_area','carpet_area']].head(5))

# print(df.isnull().sum())
all_nan_df = df[((df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull()))][['price','property_type','area','areaWithType','super_built_up_area','built_up_area','carpet_area']]
# print(all_nan_df.shape)
all_nan_index = df[((df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull()))][['price','property_type','area','areaWithType','super_built_up_area','built_up_area','carpet_area']].index

#FUNCTION TO EXTRACT PLOT AREA FROM 'areaWithType' COLUMN
def extract_plot_area(area_with_type):
    match = re.search(r'Plot area (\d+\.?\d*)', area_with_type)
    return float(match.group(1)) if match else None

all_nan_df['built_up_area'] = all_nan_df['areaWithType'].apply(extract_plot_area)
#UPDATE THE ORIGINAL DATAFRAME
#GURGAON_PROPERTIES.UPDATE(FILTERED_ROWS)
# print(all_nan_df)

#FIXING THE RATIO BETWEEN THE AREA AND THE BUILT_UP_AREA:
def convert_scale(row):
    if np.isnan(row['area']) or np.isnan(row['built_up_area']):
        return row['built_up_area']
    else:
        if round(row['area']/row['built_up_area'])==9.0:
            return row['built_up_area']*9
        elif round(row['area']/row['built_up_area'])==11.0:
            return row['built_up_area']*10.7
        else:
            return row['built_up_area']

all_nan_df['built_up_area'] = all_nan_df.apply(convert_scale, axis=1)
# print(all_nan_df.head())

#UPDATE THE ORIGINAL DATAFRAME:
df.update(all_nan_df)
print(df.head())
# print(df.head())
# print(df.isnull().sum())

#COLUMN--> additionalRoom
# print(df['additionalRoom'].value_counts())
#LIST OF NEW COLUMNS TO BE CREATED:
new_col = ['study room', 'servant room', 'store room', 'pooja room', 'others']
#POPULATE THE NEW COLUMNS BASED  ON THE 'additionalRoom' column
for cols in new_col:
    df[cols] = df['additionalRoom'].str.contains(cols).astype(int)

# print(df.head())
# print(df.sample(5)[['additionalRoom','study room', 'servant room', 'store room', 'pooja room', 'others']])

#COLUMN-->agePossession
# print(df.columns)
# print(df['agePossession'].value_counts())
def categorise_age_possession(value):
    if pd.isna(value):
        return 'Undefined'
    if '0 to 1 Year Old' in value or 'Within 6 months' in value or 'Within 3 months' in value:
        return "New Property"
    if '1 to 5 Year Old' in value:
        return 'Moderately Old'
    if '10+ Year Old' in value:
        return "Old Property"
    if "Under Construction" in value or "By" in value:
        return "Under Construction"
    try:
        #FOR ENTRIES LIKE 'MAY 2024'
        int(value.split(' ')[-1])
        return 'Under Construction'
    except:
        return 'Undefined'

df['agePossession'] = df['agePossession'].apply(categorise_age_possession)
# print(df['agePossession'])

#COLUMN-->furnishDetails
# print(df.head(5)[['furnishDetails', 'features']])
#EXTRACT ALL UNIQUE FURNISHINGS FROM THE furnishDetails COLUMN
all_furnishings = []
for detail in df['furnishDetails'].dropna():
    furnishings = detail.replace('[', '').replace(']', '').replace("'", "").split(', ')
    all_furnishings.extend(furnishings)
unique_furnishings = list(set(all_furnishings))

#DEFINE A FUNCTION TO EXTRACT THE COUNT OF A FURNISHING FROM THE furnishDetails
def get_furnishing_count(details, furnishing):
    if isinstance(details, str):
        if f'No {furnishing}' in details:
            return 0
        pattern = re.compile(rf"(\d+) {furnishing}")
        match = pattern.search(details)
        if match:
            return int(match.group(1))
        elif furnishing in details:
            return 1
    return 0

#SIMPLIFY THE FURNISHINGS LIST BY REMOVING 'No' PREFIX AND NUMBERS
columns_to_include = [re.sub(r'No |\d+','',furnishings).strip() for furnishings in unique_furnishings]
columns_to_include = list(set(columns_to_include))#GET UNIQUE FURNISHINGS
columns_to_include = [furnishing for furnishing in columns_to_include if furnishing]  #REMOVE EMPTY STRINGS

#CREATE NEW COLUMNS FOR EACH UNIQUE FURNISHING AND POPULATE WITH COUNTS
for furnishing in columns_to_include:
    df[furnishing] = df['furnishDetails'].apply(lambda x: get_furnishing_count(x, furnishing))

#CREATE THE NEW DATAFRAME WITH THE REQUIRED COLUMNS
furnishings_df = df[['furnishDetails'] + columns_to_include]
furnishings_df.drop(columns=['furnishDetails'],inplace=True)
# print(furnishings_df.head())
# print(furnishings_df.shape)

#PERFORMING STANDARD SCALER ON THE furnishings_df DATA:
scaler = StandardScaler()
scaled_data = scaler.fit_transform(furnishings_df)

wcss_reduced = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    wcss_reduced.append(kmeans.inertia_)

#PLOT THE  FINAL RESULT:
'''
plt.figure(figsize=(12,8))
plt.plot(range(1, 11), wcss_reduced, marker='o', linestyle = '--')
plt.title('Elbow Method For Optimal Number of Clusters (Reduced Range)')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()
'''

n_cluster = 3

kmeans = KMeans(n_clusters=n_cluster, random_state=42)
kmeans.fit(scaled_data)

#PREDICT THE CLUSTER ASSIGNMENTS FOR EACH ROW
cluster_assignment = kmeans.predict(scaled_data)
# print(len(cluster_assignment))
df = df.iloc[:,:-18]
# print(df.head())

df['furnishing_type'] = cluster_assignment
# print(df.sample(5)[['furnishDetails','furnishing_type']])
# 0 ->UNFURNISHED
# 1 ->SEMI FURNISHED
# 2 ->UNFURNISHED

#COLUMN-->features
# print(df[['society','features']].sample(5))
# print(df['features'].isnull().sum())

#IMPORTING A NEW DATA-SET CALLED appartments.csv
app_df = pd.read_csv('appartments.csv')
# print(app_df.head())

app_df['PropertyName'] = app_df['PropertyName'].str.lower()
# print(app_df['PropertyName'])
temp_df = df[df['features'].isnull()]
x = temp_df.merge(app_df,left_on='society',right_on='PropertyName',how='left')['TopFacilities']
df.loc[temp_df.index,'features'] = x.values
# print(df['features'].isnull().sum())

#CONVERT THE STRING REPRESENTATION OF LISTS IN THE 'features' COLUMN TO  ACTUAL LISTS
df['features_list'] = df['features'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) and x.startswith('[') else [])

#USE MultiLabelBinarizer TO CONVERT THE FEATURES LIST INTO A BINARY MATRIX
mlb = MultiLabelBinarizer()
features_binary_matrix = mlb.fit_transform(df['features_list'])

#CONVERT THE BINARY MATRIX INTO A DATAFRAME
features_binary_df = pd.DataFrame(features_binary_matrix, columns=mlb.classes_)
# print(features_binary_df.sample(5))
print(features_binary_df.shape)

wcss_reduced = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(features_binary_df)
    wcss_reduced.append(kmeans.inertia_)

#PLOT THE RESULTS
plt.figure(figsize=(12, 8))
plt.plot(range(1,11), wcss_reduced, marker='o', linestyle='--')
plt.title('Elbow Method For Optimal Number of Clusters (Reduced Range)')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

#DEFINING THE WEIGHTS FOR EACH FEATURES AS PROVIDED
#ASSIGNING WEIGHTS BASED ON PERCEIVED LUXURY CONTRIBUTION
weights = {
    '24/7 Power Backup': 8,
    '24/7 Water Supply': 4,
    '24x7 Security': 7,
    'ATM': 4,
    'Aerobics Centre': 6,
    'Airy Rooms': 8,
    'Amphitheatre': 7,
    'Badminton Court': 7,
    'Banquet Hall': 8,
    'Bar/Chill-Out Lounge': 9,
    'Barbecue': 7,
    'Basketball Court': 7,
    'Billiards': 7,
    'Bowling Alley': 8,
    'Business Lounge': 9,
    'CCTV Camera Security': 8,
    'Cafeteria': 6,
    'Car Parking': 6,
    'Card Room': 6,
    'Centrally Air Conditioned': 9,
    'Changing Area': 6,
    "Children's Play Area": 7,
    'Cigar Lounge': 9,
    'Clinic': 5,
    'Club House': 9,
    'Concierge Service': 9,
    'Conference room': 8,
    'Creche/Day care': 7,
    'Cricket Pitch': 7,
    'Doctor on Call': 6,
    'Earthquake Resistant': 5,
    'Entrance Lobby': 7,
    'False Ceiling Lighting': 6,
    'Feng Shui / Vaastu Compliant': 5,
    'Fire Fighting Systems': 8,
    'Fitness Centre / GYM': 8,
    'Flower Garden': 7,
    'Food Court': 6,
    'Foosball': 5,
    'Football': 7,
    'Fountain': 7,
    'Gated Community': 7,
    'Golf Course': 10,
    'Grocery Shop': 6,
    'Gymnasium': 8,
    'High Ceiling Height': 8,
    'High Speed Elevators': 8,
    'Infinity Pool': 9,
    'Intercom Facility': 7,
    'Internal Street Lights': 6,
    'Internet/wi-fi connectivity': 7,
    'Jacuzzi': 9,
    'Jogging Track': 7,
    'Landscape Garden': 8,
    'Laundry': 6,
    'Lawn Tennis Court': 8,
    'Library': 8,
    'Lounge': 8,
    'Low Density Society': 7,
    'Maintenance Staff': 6,
    'Manicured Garden': 7,
    'Medical Centre': 5,
    'Milk Booth': 4,
    'Mini Theatre': 9,
    'Multipurpose Court': 7,
    'Multipurpose Hall': 7,
    'Natural Light': 8,
    'Natural Pond': 7,
    'Park': 8,
    'Party Lawn': 8,
    'Piped Gas': 7,
    'Pool Table': 7,
    'Power Back up Lift': 8,
    'Private Garden / Terrace': 9,
    'Property Staff': 7,
    'RO System': 7,
    'Rain Water Harvesting': 7,
    'Reading Lounge': 8,
    'Restaurant': 8,
    'Salon': 8,
    'Sauna': 9,
    'Security / Fire Alarm': 9,
    'Security Personnel': 9,
    'Separate entry for servant room': 8,
    'Sewage Treatment Plant': 6,
    'Shopping Centre': 7,
    'Skating Rink': 7,
    'Solar Lighting': 6,
    'Solar Water Heating': 7,
    'Spa': 9,
    'Spacious Interiors': 9,
    'Squash Court': 8,
    'Steam Room': 9,
    'Sun Deck': 8,
    'Swimming Pool': 8,
    'Temple': 5,
    'Theatre': 9,
    'Toddler Pool': 7,
    'Valet Parking': 9,
    'Video Door Security': 9,
    'Visitor Parking': 7,
    'Water Softener Plant': 7,
    'Water Storage': 7,
    'Water purifier': 7,
    'Yoga/Meditation Area': 7
}

#CALCULATE LUXURY SCORE FOR EACH ROW
luxury_score = features_binary_df[list(weights.keys())].multiply(list(weights.values())).sum(axis=1)
df['luxury_score'] = luxury_score
print(df['luxury_score'].head())

# cols to drop -> nearbyLocations,furnishDetails, features,features_list, additionalRoom
df.drop(columns=['nearbyLocations','furnishDetails','features','features_list','additionalRoom','address'],inplace=True)
df.to_csv('gurgaon_properties_cleaned_v2.csv',index=False)

