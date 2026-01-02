#REQUIRED LIBRARIES=>
import ast
import json
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
import pickle


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
df = pd.read_csv('../../appartments.csv').drop(22)
# print(df.head())
# print(df.columns)

#-------------------- CREATING THE FIRST RECOMMENDATION SYSTEM ------------------->
#UNDERSTANDING THE DATASET:
# print(df.iloc[2].NearbyLocations)
# print(df.iloc[2].LocationAdvantages)
# print(df.iloc[1].PriceDetails)
# print(df.iloc[2].TopFacilities)
# print(df[['TopFacilities']][0])THIS ONE DOESN'T WORKING
# print(df[['PropertyName','TopFacilities']]['TopFacilities'][0])#BUT THIS ONE DOES
#HOW TO USE LOC AND ILOC?

#EXTRACTING DATA FROM THE COLUMN-->TopFacilities
def extract_list(s):
    return re.findall(r"'(.*?)'", s)
df['TopFacilities'] = df['TopFacilities'].apply(extract_list)

#CONVERTING LIST INTO STRING:
df['FacilitiesStr'] = df['TopFacilities'].apply(' '.join)
# print(df['FacilitiesStr'][0])

#APPLYING VECTORIZATION ON COLUMN-->FacilitiesStr
tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
tfidf_matrix = tfidf_vectorizer.fit_transform(df['FacilitiesStr'])
# print(tfidf_matrix.toarray()[0])

#NOW FINDING THE COSINE SIMILARITY BETWEEN tfidf_matrix AND tfidf_matrix:
cosine_sim1 = cosine_similarity(tfidf_matrix, tfidf_matrix)
# print(cosine_sim1.shape)

#CREATING A FUNCTION FOR RECOMMENDATION:
def recommend_properties(property_name, cosine_sim=cosine_sim1):
    #GET THE INDEX OF THE PROPERTY THAT MATCHES THE NAME:
    idx = df.index[df['PropertyName'] == property_name].tolist()[0]

    #GET THE PAIRWISE SIMILARITY SCORES WITH THAT PROPERTY:
    sim_scores = list(enumerate(cosine_sim1[idx]))

    #SORT THE PROPERTIES BASED ON THE SIMILARITY SCORES
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    #GET THE SCORES OF THE 5 MOST SIMILAR PROPERTIES:
    sim_scores = sim_scores[1:6]

    # GET THE PROPERTY INDICES:
    property_indices = [i[0] for i in sim_scores]

    recommendation_df = pd.DataFrame({
        'PropertyName': df['PropertyName'].iloc[property_indices],
        'SimilarityScore':sim_scores
    })

    #RETURN THE TOP 5 MOST SIMILAR PROPERTIES:
    return recommendation_df

recommended_properties = recommend_properties("DLF The Arbour")
# print(recommended_properties)
#--------------------------------------------------------------------------------->

#------------------- CREATING THE SECOND RECOMMENDATION SYSTEM ------------------->
#READING THE DATASET:
df_appartments = pd.read_csv('../../appartments.csv').drop(22)


# Function to parse and extract the required features from the PriceDetails column
def refined_parse_modified_v2(detail_str):
    try:
        details = json.loads(detail_str.replace("'", "\""))
    except:
        return {}

    extracted = {}
    for bhk, detail in details.items():
        # Extract building type
        extracted[f'building type_{bhk}'] = detail.get('building_type')

        # Parsing area details
        area = detail.get('area', '')
        area_parts = area.split('-')
        if len(area_parts) == 1:
            try:
                value = float(area_parts[0].replace(',', '').replace(' sq.ft.', '').strip())
                extracted[f'area low {bhk}'] = value
                extracted[f'area high {bhk}'] = value
            except:
                extracted[f'area low {bhk}'] = None
                extracted[f'area high {bhk}'] = None
        elif len(area_parts) == 2:
            try:
                extracted[f'area low {bhk}'] = float(area_parts[0].replace(',', '').replace(' sq.ft.', '').strip())
                extracted[f'area high {bhk}'] = float(area_parts[1].replace(',', '').replace(' sq.ft.', '').strip())
            except:
                extracted[f'area low {bhk}'] = None
                extracted[f'area high {bhk}'] = None

        # Parsing price details
        price_range = detail.get('price-range', '')
        price_parts = price_range.split('-')
        if len(price_parts) == 2:
            try:
                extracted[f'price low {bhk}'] = float(price_parts[0].replace('₹', '').replace(' Cr', '').replace(' L', '').strip())
                extracted[f'price high {bhk}'] = float(price_parts[1].replace('₹', '').replace(' Cr', '').replace(' L', '').strip())
                if 'L' in price_parts[0]:
                    extracted[f'price low {bhk}'] /= 100
                if 'L' in price_parts[1]:
                    extracted[f'price high {bhk}'] /= 100
            except:
                extracted[f'price low {bhk}'] = None
                extracted[f'price high {bhk}'] = None

    return extracted


# Apply the refined parsing and generate the new DataFrame structure
data_refined = []

for _, row in df_appartments.iterrows():
    features = refined_parse_modified_v2(row['PriceDetails'])

    # Construct a new row for the transformed dataframe
    new_row = {'PropertyName': row['PropertyName']}

    # Populate the new row with extracted features
    for config in ['1 BHK', '2 BHK', '3 BHK', '4 BHK', '5 BHK', '6 BHK', '1 RK', 'Land']:
        new_row[f'building type_{config}'] = features.get(f'building type_{config}')
        new_row[f'area low {config}'] = features.get(f'area low {config}')
        new_row[f'area high {config}'] = features.get(f'area high {config}')
        new_row[f'price low {config}'] = features.get(f'price low {config}')
        new_row[f'price high {config}'] = features.get(f'price high {config}')

    data_refined.append(new_row)

df_final_refined_v2 = pd.DataFrame(data_refined).set_index('PropertyName')

df_final_refined_v2['building type_Land'] = df_final_refined_v2['building type_Land'].replace({'':'Land'})
# print(df_final_refined_v2.head())
# print(df['PriceDetails'][10])

#CREATING A LIST OF STRING OR OBJECT COLUMNS SO THAT WE CAN PERFORM HE:
categorical_columns = df_final_refined_v2.select_dtypes(include=['object']).columns.tolist()
# print(categorical_columns)

#PERFORMING ONE HOT ENCODING:
ohe_df = pd.get_dummies(df_final_refined_v2, columns=categorical_columns, drop_first=True)
ohe_df.fillna(0,inplace=True)
# print(ohe_df)

#INITIALIZE THE SCALER
scaler = StandardScaler()

#APPLY THE SCALER TO ENTIRE DATAFRAME:
ohe_df_normalized = pd.DataFrame(scaler.fit_transform(ohe_df),
                                columns=ohe_df.columns,
                                index=ohe_df.index)
# print(ohe_df_normalized.head())

#COMPUTE THE COSINE SIMILARITY MATRIX:
cosine_sim2 = cosine_similarity(ohe_df_normalized)
# print(cosine_sim2.shape)

#CREATING A FUNCTION FOR RECOMMENDATION:
def recommend_properties_with_score(property_name, top_n=247):
    #GET THE SIMILARITY SCORES FOR THE PROPERTY USING ITS NAME AS THE INDEX:
    sim_scores = list(enumerate(cosine_sim2[ohe_df_normalized.index.get_loc(property_name)]))

    #SORT PROPERTIES BASED ON THE SIMILARITY SCORES:
    sorted_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)

    #GET THE INDICES AND SCORES OF THE top_n MOST SIMILAR PROPERTIES:
    top_indices = [i[0] for i in sorted_scores[1:top_n + 1]]
    top_scores = [i[1] for i in sorted_scores[1:top_n + 1]]

    #RETRIEVE THE NAMES OF THE TOP PROPERTIES USING THE INDICES:
    top_properties = ohe_df_normalized.index[top_indices].tolist()

    #CREATE A DATAFRAME WITH THE RESULTS:
    recommendations_df = pd.DataFrame({
        'PropertyName': top_properties,
        'SimilarityScore': top_scores
    })

    return recommendations_df

#TEST THE RECOMMENDER FUNCTION USING A PROPERTY NAME:
recommend_properties_with_score('M3M Golf Hills')

print(df[['PropertyName','LocationAdvantages']]['LocationAdvantages'][0])

#DISTANCE TO METERS CONVERSION GRAPH:
def distance_to_meters(distance_str):
    try:
        if 'Km' in distance_str or 'KM' in distance_str:
            return float(distance_str.split()[0])*1000
        elif 'Meter' in distance_str or 'meter' in distance_str:
            return float(distance_str.split()[0])
        else:
            return None
    except:
        return None

#EXTRACTED DISTANCES FOR EACH LOCATION:
location_matrix = {}
for index, row in df.iterrows():
    distances = {}
    for location, distance in ast.literal_eval(row['LocationAdvantages']).items():
        distances[location] = distance_to_meters(distance)
    location_matrix[index] = distances
#CONVERT THE DICTIONARY TO DATAFRAME:
location_df = pd.DataFrame.from_dict(location_matrix, orient='index')

# print(location_df.head())
# print(location_df.columns[10:50])
location_df.index = df.PropertyName
# print(location_df.head())

#FILLING THE MISSING VALUES:
location_df.fillna(54000,inplace=True)

#APPLYING THE STANDARD SCALER ON THE ENTIRE DATASETS:
#INITIALIZE THE DATASET:
scaler = StandardScaler()

#APPLY THE SCALER TO THE ENTIRE DATAFRAME:
location_df_normalized = pd.DataFrame(scaler.fit_transform(location_df),
                                      columns=location_df.columns,
                                      index=location_df.index)
cosine_sim3 = cosine_similarity(location_df_normalized)
print(cosine_sim3.shape)

#THE FINAL FUNCTION OF OUR RECOMMENDATION SYSTEM:
def recommend_properties_with_scores(property_name, top_n=247):
    cosine_sin_matrix = 30 * cosine_sim1 + 20 * cosine_sim2 + 8*cosine_sim3

    #GET THE SIMILARITY SCORES FOR THE PROPERTY USING ITS NAME AS THE INDEX:
    sim_scores = list(enumerate(cosine_sin_matrix[location_df_normalized.index.get_loc(property_name)]))

    #SORT PROPERTIES BASED ON THE SIMILARITY SCORES:
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    #GET THE INDICES AND SCORES OF THE top_n MOST SIMILAR PROPERTIES:
    top_indices = [i[0] for i in sorted_scores[1:top_n + 1]]
    top_scores = [i[1] for i in sorted_scores[1:top_n + 1]]

    #RETRIEVE THE NAMES OF THE TOP PROPERTIES USING THE INDICES
    top_properties = location_df_normalized.index[top_indices].tolist()

    #CREATE A DATAFRAME WITH THE RESULT
    recommendations_df = pd.DataFrame({
        'PropertyName': top_properties,
        'SimilarityScore': top_scores
    })
    return recommendations_df


# Test the recommender function using a property name
# print(recommend_properties_with_scores('Ireo Victory Valley'))
# print((3*cosine_sim3 + 5*cosine_sim2 + 6*cosine_sim1).shape)

# pickle.dump(location_df, open('location_distance.pkl', 'wb'))
# print(location_df)
print(location_df[location_df['Bajghera Road']<2000]['Bajghera Road'].sort_values())
#--------------------------------------------------------------------------------->

# pickle.dump(cosine_sim1, open('../../REAL_ESTATE_APPLICATION/cosine_sim1.pkl', 'wb'))
# pickle.dump(cosine_sim2, open('../../REAL_ESTATE_APPLICATION/cosine_sim2.pkl', 'wb'))
# pickle.dump(cosine_sim3, open('../../REAL_ESTATE_APPLICATION/cosine_sim3.pkl', 'wb'))
pickle.dump(df, open('../../df1.pkl', 'wb'))
#------------------- CREATING THE THIRD RECOMMENDATION SYSTEM -------------------->
#--------------------------------------------------------------------------------->




