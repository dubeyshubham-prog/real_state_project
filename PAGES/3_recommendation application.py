#REQUIRED LIBRARIES=>
import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
#===================>

st.set_page_config(page_title='Recommendation System')
st.subheader('Recommendation System')

#READ THE PICKLE FILE:
location_df = pickle.load(open('location_distance.pkl', 'rb'))
cosine_sim1 = pickle.load(open('cosine_sim1.pkl', 'rb'))
cosine_sim2 = pickle.load(open('cosine_sim2.pkl', 'rb'))
cosine_sim3 = pickle.load(open('cosine_sim3.pkl', 'rb'))
df1 = pickle.load(open('df1.pkl', 'rb'))

#SHOWING THE DATAFRAME ON STREAMLIT:
# st.dataframe(location_df)

#SELECT THE LOCATION AND THE RADIUS:
st.subheader('Select  the location and the radius')
location = st.selectbox('Location', sorted(location_df.columns.to_list()))
radius = st.number_input('Radius in Kms')

if st.button('Search'):
    selected_location = location_df[location_df[location] < radius * 1000][location].sort_values()
    for key,value in selected_location.items():
        st.text(str(key) + '-->' + str(round(value/1000)) + 'Kms')

#FINAL RECOMMENDATION SYSTEM:
#THE FINAL FUNCTION OF OUR RECOMMENDATION SYSTEM:
def recommend_properties_with_scores(property_name, top_n=5):
    cosine_sin_matrix = 30 * cosine_sim1 + 20 * cosine_sim2 + 8*cosine_sim3

    #GET THE SIMILARITY SCORES FOR THE PROPERTY USING ITS NAME AS THE INDEX:
    sim_scores = list(enumerate(cosine_sin_matrix[location_df.index.get_loc(property_name)]))

    #SORT PROPERTIES BASED ON THE SIMILARITY SCORES:
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    #GET THE INDICES AND SCORES OF THE top_n MOST SIMILAR PROPERTIES:
    top_indices = [i[0] for i in sorted_scores[1:top_n + 1]]
    top_scores = [i[1] for i in sorted_scores[1:top_n + 1]]

    #RETRIEVE THE NAMES OF THE TOP PROPERTIES USING THE INDICES
    top_properties = location_df.index[top_indices].tolist()

    #CREATE A DATAFRAME WITH THE RESULT
    recommendations_df = pd.DataFrame({
        'PropertyName': top_properties,
        'SimilarityScore': top_scores
    })
    return recommendations_df

selected_apartment = st.selectbox('Select any apartment', df1['PropertyName'])
if st.button('Recommend Apartment'):
    recommendation_df = recommend_properties_with_scores(selected_apartment)

    st.dataframe(recommendation_df)

