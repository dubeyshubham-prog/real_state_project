#REQUIRED LIBRARIES=>
import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
#===================>

st.set_page_config(page_title='Plotting Demo')
st.title('Analytics')

#READING THE DATASET HERE===>
new_df = pd.read_csv('C:\datasciencejourney\REAL_STATE_CAMPUSX\REAL_ESTATE_APPLICATION\data_viz1.csv').drop(columns=['Unnamed: 0'])
st.dataframe(new_df)

feature_text = pickle.load(open(r'C:\datasciencejourney\REAL_STATE_CAMPUSX\REAL_ESTATE_APPLICATION\feature_text.pkl','rb'))
#===========================>

#DATA PREPROCESSING SECTION=>
#GROUP BY ON THE BASIS OF sector:
#FOR GRAPH_1
group_df = new_df.groupby('sector')[['price',
                                     'price_per_sqft',
                                     'built_up_area',
                                     'latitude',
                                     'longitude']].mean()

#FOR GRAPH_5
sector_list = new_df['sector'].unique().tolist()
sector_list.insert(0,'Overall')
# sector_list = pd.Series(sector_list)

#FOR GRAPH_5
temp_df = new_df[new_df['bedRoom'] <= 4]
#===========================>

#VISUALIZATION SECTION======>
#GRAPH_1. PLOTTING MAP OF ALL SECTION:
st.subheader('Visual representation of all sectors')
fig1 = px.scatter_mapbox(group_df,
                        lat='latitude',
                        lon='longitude',
                        color="price_per_sqft",
                        size='built_up_area',
                        color_continuous_scale=px.colors.cyclical.IceFire,
                        zoom=10,
                        mapbox_style="open-street-map",
                        width=1200,
                        height=700)
st.plotly_chart(fig1, use_container_width=True)

#GRAPH_2. PLOTTING FEATURES IN THE FORM OF TEXT:
st.subheader('Visual representation of all features')
wordcloud = WordCloud(width = 800, height = 800,
                      background_color ='white',
                      stopwords = set(['s']),  # Any stopwords you'd like to exclude
                      min_font_size = 10).generate(feature_text)

fig2, ax = plt.subplots(figsize=(8, 8))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
plt.tight_layout(pad = 0)
st.pyplot(fig2)

#GRAPH_3. SHOWING PROPERTY DISTRIBUTION WITH SUNBURST:
st.subheader('Property distribution with sunburst')
fig3 = px.sunburst(
    new_df,
    path=['bedRoom', 'property_type'],
    values='price_per_sqft'
)
st.plotly_chart(fig3, use_container_width=True)

#GRAPH_4. AREA VS PRICE SCATTER PLOT:
st.subheader('Area vs Price')
property_type = st.selectbox('Select on of the property types', ['flat','house'])
if property_type == 'flat':
    fig4 = px.scatter(new_df[new_df['property_type']=='flat'],
                      x="built_up_area",
                      y="price",
                      color="bedRoom")
    st.plotly_chart(fig4, use_container_width=True)
else:
    fig4 = px.scatter(new_df[new_df['property_type']=='house'],
                      x="built_up_area",
                      y="price",
                      color="bedRoom")
    st.plotly_chart(fig4, use_container_width=True)

#GRAPH_5. VISUALIZATION OF PIE BEDROOMS USING PIE CHART:
st.subheader('Bedroom percentage sector wise')
selected_sector = st.selectbox('Select any preferred sector',sector_list)
if selected_sector=='Overall':
    fig5 = px.pie(new_df, names='bedRoom')
    st.plotly_chart(fig5, use_container_width=True)
else:
    fig5 = px.pie(new_df[new_df['sector']==selected_sector], names='bedRoom')
    st.plotly_chart(fig5, use_container_width=True)

#GRAPH_6. BEDROOM TO PRICE COMPARISON USING BOX PLOT:
st.subheader('Bedroom to price comparison')
fig6 = px.box(temp_df,
              x='bedRoom',
              y='price',
              color=temp_df['bedRoom'],
              title='BHK Price Range')
st.plotly_chart(fig6, use_container_width=True)
#===========================>
