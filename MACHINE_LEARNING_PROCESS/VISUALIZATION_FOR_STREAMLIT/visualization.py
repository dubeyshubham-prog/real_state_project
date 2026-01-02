#REQUIRED LIBRARIES=>
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import plotly.express as px
import ast
from wordcloud import WordCloud
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
df = pd.read_csv('gurgaon_properties_missing_value_imputation.csv')
latlong = pd.read_csv('latlong.csv')

#SPLITTING THE latlong DATASET ON THE BASIS OF:
latlong['latitude'] = latlong['coordinates'].str.split(',').str.get(0).str.split('°').str.get(0).astype('float')
latlong['longitude'] = latlong['coordinates'].str.split(',').str.get(1).str.split('°').str.get(0).astype('float')

#CREATING A NEW DATAFRAME BY MERGING BOTH OF THEM:
new_df = df.merge(latlong, on='sector')
# print(new_df.columns)

#GROUP BY ON THE BASIS OF sector:
group_df = new_df.groupby('sector')[['price','price_per_sqft','built_up_area','latitude','longitude']].mean()

#PLOTTING FIGURE:

# fig = px.scatter_mapbox(group_df,
#                         lat='latitude',
#                         lon='longitude',
#                         color="price_per_sqft",
#                         size='built_up_area',
#                         color_continuous_scale=px.colors.cyclical.IceFire,
#                         zoom=10,
#                         mapbox_style="open-street-map",
#                         text=group_df.index)
# fig.show()

# new_df.to_csv('data_viz1.csv',index=False)
#READING THE DATASET:
df1 = pd.read_csv('gurgaon_properties.csv')
# print(df1.head())
property_type_count = df['property_type'].value_counts()
# print(property_type_count)
# print(df1.head())

#MERGING DF AND DF1 TOGETHER:
wordcloud_df = df1.merge(df,
                         left_index=True,
                         right_index=True)[['features','sector']]
# print(wordcloud_df.head())

#COLLECTING ALL THE FEATURES IN A SINGLE STRING:
main=[]
for item in wordcloud_df['features'].dropna().apply(ast.literal_eval):
    main.extend(item)

#NOW ACCUMULATE THE ENTIRE LIST IN A STRING:
feature_text = ' '.join(main)
pickle.dump(feature_text, open('../../feature_text.pkl', 'wb'))
# print(feature_text)
print(df.columns)
#HAVE TO KNOW NLP TO UNDERSTAND CODE BELOW:
'''
plt.rcParams["font.family"] = "Arial"

wordcloud = WordCloud(width = 800, height = 800,
                      background_color ='white',
                      stopwords = set(['s']),  # Any stopwords you'd like to exclude
                      min_font_size = 10).generate(feature_text)

plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show() # st.pyplot()
'''

data = dict(
    names=["A", "B", "C", "D", "E", "F"],
    parents=["", "", "", "A", "A", "C"],
    values=[10, 20, 30, 40, 50, 60],
)
#
# fig = px.sunburst(
#     df1,
#     path=['bedRoom', 'property_type'],
#     values='price_per_sqft',
#     title="Property Distribution"
# )
# fig.show()

# fig = px.scatter(df, x="built_up_area", y="price", color="bedRoom", title="Area Vs Price")
#
# # Show the plot
# fig.show()

# fig = px.pie(df, names='bedRoom', title='Total Bill Amount by Day')
# # Show the plot
# fig.show()

# temp_df = df[df['bedRoom'] <= 4]
# # Create side-by-side boxplots of the total bill amounts by day
# fig = px.box(temp_df, x='bedRoom', y='price', title='BHK Price Range')
#
# # Show the plot
# fig.show()

# sns.distplot(df[df['property_type'] == 'house']['price'])
# sns.distplot(df[df['property_type'] == 'flat']['price'])
# plt.show()
# new_df.to_csv('data_viz1.csv')
sector_list = new_df['sector'].tolist()
sector_list.insert(0,'Overall')
print(sector_list)