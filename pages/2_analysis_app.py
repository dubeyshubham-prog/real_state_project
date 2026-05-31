import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title='EliteEstate | Market Analytics', layout='wide')
st.title('📊 Market Trend Analytics')
st.markdown("Explore regional housing pricing spread and layout concentrations.")
st.markdown("---")

# Data pipeline aggregation
new_df = pd.read_csv('data_viz1.csv').drop(columns=['Unnamed: 0'])
feature_text = pickle.load(open('feature_text.pkl', 'rb'))

group_df = new_df.groupby('sector')[['price', 'price_per_sqft', 'built_up_area', 'latitude', 'longitude']].mean()

sector_list = new_df['sector'].unique().tolist()
sector_list.insert(0, 'Overall')
temp_df = new_df[new_df['bedRoom'] <= 4]

# Layout: Geographic insights mapped large
st.subheader('📍 Spatial Pricing Map Across Sectors')
fig1 = px.scatter_mapbox(group_df, lat='latitude', lon='longitude', color="price_per_sqft",
                        size='built_up_area', color_continuous_scale=px.colors.cyclical.IceFire,
                        zoom=10, mapbox_style="open-street-map", height=500)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# Layout: Creating side-by-side split dashboards to reduce layout scrolling
left_col, right_col = st.columns(2)

with left_col:
    st.subheader('☀️ Structural Property Breakdown (Sunburst)')
    fig3 = px.sunburst(new_df, path=['bedRoom', 'property_type'], values='price_per_sqft')
    st.plotly_chart(fig3, use_container_width=True)

with right_col:
    st.subheader('🏷️ Structural Features Cloud')
    wordcloud = WordCloud(width=600, height=450, background_color='white', stopwords=set(['s']), min_font_size=10).generate(feature_text)
    fig2, ax = plt.subplots(figsize=(6, 4.5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(fig2)

st.markdown("---")

# Layout: Interactive dynamic scatter plotting section
st.subheader('📐 Size (Built-up Area) vs Pricing Scale')
property_type_filter = st.segmented_control('Filter Classification', ['flat', 'house'], default='flat')

fig4 = px.scatter(new_df[new_df['property_type'] == property_type_filter],
                  x="built_up_area", y="price", color="bedRoom",
                  color_continuous_scale=px.colors.sequential.Viridis)
st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# Layout: Sector breakdowns versus broader box plots
b1, b2 = st.columns([1, 1])

with b1:
    st.subheader('🍰 Bedroom Quantities Percentage')
    selected_sector = st.selectbox('Select target sector to review configuration profiles:', sector_list)
    active_df = new_df if selected_sector == 'Overall' else new_df[new_df['sector'] == selected_sector]
    fig5 = px.pie(active_df, names='bedRoom', hole=0.3)
    st.plotly_chart(fig5, use_container_width=True)

with b2:
    st.subheader('📦 Price Range Dispersion (BHK Boxplot)')
    fig6 = px.box(temp_df, x='bedRoom', y='price', color='bedRoom')
    st.plotly_chart(fig6, use_container_width=True)

# 1. Add a divider line in the sidebar
st.sidebar.markdown("---")

# 2. Add your developer title
st.sidebar.markdown("### 👨‍💻 Developer Profile")
st.sidebar.info("""
**Shubham Dubey** 🎯 AI / Data Science Specialist
""")

# 3. Add the clickable portfolio button (REPLACE THE URL BELOW WITH YOUR LINK)
st.sidebar.link_button(
    label="🌐 Visit My Portfolio",
    url="https://dazzling-pudding-0b3156.netlify.app/",
    use_container_width=True
)