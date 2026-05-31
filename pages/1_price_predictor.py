import pickle
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title='EliteEstate | Price Predictor', layout='wide')

# Load files safely
with open('df.pkl', 'rb') as file:
    df = pickle.load(file)

with open('pipelines.pkl', 'rb') as file:
    pipeline = pickle.load(file)

st.title('🏡 Property Price Predictor')
st.markdown("Fill out the property characteristics below to estimate the current market valuation.")
st.markdown("---")

# Grouping inputs into neat columns instead of a long single vertical list
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📍 Location & Type")
    property_type = st.selectbox('Property Type', ['flat', 'house'])
    sector = st.selectbox('Sector', sorted(df['sector'].unique().tolist()))
    property_age = st.selectbox('Property Age', sorted(df['agePossession'].unique().tolist()))

with col2:
    st.markdown("### 🛏️ Rooms & Layout")
    bedrooms = float(st.selectbox('Number of Bedrooms', sorted(df['bedRoom'].unique().tolist())))
    bathroom = float(st.selectbox('Number of Bathrooms', sorted(df['bathroom'].unique().tolist())))
    balcony = st.selectbox('Number of Balconies', sorted(df['balcony'].unique().tolist()))
    built_up_area = float(st.number_input('Built Up Area (SqFt)', min_value=100.0, value=1000.0))

with col3:
    st.markdown("### ✨ Amenities & Premium")
    furnishing_type = st.selectbox('Furnishing Type', sorted(df['furnishing_type'].unique().tolist()))
    luxury_category = st.selectbox('Luxury Category', sorted(df['luxury_category'].unique().tolist()))
    floor_category = st.selectbox('Floor Category', sorted(df['floor_category'].unique().tolist()))

    # Extra utility rooms grouped cleanly
    c_sub1, c_sub2 = st.columns(2)
    with c_sub1:
        servant_room = float(st.selectbox('Servant Room', [0.0, 1.0]))
    with c_sub2:
        store_room = float(st.selectbox('Store Room', [0.0, 1.0]))

st.markdown("---")

# Execution Action Center
if st.button('🎯 Predict Property Value', use_container_width=True):
    data = [[property_type, sector, bedrooms, bathroom, balcony, property_age, built_up_area,
             servant_room, store_room, furnishing_type, luxury_category, floor_category]]
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony', 'agePossession',
               'built_up_area', 'servant room', 'store room', 'furnishing_type', 'luxury_category', 'floor_category']

    one_df = pd.DataFrame(data, columns=columns)

    # Run the model pipeline
    base_price = (np.expm1(pipeline.predict(one_df)))[0]
    low = max(0.0, base_price - 0.22)
    high = base_price + 0.22

    # UI presentation container for the result
    st.success("### Valuation Model Output Summary")
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric(label="Estimated Lower Range", value=f"₹ {round(low, 2)} Cr")
    with metric_col2:
        st.metric(label="Estimated Upper Range", value=f"₹ {round(high, 2)} Cr")

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