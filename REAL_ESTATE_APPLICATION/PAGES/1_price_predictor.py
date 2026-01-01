#REQUIRED LIBRARIES=>
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title='Viz Demo')

#LOAD THE DATASET FROM PKL:
with open('df.pkl','rb') as file:
    df = pickle.load(file)

#LOAD THE PIPELINE FROM PKL:
with open('pipeline.pkl','rb') as file:
    pipeline = pickle.load(file)

st.header('Enter your inputs')

#1.PROPERTY_TYPE:
property_type = st.selectbox('Property Type', ['flat', 'house'])

#2.SECTOR:
sector = st.selectbox('Sector',sorted(df['sector'].unique().tolist()))

#3.BEDROOMS:
bedrooms = float(st.selectbox('Number of Bedrooms',sorted(df['bedRoom'].unique().tolist())))

#4.BATHROOM:
bathroom = float(st.selectbox('Number of Bathroom',sorted(df['bathroom'].unique().tolist())))

#4.BALCONY:
balcony = st.selectbox('Number of Bathroom',sorted(df['balcony'].unique().tolist()))

#5.AGE_POSSESSION:
property_age = st.selectbox('Propert Age',sorted(df['agePossession'].unique().tolist()))

#6.BUILT_UP_AREA:
built_up_area = float(st.number_input('Built Up Area'))

#7.SERVANT_ROOM:
servant_room = float(st.selectbox('Servant Room', [0.0,1.0]))

#8.STORE_ROOM
store_room = float(st.selectbox('Store room', [0.0,1.0]))

#9.FURNISHING TYPE:
furnishing_type = st.selectbox('Furnishing Type',sorted(df['furnishing_type'].unique().tolist()))

#10.LUXURY CATEGORY:
luxury_category = st.selectbox('Luxury Category',sorted(df['luxury_category'].unique().tolist()))

#11.FLOOR CATEGORY:
floor_category = st.selectbox('Floor Category',sorted(df['floor_category'].unique().tolist()))

#CREATE THE LOGIC FOR PREDICTION OF THE PRICE:
if st.button('Predict'):
    #FORM A DATAFRAME:
    data = [[property_type, sector, bedrooms, bathroom, balcony, property_age,built_up_area,
            servant_room, store_room, furnishing_type, luxury_category, floor_category]]
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony', 'agePossession',
               'built_up_area', 'servant room', 'store room', 'furnishing_type', 'luxury_category', 'floor_category']

    one_df = pd.DataFrame(data, columns=columns)
    st.dataframe(one_df)
    #PREDICTION LOGIC:
    base_price = (np.expm1(pipeline.predict(one_df)))[0]
    low = base_price - 0.22
    high = base_price + 0.22

    #DISPLAY:
    st.text('The price of the flat is between {}Cr and {}Cr'.format(round(low,2), round(high,2)))

