import streamlit as st
from tensorflow import keras
from tensorflow.image import resize
import numpy as np
import pandas as pd
from PIL import Image
import time
import os
import base64
import imageio 
import cv2
from marswinds.predict import Predictor

st.set_page_config(layout="wide", 
                    page_icon=":earth_africa:", 
                    page_title="Mapping Martian Winds", 
                     initial_sidebar_state="expanded")
image_1=None
pixel_dimension = ''
imgsize = 256
#Information
side= ['Horizontal side', 'Vertical side']

#Import model
model_class = keras.models.load_model("raw_data/trained_models/classifier.h5")
model_reg = keras.models.load_model("raw_data/trained_models/regressor.h5")

#logo at the top of page
image = Image.open('website/ESP_015917_2650.jpg')
st.image(image, caption='', use_column_width=True)

#Create sidebar
st.sidebar.markdown("""
# Martian Winds
Please upload a photo of Mars to predict wind direction
    """)


if st.sidebar.checkbox('Insert pixel dimension'):
    pixel_dimension = st.sidebar.number_input('Insert pixel dimension of your image (METERS)', 5.000)

st.sidebar.markdown(" **OR**")

if st.sidebar.checkbox('Insert side dimension of image'):
    option = st.sidebar.selectbox('Insert side dimension of your image', ['Horizontal side', 'Vertical side'])
    img_axis = st.sidebar.number_input('Insert METERS represented by side of the image', 10.0)
    
uploaded_file = st.sidebar.file_uploader("Choose a photo of dunes in Mars*")

st.markdown(uploaded_file)
if uploaded_file:
    image_1 = imageio.imread(uploaded_file)

st.sidebar.markdown(f"""
/* Mandatory field
    """)

st.markdown("<h1 style='text-align: center;'>Mapping Martian Winds</h1>", unsafe_allow_html=True)

#Description text

st.markdown(
os.getcwd()
)

if image_1 is not None:

    def px_dimension():
        if pixel_dimension:
            pixels = pixel_dimension 
            return pixels

        if option =="Horizontal side":
            pixels = img_axis/image_1.shape[0] 
            return pixels

        if option =="Vertical side":
            pixels = img_axis/image_1.shape[1] 
            return pixels

    pixel_dim = px_dimension()
    
    st.image(uploaded_file, caption=f"Pixel dimension: {pixel_dim} m/px", use_column_width=True)
    # img = Image.load_img(uploaded_file)
    st.markdown(image_1)
    st.markdown(image_1.shape)
    st.markdown(type(image_1))
    prediction = st.button('Prediction')
    if prediction:
        with st.spinner('Wait for it...'):
            time.sleep(5)
            st.success('Done!')
        image, data= Predictor().get_prediction_image(image_1, pixel_dim)
        pred_image = Image.open(image)
        st.image(pred_image,caption ='', use_column_width=True)
        st.write(data)
        # download=st.button('Download data')
        # if download:
        #     'Download started'
        #     # list = ['A','B','C']
        #     # data = pd.DataFrame(list) #result from function
        #     # data.columns=['Title']
        #     data
        #     csv=data.to_csv(index=False)
        #     b64 = base64.b64encode(csv.encode()).decode()  # some strings
        #     link= f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download csv file</a>'
        #     st.markdown(link, unsafe_allow_html=True)
        #     st.image(pred_image, caption='', use_column_width=True)
else:
    st.markdown('Please upload image and fill information requested')

    









