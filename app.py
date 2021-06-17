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
from marswinds.predict import Predictor

st.set_page_config(layout="wide", 
                    page_icon=":earth_africa:", 
                    page_title="Mapping Martian Winds", 
                     initial_sidebar_state="expanded")
image_1=None
pixel_dimension = ''
imgsize = 256

#Information

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

#Show image to be predicted
if uploaded_file:
    base_image_path = 'website/prediction/base_image.jpg'
    image_1 = imageio.imread(uploaded_file)
    imageio.imwrite(base_image_path,image_1)

st.sidebar.markdown(f"""
/* Mandatory field
    """)

st.markdown("<h1 style='text-align: center;color: rgb(138, 19, 11);'>Mapping Martian Winds</h1>", unsafe_allow_html=True)

#Description text

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

    #Predict
    prediction = st.button('Predict')
    if prediction:
        image_pred, data= Predictor().get_prediction_image(base_image_path, pixel_dim)
        progress = pd.read_csv('website/progress_log.csv')
        # Add a placeholder
        latest_iteration = st.empty()
        bar = st.progress(0)
        while i in range(progress['total']):
             # Update the progress bar with each iteration.
            latest_iteration.text(f'Image tile {i+1}')
            bar.progress(i + 1)
            time.sleep(1)

        'Prediction completed'

        #Open image and data after prediction
        pred_image = Image.open(image_pred)
        st.image(pred_image,caption ='', use_column_width=True)
        st.write(data)

        #Create csv to be downloaded
        csv=data.to_csv(index=True)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings
        link= f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download csv file</a>'
        st.markdown(link, unsafe_allow_html=True)
else:
    st.markdown('Please upload image and fill information requested.')

    









