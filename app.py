import streamlit as st
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.image import resize
import numpy as np
import time
from PIL import Image

imgsize = 256
#Classes used for prediction
classes = ['no_dunes', 
         'dunes']

#Import model
model = keras.models.load_model("notebooks/raw_data/models_regressionXception_0.13006779551506042__0.13006779551506042.h5")

#logo at the top of page
image = Image.open('ESP_015917_2650.jpg')
st.image(image, caption='', use_column_width=True)


#Create sidebar
st.sidebar.markdown(f"""
# Martian Winds
Please upload a photo of Mars to predict wind direction
    """)


img_x_axis = st.sidebar.number_input('Insert number of kilometers represented by horizontal side of the image', 1)

img_y_axis = st.sidebar.number_input('Insert number of kilometers represented by vertical side of the image', 1)


uploaded_file = st.sidebar.file_uploader("Choose a photo of dunes in Mars")


st.markdown("<h1 style='text-align: center;'>Mapping Martian Winds</h1>", unsafe_allow_html=True)


#Description text

st.markdown('''
Bla Bla Bla 
''')


#Get user photo input

if uploaded_file is not None:
    st.image(uploaded_file, caption='', use_column_width=True)

st.write('Horizontal axis in kilometers ', img_x_axis)

st.write('Vertical axis in kilometers ', img_y_axis)


'''
## Once we have these, we will analyze the image and determine dunes vs no dunes

Analyzing Mars image
'''
if st.button('prediction'):
    # print is visible in server output, not in the page
    prediction_class= model.predict(resize(np.expand_dims(np.expand_dims(uploaded_file, axis=0), axis=3),[imgsize, imgsize])/255.)[0]
    st.write((prediction_class['prediction']))

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'



