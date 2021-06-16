import streamlit as st
from tensorflow import keras
from tensorflow.image import resize
import numpy as np
import pandas as pd
from PIL import Image
import time
import base64
import imageio
st.set_page_config(layout="wide", 
                    page_icon=":earth_africa:", 
                    page_title="Mapping Martian Winds", 
                     initial_sidebar_state="expanded")
# image_1=None
# # pixel_dim=None
imgsize = 256
prediction_path='website/prediction/prediction.jpg'
#Information
info= ['Pixel dimension', 'Horizontal/vertical side dimension']

side= ['Horizontal side', 'Vertical side']

#Import model
model = keras.models.load_model("notebooks/raw_data/models_regressionXception_0.13006779551506042__0.13006779551506042.h5")

#logo at the top of page
image = Image.open('website/ESP_015917_2650.jpg')
st.image(image, caption='', use_column_width=True)

#Create sidebar
st.sidebar.markdown("""
# Martian Winds
Please upload a photo of Mars to predict wind direction
    """)



# info_option = st.sidebar.selectbox('Insert image information', info)
# def image_info(info_option):
#     if info_option == 'Pixel dimension':
#         pixel_dim = st.sidebar.number_input('Insert pixel dimension of your image (METERS)', 0.000)
#         st.image(uploaded_file, caption=f"Pixel dimension: {pixel_dim} m/px", use_column_width=True)
#         pred_image = Image.open(prediction_path)
#         return st.image(pred_image, caption='', use_column_width=True)
#     if info_option == "Horizontal/vertical side dimension":
#         side_option = st.sidebar.selectbox('Insert image side', side)
#         if side_option == 'Horizontal side':
#             pixel_dim=img_axis/uploaded_file.size[0]
#             st.image(uploaded_file, caption=f"Pixel dimension: {pixel_dim} m/px", use_column_width=True)
#             pred_image = Image.open(prediction_path)
#             return st.image(pred_image, caption='', use_column_width=True)
#         if side_option == 'Vertical side':
#             pixel_dim=img_axis/uploaded_file.size[1]
#             st.image(uploaded_file, caption=f"Pixel dimension: {pixel_dim} m/px", use_column_width=True)
#             pred_image = Image.open(prediction_path)
# #             return st.image(pred_image, caption='', use_column_width=True)
# @st.cache(suppress_st_warning=True, allow_output_mutation=True)
# def px_dimension():
#     if st.sidebar.checkbox('Insert pixel dimension'):
#         pixel_dimension = st.sidebar.number_input('Insert pixel dimension of your image (METERS)', 0.000)
#         return pixel_dimension
# px_dimension()

if st.sidebar.checkbox('Insert pixel dimension'):
    # @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    pixel_dimension = st.sidebar.number_input('Insert pixel dimension of your image (METERS)', 0.000)
st.sidebar.markdown(" **OR**")

if st.sidebar.checkbox('Insert side dimension of image'):
    option = st.sidebar.selectbox('Insert side dimension of your image', ['Horizontal side', 'Vertical side'])
    img_axis = st.sidebar.number_input('Insert METERS represented by side of the image', 0.0)

uploaded_file = st.sidebar.file_uploader("Choose a photo of dunes in Mars*")
if uploaded_file:
    image_1 = imageio.imread(uploaded_file)

st.sidebar.markdown(f"""
/* Mandatory field
    """)

st.markdown("<h1 style='text-align: center;'>Mapping Martian Winds</h1>", unsafe_allow_html=True)


#Description text

st.markdown('''
Bla Bla Bla 
''')


# #Get user photo input
# if uploaded_file is not None:
#     if st.sidebar.checkbox('Insert pixel dimension')==True:
#         st.image(uploaded_file, caption=f"Pixel dimension: {pixel_dim} m/px", use_column_width=True)
#         pred_image = Image.open(prediction_path)
#         st.image(pred_image, caption='', use_column_width=True)
#     if st.sidebar.checkbox('Insert side dimension of image')==True:
#         if option=='Horizontal side':
#             pixel_dim=img_axis/uploaded_file.size[0]
#             st.image(uploaded_file, caption=f"Pixel dimension: {pixel_dim} m/px", use_column_width=True)
#             pred_image = Image.open(prediction_path)
#             st.image(pred_image, caption='', use_column_width=True)
#         if option=='Vertical side':
#             pixel_dim=img_axis/uploaded_file.size[1]
#             st.image(uploaded_file, caption=f"Pixel dimension: {pixel_dim} m/px", use_column_width=True)
#             pred_image = Image.open(prediction_path)
#             st.image(pred_image, caption='', use_column_width=True)
# else:
#     st.markdown('Please upload image and fill information requested')

if image_1 is not None:

    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def px_dim(pixel_dimension):
        if pixel_dimension: 
            pixel_dim = pixel_dimension 
            return pixel_dim
        else:
            return None
    pixel_dim_1 =px_dim(pixel_dimension)
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def px_dim_side(option):
        if option =="Horizontal side":
            pixel_dim = img_axis/image_1.shape[0] #error here, uploaded_file type? 
            return pixel_dim
        if option =="Vertical side":
            pixel_dim = img_axis/image_1.shape[1] #error here, uploaded_file type? 
            return pixel_dim
    pixel_dim = px_dim_side(option)
    
    # @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    # def show_image(uploaded_file):
        # return 
    st.image(uploaded_file, caption=f"Pixel dimension: {pixel_dim} m/px", use_column_width=True)

    prediction = st.button('Prediction')
    if prediction:
        with st.spinner('Wait for it...'):
            time.sleep(5)
            st.success('Done!')
        response= "hello"
        
        pred_image = Image.open(prediction_path)
        st.image(pred_image, caption=f"Pixel dimension: {pixel_dim} m/px", use_column_width=True)
        st.write((response))
        download=st.button('Download data')
        if download:
            'Download started'
            list = ['A','B','C']
            data = pd.DataFrame(list) #result from function
            data.columns=['Title']
            data
            csv=data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings
            link= f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download csv file</a>'
            st.markdown(link, unsafe_allow_html=True)
            st.image(pred_image, caption='', use_column_width=True)
else:
    st.markdown('Please upload image and fill information requested')

    









'''
## Once we have these, we will analyze the image and determine dunes vs no dunes

Analyzing Mars image
'''
# if st.button('prediction'):
#     # print is visible in server output, not in the page
#     prediction_class= model.predict(resize(np.expand_dims(np.expand_dims(uploaded_file, axis=0), axis=3),[imgsize, imgsize])/255.)[0]
#     st.write((prediction_class['prediction']))

# Add a placeholder
# latest_iteration = st.empty()
# bar = st.progress(0)

# for i in range(100):
#   # Update the progress bar with each iteration.
#   latest_iteration.text(f'Iteration {i+1}')
#   bar.progress(i + 1)
#   time.sleep(0.1)

# '...and now we\'re done!'



