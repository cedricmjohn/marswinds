FROM python:3.8.6-buster

COPY app.py /app.py 
COPY notebooks/raw_data/models_regressionXception_0.13006779551506042__0.13006779551506042.h5 /notebooks/raw_data/models_regressionXception_0.13006779551506042__0.13006779551506042.h5
COPY trainer /trainer
COPY images_large_balanced.zip /images_large_balanced.zip
COPY website /website
COPY requirements.txt /requirements.txt
RUN apt-get update 

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-install libgl1-mesa-glx -y 
RUN apt-get install ffmpeg libsm6 libxext6  -y

CMD streamlit run app.py  --server.port 8080

