FROM python:3.8.6-buster

COPY app.py /app.py 
COPY marswinds /marswinds
COPY raw_data /raw_data
COPY website /website
COPY requirements.txt /requirements.txt

RUN mkdir $USER/.streamlit
RUN apt-get update 

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt install libgl1-mesa-glx -y 

COPY config.toml $USER/.streamlit/config.toml 

CMD streamlit run app.py  --server.port 8080
