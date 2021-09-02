import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import math
from marswinds.utils import radian_to_degree, degree_to_radian
from tensorflow.keras.models import load_model
import joblib
import random
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

class TestModelPredictions:
    def __init__(self, data):
        self.data = data
        self.data["Actual wind direction"] = []
        self.data["Actual wind strength"] = []

    def get_label(self,image_name):
        parts = image_name[:-4].split('_')
        sin = float(parts[-3])
        cos = float(parts[-2])
        angle = radian_to_degree(math.atan2(sin, cos))
        if angle<0: angle+=360
        wind_strength = float(parts[-1])
    
        return(wind_strength, angle)

    def calculate_angular_diff(self, label_angle, pred_angle):
        if label_angle <0:
            label_angle = 360+label_angle
        if pred_angle <0:
            pred_angle = 360+pred_angle
    
        angle_diff = np.abs(label_angle-pred_angle)
        if angle_diff > 180: angle_diff = 360-angle_diff
        
        return angle_diff

    def get_diff(self, files, path=False):
    
     for file in tqdm(files):
            label_wind, label_angle = self.get_label(file)
            pred_angle = self.data["Wind Direction (degrees)"].values
            pred_wind = self.data["Wind Strength"].values
        
            if pred_angle <0:
                pred_angle = 360+pred_angle
        
            wind_diff = np.abs(label_wind-pred_wind)
            angle_diff = self.calculate_angular_diff(label_angle, pred_angle)
            
        d = {'filename':[file],
        'angular_diff':[angle_diff**2],
        'speed_diff':[wind_diff**2],
        'predicted_angle':[pred_angle],
        'actual_angle':[label_angle],
        'predicted_strength':[pred_wind],
        'actual_strength':[label_wind]}

        pd.concat([pd.read_csv(log_name), pd.DataFrame.from_dict(d)]).to_csv(log_name, index=False)
    
        return pd.read_csv(log_name)

def add_arrow(ax, wind_strength, original_angle, color):
    
    arrow_length = int(64 * wind_strength / 2)
    if original_angle < 0: original_angle += 360
    
    if (original_angle<=90):
        arr_angle = degree_to_radian(original_angle)
        adj = np.cos(arr_angle)*arrow_length
        op = np.sin(arr_angle)*arrow_length
        dx = op*2
        dy = -adj*2
        tail_x = 128-op
        tail_y = adj+128
    if (original_angle<=180) & (original_angle>90):
        arr_angle = degree_to_radian(original_angle-90)
        adj = np.cos(arr_angle)*arrow_length
        op = np.sin(arr_angle)*arrow_length
        dx = adj*2
        dy = op*2
        tail_x = 128-adj
        tail_y = 128-op
    if (original_angle<=270) & (original_angle>180):
        arr_angle = degree_to_radian(original_angle-180)
        adj = np.cos(arr_angle)*arrow_length
        op = np.sin(arr_angle)*arrow_length
        dx = -op*2
        dy = adj*2
        tail_x = 128+op
        tail_y = 128-adj
    if (original_angle<=360) & (original_angle>270):
        arr_angle = degree_to_radian(original_angle-270)
        adj = np.cos(arr_angle)*arrow_length
        op = np.sin(arr_angle)*arrow_length
        dx = -adj*2
        dy = -op*2
        tail_x = 128+adj
        tail_y = 128+op

    ax.arrow(tail_x,tail_y,dx,dy,color=color,
         length_includes_head=True,
         width=wind_strength*5,
        alpha=.3)
    
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    
    ax.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        right=False,         # ticks along the top edge are off
        labelleft=False) # labels along the bottom edge are off
    
    return ax


def plot_image(folder_name='testing',image_type='dunes', selected_results=None, image_name=None, ax=None, full_path=False):
    
    if ax == None:
        fig, ax = plt.subplots(1,1,figsize=(4,4))
        
    folder_path = f'../raw_data/images/{folder_name}/{image_type}'
    
    if image_name == None:
        image_name = random.choice(selected_results.filename.values)
    
    if full_path:
        path = image_name
    else:
        path = f'{folder_path}/{image_name}'
        
    data = image_name.split('_')
    wind_strength = float(data[-1].split('.')[0])
    original_sin = float(data[-3])
    original_cos = float(data[-2])
    original_angle = math.atan2(original_sin, original_cos)
    original_angle = radian_to_degree(original_angle)
    
    img = cv2.imread(path)
    ax.imshow(img, cmap='Greys')
    
    add_arrow(ax, wind_strength, original_angle, 'r')
    
    wind_strength, wind_direction =  prediction_from_models(image_name, path=full_path)
    
    add_arrow(ax, wind_strength, wind_direction, 'b')
    
    return ax

def plot_rand_image_grid(grid_dim=5,image_dim=3, selected_results=None, full_path=False, folder_name='testing', image_type='dunes'):
    fig, axes = plt.subplots(grid_dim,grid_dim,figsize=(grid_dim*image_dim,grid_dim*image_dim))
    for ax in axes.flatten():
        plot_image(ax=ax, selected_results=selected_results, full_path = full_path, folder_name=folder_name, image_type=image_type)
    fig.tight_layout()
    return fig
        
def prediction_from_Xception(image):
        try:
            predicted_proba=classifier.predict(image)[0][0]
            if predicted_proba > 0.5:
                predicted_class = 'dunes'
            else:
                predicted_class = 'no_dunes'
        except Exception as e:
            print(f'Problem with file: {e.args}')
            return (0,0)
        
        return (predicted_proba, predicted_class)


def convert_label(label):
    if label > 0:
        label = 'dunes'
    else:
        label = 'no_dunes'
    return label

log_name = '../raw_data/logs/testing_regressor_tfl.csv'

d = {'filename':[],
        'angular_diff':[],
        'speed_diff':[],
    'predicted_angle':[],
    'actual_angle':[],
    'predicted_strength':[],
    'actual_strength':[]}
df = pd.DataFrame.from_dict(d).to_csv(log_name, index=False)

test_files = [f for f in listdir(dune_test_path) if len(f)>30]
labels = [get_label(f) for f in test_files]

get_diff(test_files)

max_error = 45
plot_rand_image_grid(image_dim=5,selected_results = res[res.angular_diff >= max_error**2])

