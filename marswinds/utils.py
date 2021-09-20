import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random, os

def radian_to_degree(rad_angle):
    return rad_angle * 180 / math.pi

def degree_to_radian(deg_angle):
    return deg_angle / 180 * math.pi

def sin_cos_to_degrees(angular_sin, angular_cos):
    original_angle = radian_to_degree(math.atan2(angular_sin, angular_cos))
    if original_angle < 0: original_angle += 360
    
    return original_angle

def decode_angle(image_name):
    label = image_name.split('/')[-1]
    original_sin = float(label.split('_')[-3])
    original_cos = float(label.split('_')[-2])
    original_angle = sin_cos_to_degrees(original_sin, original_cos)
    
    return original_angle

def encode_angle(new_angle):
    if new_angle > 360: new_angle -= 360
    rad_angle = math.pi * new_angle / 180
    new_sin = np.sin(rad_angle)
    new_cos = np.cos(rad_angle)
    return (new_sin, new_cos)

def plot_image(folder_name='training',image_type='dunes', image_name=None, ax=None):
    
    if ax == None:
        fig, ax = plt.subplots(1,1,figsize=(4,4))
        
    folder_path = f'../images/{folder_name}/{image_type}'
    if image_name == None:
        image_name = random.choice(os.listdir(folder_path))
        path = f'{folder_path}/{image_name}'
    else:
        path = f'{folder_name}/{image_name}'
        
        
    data = image_name.split('_')
    wind_strength = float(data[-1].split('.')[0])
    original_sin = float(data[-3])
    original_cos = float(data[-2])
    original_angle = math.atan2(original_sin, original_cos)
    original_angle = radian_to_degree(original_angle)
    arrow_length = 64 
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

    print(path)
    img = cv2.imread(path)

    ax.imshow(img, cmap='Greys')
    ax.arrow(tail_x,tail_y,dx,dy,color='r',
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
    