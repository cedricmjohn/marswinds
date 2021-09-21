from marswinds.utils import radian_to_degree, degree_to_radian
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class PredictionImage():
    def __init__(self, image, prediction_path, predictions=None):
        self.predictions = predictions
        self.image = image
        self.create_base_image(self.image)
        self.prediction_path = prediction_path
        
    
    def create_base_image(self, image):
        nb_h_tiles = image.shape[1]/256
        nb_v_tiles = image.shape[0]/256
        
        larger_side = nb_v_tiles
        if nb_h_tiles > nb_v_tiles:
            larger_side = nb_h_tiles
        self.pixel_ratio = larger_side / (nb_h_tiles*nb_v_tiles)
        
        fig = plt.figure(dpi=96)
        fig.set_size_inches(nb_h_tiles, nb_v_tiles, forward = False)
        ax = plt.Axes(fig, [0., 0., nb_h_tiles,nb_v_tiles])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        ax.set_axis_off()
        
        ax.imshow(image, origin='upper',
                  cmap='gray')
        
        ax.tick_params(
                axis='x',          
                which='both',      
                bottom=False,      
                top=False,         
                labelbottom=False)
        ax.tick_params(
                axis='y',          
                which='both',      
                bottom=False,      
                top=False,         
                labelbottom=False) 
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        
        self.figure = fig
        return self
        
    def add_predictions(self):
        fig = self.figure
        ax = fig.axes[0]
        
        for _, row in self.predictions.iterrows():
            if row['is_dune'] == 1:
                self.add_arrows(ax,
                        row['angle'],
                           row['x_pixels'],
                           row['y_pixels'],
                           arrow_type='prediction')
        
        self.figure = fig
        self.save_figure()
        
        return self
    
    def add_labels(self,labels):
        fig = self.figure
        ax = fig.axes[0]
        
        for _, row in self.labels.iterrows():
            self.add_arrows(ax,
                        row['angle'],
                           row['x_pixels'],
                           row['y_pixels'],
                           arrow_type='labels')
        
        self.figure = fig
        self.save_figure()
        
        return self
    
    def add_arrows(self, ax, 
                   original_angle,
                   x_origin,
                   y_origin,
                   arrow_type = 'prediction'):
    
        color = []
        if arrow_type == 'prediction':
            color = [.9,.0,.1]
        else:
            color = [.1, .1, .8]
            
        arrow_length = 100
        if original_angle < 0: original_angle += 360
        if original_angle >360: original_angle -+360
        
        if (original_angle<=90):
            arr_angle = degree_to_radian(original_angle)
            adj = np.cos(arr_angle)*arrow_length
            op = np.sin(arr_angle)*arrow_length
            dx = op*2
            dy = -adj*2
            tail_x = x_origin-op
            tail_y = adj+y_origin
        if (original_angle<=180) & (original_angle>90):
            arr_angle = degree_to_radian(original_angle-90)
            adj = np.cos(arr_angle)*arrow_length
            op = np.sin(arr_angle)*arrow_length
            dx = adj*2
            dy = op*2
            tail_x = x_origin-adj
            tail_y = y_origin-op
        if (original_angle<=270) & (original_angle>180):
            arr_angle = degree_to_radian(original_angle-180)
            adj = np.cos(arr_angle)*arrow_length
            op = np.sin(arr_angle)*arrow_length
            dx = -op*2
            dy = adj*2
            tail_x = x_origin+op
            tail_y = y_origin-adj
        if (original_angle<=360) & (original_angle>270):
            arr_angle = degree_to_radian(original_angle-270)
            adj = np.cos(arr_angle)*arrow_length
            op = np.sin(arr_angle)*arrow_length
            dx = -adj*2
            dy = -op*2
            tail_x = x_origin+adj
            tail_y = y_origin+op
        try:
            ax.arrow(tail_x,tail_y,dx,dy,color=color,
            length_includes_head=True,
            width=20,alpha=.6)
        except Exception as e:
            print(f'The following error was raised when trying to plot arrows:{e.args}')
            print(f'The predicted angle is:{original_angle}')

        return self
    
    def save_figure(self):
        plt.figure(self.figure.number)
        plt.savefig(self.prediction_path, 
                    dpi=256*self.pixel_ratio,  
                    bbox_inches = 'tight',     
                    pad_inches = 0,
                    format='jpg')
    
    