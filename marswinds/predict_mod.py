import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import math
from marswinds.utils import radian_to_degree, degree_to_radian
from tensorflow.keras.models import load_model
import joblib
import imageio

class Predictor:
    
    def __init__(self):
        base_folder='raw_data/trained_models'
        self.regressor = load_model(f'{base_folder}/regressor.h5')
        self.classifier = load_model(f'{base_folder}/classifier.h5')
        self.scaler = joblib.load(f'{base_folder}/scaler.lib')
    
    def prediction_from_models(self, tile):
        
        if len(tile.shape)>=3:
            grey_image = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY) / 255.0
        else:
            grey_image = tile
        
        expanded = np.expand_dims(np.expand_dims(grey_image,axis=0), axis=3)
        class_proba = self.classifier.predict(expanded)[0][0]
    
        if class_proba <= 0.5:
            predicted_regressed = self.regressor.predict(expanded)[0]
            wind_strength = predicted_regressed[0]
            sin = predicted_regressed[1]
            cos = predicted_regressed[0]
            angle_rad = math.atan2(sin, cos)
        else:
            angle_rad = np.nan
            wind_strength = np.nan
        
        return (class_proba,wind_strength, angle_rad)
    
    def add_arrows(self, img, ax, col, row, dim_factor):
    
        dune_proba, wind_strength_norm, original_angle = self.prediction_from_models(img)
    
        y_origin = col*256+128
        x_origin = row*256+128
    
        red_component = wind_strength_norm
        blue_component= 1 - red_component
    
        color = [red_component, 0, blue_component]
        color = 'r'
        if dune_proba <= 0.5:
            original_angle = radian_to_degree(original_angle)
            arrow_length = 100
            if original_angle < 0: original_angle += 360
    
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
            ax.arrow(tail_x,tail_y,dx,dy,color=color,
             length_includes_head=True,
             width=20,alpha=.6)
            wind_strength = self.scaler.inverse_transform(np.array([[wind_strength_norm]]))[0][0]
        else:
            wind_strength = np.nan
        
        return (x_origin, y_origin, dune_proba, wind_strength, original_angle)
    
    def create_figure(self, image, tiles, x, y, dim):
        
        if dim[0] > dim [1]:
            dim_factor = 20/dim[0]
        else:
            dim_factor = 20/dim[1]
            
        dim_1 = dim[0]*dim_factor
        dim_2 = dim[1]*dim_factor
        
        print(dim_1)
        print(dim_2)
    
        fig,ax = plt.subplots(figsize = (dim_1,dim_2))
    
        ax.imshow(image, cmap='Greys')
    
        predictions = {"X-position (pixel)":[],
                "Y-position (pixel)":[],
                "Contains dunes?":[],   
                "Probability":[],
                "Wind Strength":[],
                "Wind Direction (degrees)":[]}
            
        results = pd.DataFrame.from_dict(predictions)
    
        for col in range(x):
            for row in range(y):

                im = tiles[col*y+row]
                predicted_values = self.add_arrows(img=im, ax=ax, col=col, row=row,dim_factor=dim_factor)
            
                x_origin=predicted_values[0] 
                y_origin=predicted_values[1] 
                dune_proba=predicted_values[2] 
                wind_strength=predicted_values[3]
                angle=predicted_values[4]
            
                prediction = {"X-position (pixel)":[x_origin],
                    "Y-position (pixel)":[y_origin],
                    "Contains dunes?":[dune_proba>=0.5],   
                    "Probability":[dune_proba],
                    "Wind Strength":[wind_strength],
                    "Wind Direction (degrees)":[angle]}
            
                results = pd.concat([results, pd.DataFrame.from_dict(prediction)])

        
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
    
        plt.savefig('website/prediction/prediction.jpg', format='jpg')
        
        return results
    
    def prepare_tiles(self, image):
        nb_h_tiles = int((image.shape[0]-image.shape[0]%256)/256)
        nb_v_tiles = int((image.shape[1]-image.shape[1]%256)/256)
        
        tiles = []
  
        for x in range(nb_h_tiles):
            for y in range(nb_v_tiles):
                tile = image[x*256:(x+1)*256,y*256:(y+1)*256]
                print(tile.shape)
                tiles.append(tile)
                
        return (tiles,nb_h_tiles,nb_v_tiles)
    
    def get_prediction_image(self, image_path, pix_dim):
        pix_factor = 10/pix_dim

        if type(image_path)==str:
            image = cv2.imread(image_path)
        else:
            image = np.asarray(image_path)

        dim=(int(image.shape[0]/pix_factor), int(image.shape[1]/pix_factor))

        resized_image = cv2.resize(image,dim)

        tiles = self.prepare_tiles(resized_image)
        results = self.create_figure(resized_image, tiles[0],tiles[1],tiles[2], dim)
        
        return ('website/prediction/prediction.jpg',results)
    
if __name__ == '__main__':
    predictor = Predictor()
    image = imageio.imread('raw_data/mars_images/Blue-Dunes-on-Mars_r.jpg')
    predictor.get_prediction_image(image,10)