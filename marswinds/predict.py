import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import math
from marswinds.utils import radian_to_degree, degree_to_radian
from tensorflow.keras.models import load_model
import joblib

class Predictor:
    
    def __init__(self, probability_threshold = 0.5):
        base_folder='raw_data/trained_models'
        self.regressor = load_model(f'{base_folder}/regressor.h5')
        self.classifier = load_model(f'{base_folder}/classifier.h5')
        self.scaler = joblib.load(f'{base_folder}/scaler.lib')
        self.probability_threshold = probability_threshold
    
    def prediction_from_models(self, tile):
        if len(tile.shape)==3:
            grey_image = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY) / 255.0
        else:
            grey_image = tile / 255.0

        expanded = np.expand_dims(np.expand_dims(grey_image,axis=0), axis=3)
        class_proba = self.classifier.predict(expanded)[0][0]
    
        if class_proba >= self.probability_threshold:
            predicted_regressed = self.regressor.predict(expanded)[0]
            wind_strength = predicted_regressed[0]
            sin = predicted_regressed[2]
            cos = predicted_regressed[1]
            angle_rad = math.atan2(sin, cos)
        else:
            angle_rad = np.nan
            wind_strength = np.nan
        
        return (class_proba,wind_strength, angle_rad)
    
    def add_arrows(self, img, ax, col, row, dim_factor):
    
        dune_proba, wind_strength_norm, original_angle = self.prediction_from_models(img)
    
        x_origin = col*256+128
        y_origin = row*256+128
    
        R=255
        G=204
        B=204
            
        if(wind_strength_norm>1):
            print('wind too high')
        
        if (wind_strength_norm<=0.2):
            R=255
            G=204
            B=204
        if (wind_strength_norm>0.2 and wind_strength_norm<=0.5):
            R=255
            G=128
            B=0
        if (wind_strength_norm>0.5 and wind_strength_norm<=0.7):
            R=0
            G=0
            B=153
        if (wind_strength_norm>0.7 and wind_strength_norm<=1.0):
            R=153
            G=0
            B=0
            
        color = [R/255,G/255,B/255]
    
        if dune_proba >= self.probability_threshold:
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
        
        dim_factor = 20/dim[0]
        dim_2 = dim[1]*dim_factor
    
        fig,ax = plt.subplots(figsize = (20,dim_2))

        extent = (0,dim[0],0,dim[1])
        ax.imshow(image, origin='upper', extent = extent, cmap='Greys')
    
        predictions = {"X-position (pixel)":[],
                "Y-position (pixel)":[],
                "Contains dunes?":[],   
                "Probability":[],
                "Wind Strength":[],
                "Wind Direction (degrees)":[]}
            
        results = pd.DataFrame.from_dict(predictions)
        for col in range(x):
            for row in range(y):

                d = {'current':[col*row],
                'total':[x*y]}

                pd.DataFrame.from_dict(d).to_csv('website/progress_log.csv')
                im = tiles[row*x+col]
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
                tiles.append(tile)
        
        trimmed_image = image[0:nb_h_tiles*256,0:nb_v_tiles*256]
        return (trimmed_image,(tiles,nb_h_tiles,nb_v_tiles))
    
    def get_prediction_image(self, image_path, pix_dim):
        pix_factor = 10/pix_dim

        if type(image_path)==str:
            image = cv2.imread(image_path)
        else:
            image = np.asarray(image_path)
            
        print(f'image shape:{image.shape}')
        dim=(int(image.shape[0]/pix_factor), int(image.shape[1]/pix_factor))

        resized_image = cv2.resize(image,dim)

        trimmed_image, tiles = self.prepare_tiles(resized_image)
        trimmed_dims = (trimmed_image.shape[0],trimmed_image.shape[1])
        results = self.create_figure(trimmed_image, tiles[0],tiles[1],tiles[2], trimmed_dims)
        
        return ('website/prediction/prediction.jpg',results)
    
if __name__ == '__main__':
    predictor = Predictor()
    image = cv2.imread('raw_data/mars_images/Murray-Lab_CTX-Mosaic_beta01_E-038_N-36.jpg')
    #image = cv2.imread('raw_data/images/testing/dunes/21.8424991607666_50.2474983215332_031_CW000_-0.6031867265701294_-0.6447361707687378_1.9967776536941528.jpg')
    #image = cv2.imread('raw_data/mars_images/Mars_small_new.jpg')
    path,df = predictor.get_prediction_image(image,10)
    print(df)