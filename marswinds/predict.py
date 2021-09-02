import cv2
import os
from os import listdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from marswinds.utils import radian_to_degree, degree_to_radian
from tensorflow.keras.models import load_model
import joblib

class Predictor:
    
    def __init__(self, probability_threshold = 0.97):
        print(os.getcwd())
        base_folder='../raw_data/trained_models'
        self.regressor = load_model(f'{base_folder}/regressor/regressor_210720_tfl_Xception.h5')
        self.classifier = load_model(f'{base_folder}/classifier/classifier_210719_Xception.h5')
        self.scaler = joblib.load(f'{base_folder}/regressor/scaler.joblib')
        self.prediction_path = '../website/prediction/prediction.jpg'
        self.image_name = ''
        self.probability_threshold = probability_threshold
        self.initialize_results()
        
    
    def initialize_results(self):
        predictions_fields = {
                "Image":[],
                "X-position (pixel)":[],
                "Y-position (pixel)":[],
                "Contains dunes?":[],   
                "Probability":[],
                "Wind Strength":[],
                "Wind Direction (degrees)":[]}
        self.predictions = pd.DataFrame.from_dict(predictions_fields)
        
    def prediction_from_models(self, tile):
    
        expanded = np.expand_dims(np.expand_dims(tile,axis=0), axis=3)
        class_proba = self.classifier.predict(expanded)[0][0]
    
        if class_proba >= self.probability_threshold:
            tile_rgb = np.stack([tile, tile, tile]).T
            expanded_rgb = np.expand_dims(tile_rgb,axis=0)
            predicted_regressed = self.regressor.predict(expanded_rgb)[0]
            wind_strength = predicted_regressed[0]
            sin = predicted_regressed[1]
            cos = predicted_regressed[2]
            angle_rad = math.atan2(sin, cos)
        else:
            angle_rad = np.nan
            wind_strength = np.nan
        
        return (class_proba,wind_strength, angle_rad)
    
    def add_arrows(self, img, ax, col, row, dim_factor):
    
        dune_proba, wind_strength_norm, original_angle = self.prediction_from_models(img)
    
        x_origin = col*256+128
        y_origin = row*256+128
    
        R=(dune_proba-self.probability_threshold)/(1-self.probability_threshold)
        G=0
        B=1-R
         
        color = [R,G,B]
    
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
    
    def create_figure(self, tiled_data):
        image = tiled_data.get('trimmed image')
        tiles = tiled_data.get('tiles')
        nb_h_tiles = tiled_data.get('nb horizontal tiles')
        nb_v_tiles = tiled_data.get('nb vertical tiles')
        print(f'V:{nb_v_tiles}, h:{nb_h_tiles}')
        dim = image.shape
        dim_factor = 256/96
        dim_2 = dim[0]*dim_factor
         
        fig = plt.figure(dpi=96)
        fig.set_size_inches(nb_h_tiles, nb_v_tiles, forward = False)
        ax = plt.Axes(fig, [0., 0., nb_h_tiles, nb_v_tiles])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        ax.imshow(image, origin='upper',
                  cmap='Greys')
     
        for row in tqdm(range(nb_v_tiles)):
            for col in tqdm(range(nb_h_tiles)):
                im = tiles[row*nb_h_tiles+col]
                predicted_values = self.add_arrows(img=im, ax=ax, col=col, row=row,dim_factor=dim_factor)
            
                x_origin=predicted_values[0] 
                y_origin=predicted_values[1] 
                dune_proba=predicted_values[2] 
                wind_strength=predicted_values[3]
                angle=predicted_values[4]
            
                prediction = {
                    "Image":[self.prediction_path],
                    "X-position (pixel)":[x_origin],
                    "Y-position (pixel)":[y_origin],
                    "Contains dunes?":[bool(dune_proba>=self.probability_threshold)],   
                    "Probability":[dune_proba],
                    "Wind Strength":[wind_strength],
                    "Wind Direction (degrees)":[angle]}
            
                self.predictions = pd.concat([self.predictions, pd.DataFrame.from_dict(prediction)])

        
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
        plt.savefig(self.prediction_path, 
                    dpi=256,  
                    bbox_inches = 'tight',     
                    pad_inches = 0,
                    format='jpg')
        
        return self.predictions
    
    def prepare_tiles(self, image, original_image):
        nb_v_tiles = int((image.shape[0]-image.shape[0]%256)/256)
        nb_h_tiles = int((image.shape[1]-image.shape[1]%256)/256)
        tiles = []
        
        for x in range(nb_v_tiles):
            for y in range(nb_h_tiles):
                tile = image[x*256:(x+1)*256,y*256:(y+1)*256]
                tiles.append(tile)

        
        trimmed_image = original_image[0:nb_v_tiles*256,0:nb_h_tiles*256]
   
        tiled_data = {
            "trimmed image":trimmed_image,
            'tiles':tiles,
            'nb horizontal tiles':nb_h_tiles,
            'nb vertical tiles':nb_v_tiles,
        }
        return tiled_data
    
    def get_prediction_image(self, image_path, pix_dim):
        pix_factor = 10/pix_dim

        if type(image_path)==str:
            original_image = cv2.imread(image_path)
            grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)/255.
        else:
            original_image = np.asarray(image_path)
            grayscale_image = np.asarray(image_path)
        
        prediction_path_elements = image_path.split("/")
        self.image_name = prediction_path_elements[-1]
        prediction_folder_path = prediction_path_elements[:-1]
        prediction_folder_path.append("predictions")
        prediction_folder_path = "/".join(prediction_folder_path)
        
        try:
            os.mkdir(prediction_folder_path)
        except:
            pass
        
        self.prediction_path = f"{prediction_folder_path}/{self.image_name}_pred.jpg"

        
        dim=(int(grayscale_image.shape[1]/pix_factor), int(grayscale_image.shape[0]/pix_factor))

        resized_image = cv2.resize(grayscale_image,dim)
        resized_original_image = cv2.resize(original_image,dim)

        tiled_data = self.prepare_tiles(resized_image,resized_original_image)
        results = self.create_figure(tiled_data)
        
        return (results, prediction_folder_path)
    
    def predict_from_file(self, path, pixel_resolution=10):
        results, prediction_folder_path =  self.get_prediction_image(path, pixel_resolution)
        results.to_csv(f'{prediction_folder_path}/predictions.csv', index=False)
        
        return results
    
    def predict_from_folders(self, foldernames, labelled=False, pixel_resolution=10):
        if type(foldernames)==str:
            foldernames = [foldernames]
        print('STARTING TO ITERATE FOLDER')
        
        for folder in tqdm(foldernames):
            print(f'Now predicting images in folder "{folder.split("/")[-1]}"')
            print(folder)
            files = [f for f in listdir(folder)]
            print(files)
            for file in tqdm(files):
                try:
                    path = f'{folder}/{file}'
                    _,_ =  self.get_prediction_image(path, pixel_resolution)
                except:
                    pass
            prediction_folder_path = self.prediction_path.split("/")[:-1] 
            prediction_folder_path = "/".join(prediction_folder_path)
            
            if labelled:
                pass
            
            self.predictions.to_csv(f'{prediction_folder_path}/predictions.csv', index=False)
            self.initialize_results()
            
        return None
   
    
if __name__ == '__main__':
    predictor = Predictor(probability_threshold=0.5)
    #image = '../raw_data/mars_images/Murray-Lab_CTX-Mosaic_beta01_E-038_N-36.jpg'
    #image = '../raw_data/images/testing/dunes/21.8424991607666_50.2474983215332_031_CW000_-0.6031867265701294_-0.6447361707687378_1.9967776536941528.jpg'
    #image = '../raw_data/mars_images/demo.jpg'
    #image = '../raw_data/mars_images/alt/Murray-Lab_CTX-Mosaic_beta01_E-178_N-04.jpg'
    #image = '../raw_data/mars_images/control/Murray-Lab_CTX-Mosaic_beta01_E-038_N-36.jpg'
    #image = '../raw_data/moon_images/SosigenesCrater.png'
    #image = '../raw_data/earth_images/landsat-106.473427_32.923187_-106.27342700000001_33.123187.png'
    #image = '../raw_data/mars_images/alt/Mars_small_new.jpg'
    #path,df = predictor.get_prediction_image(image,10)
    #predictor.predict_from_file('../raw_data/earth_images/landsat-106.473427_32.923187_-106.27342700000001_33.123187.png')
    predictor.predict_from_folders('../raw_data/images/testing/earth_test_small')