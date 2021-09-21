from data.wind_data import WindData
from data.prep_data import DataPreparation
import cv2
import os
from os import listdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import math
from tensorflow.python.keras.backend import resize_images
from tqdm import tqdm
from marswinds.utils import radian_to_degree, sin_cos_to_degrees
from marswinds.plot.plot_images import PredictionImage
from tensorflow.keras.models import load_model
import joblib

class Predictor:
    
    def __init__(self, probability_threshold = 0.97, planet='EARTH'):
        base_folder='../raw_data/trained_models'
        self.regressor = load_model(f'{base_folder}/regressor/regressor_210720_tfl_Xception.h5')
        self.classifier = load_model(f'{base_folder}/classifier/classifier_210719_Xception.h5')
        self.scaler = joblib.load(f'{base_folder}/regressor/scaler.joblib')
        self.prediction_path = '../website/prediction/prediction.jpg'
        self.image_name = ''
        self.probability_threshold = probability_threshold
        self.planet = planet
        self.initialize_results()
        
    
    def initialize_results(self):
        predictions_fields = {
                "Image":[],
                "x_pixels":[],
                "y_pixels":[],
                "is_dunes":[],   
                "probability":[],
                "strength":[],
                "angle":[]}
        self.predictions = pd.DataFrame.from_dict(predictions_fields)
        self.image_objects = []
        
        
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
        
        return (class_proba,wind_strength, radian_to_degree(angle_rad))
    
    
    def open_and_scale_image(self,image_path, pix_dim):
        pix_factor = 10/pix_dim

        if type(image_path)==str:
            original_image = cv2.imread(image_path)
            grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)/255.
        else:
            original_image = np.asarray(image_path)
            grayscale_image = np.asarray(image_path)
        
        dim=(int(grayscale_image.shape[1]/pix_factor), int(grayscale_image.shape[0]/pix_factor))

        resized_image = cv2.resize(grayscale_image,dim)
        resized_original_image = cv2.resize(original_image,dim)
        
        return resized_image, resized_original_image
    
    
    def create_prediction_folder(self, image_path):
        
        prediction_path_elements = image_path.split("/")
        self.image_name = prediction_path_elements[-1]
        prediction_folder_path = prediction_path_elements[:-1]
        prediction_folder_path.append(f"predictions_{self.image_name}")
        prediction_folder_path = "/".join(prediction_folder_path)
        
        try:
            os.mkdir(prediction_folder_path)
        except:
            pass
        
        self.prediction_folder = prediction_folder_path
        self.prediction_path = f"{prediction_folder_path}/{self.image_name}.jpg"
        
        return prediction_folder_path
        
    def get_coordinates(self,base_coordinates, tile_coordinates, image_size,pixel_resolution=10,planet='EARTH'):
        planet = planet.capitalize()
        base_latitude = base_coordinates[0]
        base_longitude = base_coordinates[1]
        
        METERS_TO_DEGREES = {
            'EARTH':1/111000,
            'MARS':180/6794000,
        }
        
        x = tile_coordinates[0]
        y = tile_coordinates[1]
        
        x_center = image_size[0]/2
        y_center = image_size[1]/2
        
        delta_x = x-x_center
        delta_y = y-y_center
         
        degrees_to_pixels = METERS_TO_DEGREES.get(planet,0.001/111) * pixel_resolution
        
        lat = base_latitude - delta_x * degrees_to_pixels
        long = base_longitude + delta_y * degrees_to_pixels * (90-abs(base_latitude))/90
        
        if lat>90:
            lat=180-(lat)
        if lat<-90:
            lat=-180-(lat)
        if long > 180:
            long = 360-(long)
        if long < -180:
            long = -360-(long)
        
        return (lat, long)
    
    def get_coordinates_from_filename(self,filename):
        file_tokens = filename.split('_')
        return (float(file_tokens[0]),float(file_tokens[1]))
    
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
            "full image":original_image,
            'tiles':tiles,
            'nb horizontal tiles':nb_h_tiles,
            'nb vertical tiles':nb_v_tiles,
        }
        return tiled_data
    
    def save_tiled_data(self, image_path, 
                        pixel_resolution=10, 
                        coordinates=None,
                        label_wind=True):
        
        prediction_folder_path = self.create_prediction_folder(image_path)
        resized_image,resized_original_image = self.open_and_scale_image(image_path,pixel_resolution)
        
        try:
            os.mkdir(f"{prediction_folder_path}/tiles")
        except:
            pass
        
        tiled_data = self.prepare_tiles(resized_image,resized_original_image)
        x = tiled_data.get('nb horizontal tiles')
        y= tiled_data.get('nb vertical tiles')
        tiles = tiled_data.get('tiles')
        trimmed_image = tiled_data.get('trimmed image')
        full_image = tiled_data.get('full image')
    
        PredictionImage(trimmed_image,
                        self.prediction_path).save_figure()
        
        labels = {"filename":[],
                "x_pixels":[],
                "y_pixels":[],   
                "latitude":[],
                "longitude":[],
                "is_dune_label":[],   
                "strength_label":[],
                "angle_label":[]}
            
        results = pd.DataFrame.from_dict(labels)
                
        for col in range(y):
            for row in range(x):
                im = tiles[col*x+row]*255
                x_origin=row*256+128
                y_origin=col*256+128
                
                if coordinates is not None:
                    latitude, longitude = self.get_coordinates(
                        base_coordinates=coordinates,
                        tile_coordinates = (x_origin,y_origin),
                        image_size = full_image.shape,
                        pixel_resolution=pixel_resolution,
                        planet=self.planet)
                if coordinates == None:
                    latitude, longitude = (x_origin, y_origin)
                    label_wind = False
                
                filename = f"{prediction_folder_path}/tiles/{x_origin}_{y_origin}.jpeg"
                cv2.imwrite(filename,im)
               
                prediction = {"filename":[filename],
                            "x_pixels":[x_origin],
                            "y_pixels":[y_origin],
                            "latitude":[latitude],
                            "longitude":[longitude],
                            "is_dune_label":[0],   
                            "strength_label":[0],
                            "angle_label":[0]}
            
                results = pd.concat([results, pd.DataFrame.from_dict(prediction)])
        if label_wind:
            wind_data = DataPreparation().fetch_wind_only(labels_df=results)
            wind_data.to_csv(f'{prediction_folder_path}/label_data_wind.csv', index=False)
        try:
            wind_data = pd.read_csv(f'{prediction_folder_path}/label_data_wind.csv')   
            for index,row in results.iterrows():
                lat_diff = abs(wind_data.latitude-row['latitude'])
                long_diff = abs(wind_data.longitude-row['longitude'])
                total_diff = lat_diff+long_diff
                idx = total_diff.argmin()
                results.loc[index,'strength_label'] = wind_data.loc[idx,'wind_strength']
                angular_sin = wind_data.loc[idx,'sin']
                angular_cos = wind_data.loc[idx,'cos']
                results.loc[index,'angle_label'] = sin_cos_to_degrees(angular_sin, angular_cos)
            results.to_csv(f'{prediction_folder_path}/label_data.csv', index=False)
        except:
            pass
        
        return self
    
        
    def get_predictions(self, image_path, 
                        pix_dim=10,
                        coordinates=None):
        prediction_folder_path = self.create_prediction_folder(image_path)
        resized_image,resized_original_image = self.open_and_scale_image(image_path,pix_dim)
        tiled_data = self.prepare_tiles(resized_image,resized_original_image)
        
        x = tiled_data.get('nb horizontal tiles')
        y= tiled_data.get('nb vertical tiles')
        tiles = tiled_data.get('tiles')
        full_image = tiled_data.get('full image')
                
        predictions = {"filename":[],
                       "x_pixels":[],
                "y_pixels":[],
                "latitude":[],
                "longitude":[],
                "is_dune":[],   
                "probability":[],
                "strength":[],
                "angle":[]}
            
        results = pd.DataFrame.from_dict(predictions)
        
        for col in range(y):
            for row in range(x):
                im = tiles[col*x+row]
                predicted_values = self.prediction_from_models(im)
            
                x_origin=row*256+128
                y_origin=col*256+128
                dune_proba=predicted_values[0] 
                wind_strength=predicted_values[1]
                angle=predicted_values[2]
                
                if angle <0:
                    angle = 360+angle
                
                is_dune = False
                if dune_proba>=self.probability_threshold:
                    is_dune = True
                
                if coordinates == 'label':
                    latitude, longitude = self.get_coordinates_from_filename(self.image_name)
                else:
                    if coordinates is not None:
                        latitude, longitude = self.get_coordinates(
                        base_coordinates=coordinates,
                        tile_coordinates = (x_origin,y_origin),
                        image_size = full_image.shape,
                        pixel_resolution=pix_dim,
                        planet=self.planet)
                    
                    if coordinates == None:
                        latitude, longitude = (np.nan, np.nan)
                    
                prediction = {"filename":[f"{self.prediction_folder}/tiles/{x_origin}_{y_origin}.jpeg"],
                              "x_pixels":[x_origin],
                    "y_pixels":[y_origin],
                    "latitude":[latitude],
                    "longitude":[longitude],
                    "is_dune":[is_dune],   
                    "probability":[dune_proba],
                    "strength":[wind_strength],
                    "angle":[angle]}
            
                results = pd.concat([results, pd.DataFrame.from_dict(prediction)])
                
        imager = PredictionImage(predictions=results,
                                  image = tiled_data.get('trimmed image'),
                                  prediction_path = self.prediction_path)
        prediction_image = imager.add_predictions()
        self.image_objects.append(imager)
        
        return (results, prediction_folder_path, prediction_image)
    
    def predict_from_file(self, path, pixel_resolution=10, coordinates=None):
        results, prediction_folder_path,_ =  self.get_predictions(path, pixel_resolution, coordinates)
        results.to_csv(f'{prediction_folder_path}/predictions.csv', index=False)
        
        return results
    
    def predict_from_folders(self, foldernames, pixel_resolution=10):
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
                    _,_ =  self.get_predictions(path, pixel_resolution)
                except:
                    pass
            prediction_folder_path = self.prediction_path.split("/")[:-1] 
            prediction_folder_path = "/".join(prediction_folder_path)
            self.predictions.to_csv(f'{prediction_folder_path}/predictions.csv', index=False)
            
            predictions = self.predictions
            self.initialize_results()
            
        return predictions
   
    
if __name__ == '__main__':
    predictor = Predictor(probability_threshold=0.5)
    #image = '../raw_data/mars_images/Murray-Lab_CTX-Mosaic_beta01_E-038_N-36.jpg'
    #image = '../raw_data/images/testing/dunes/21.8424991607666_50.2474983215332_031_CW000_-0.6031867265701294_-0.6447361707687378_1.9967776536941528.jpg'
    #image = '../raw_data/mars_images/demo.jpg'
    #image = '../raw_data/mars_images/alt/Murray-Lab_CTX-Mosaic_beta01_E-178_N-04.jpg'
    #image = '../raw_data/mars_images/control/Murray-Lab_CTX-Mosaic_beta01_E-038_N-36.jpg'
    #image = '../raw_data/moon_images/SosigenesCrater.png'
    image = '../raw_data/earth_images/landsat-106.473427_32.923187_-106.27342700000001_33.123187.png'
    #image = '../raw_data/mars_images/alt/Mars_small_new.jpg'
    #path,df = predictor.get_prediction_image(image,10)
    #predictor.predict_from_file('../raw_data/earth_images/landsat-106.473427_32.923187_-106.27342700000001_33.123187.png')
    predictor.save_tiled_data(image, coordinates=(32.7982115,-106.3750148),label_wind=False)
    #predictor.predict_from_folders('../raw_data/images/testing/earth_test_small')