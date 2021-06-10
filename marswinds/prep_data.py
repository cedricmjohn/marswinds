import pandas as pd
import xarray as xr
import numpy as np
import math
import glob, os
from os import listdir
from os.path import isfile, join
import cv2
from marswinds.satellite_data import SatelliteData
from marswinds.wind_data import WindData
    

class DataPreparation:
    def __init__(self,nb_lines=None, image_type='dunes', 
                 clean_download=False,
                 auto_rotate=True):
        self.nb_lines=nb_lines
        self.image_type=image_type
        self.auto_rotate = auto_rotate
        
        if clean_download:
            os.rmdir(f'../raw_data/images/{self.image_type}')
            
        
    
    def fetch_all_data(self):
        df = pd.read_csv(f'../raw_data/lists/{self.image_type}.csv')
        
        if self.nb_lines:
            coordinates = df.head(self.nb_lines).copy()
        else:
            coordinates = df.copy()
        
        
        for index in coordinates.index.values:
            item = coordinates.iloc[index,:]
            print('Now fetching:')
            print(item)
            wind_data = WindData(file_=item.name,
                                 NWSE_coordinates = [item.north,
                                  item.west,
                                  item.south,
                                  item.east],
                                 image_type=self.image_type) \
                        .get_data_one_region() \
                        .prep_data()
            
            for data_index in wind_data.index.values:
                data = wind_data.iloc[data_index,:]
                SatelliteData(data).get_image_per_coordinates()
        
        if self.auto_rotate:
            self.rotate_images()
            
        return self                 
    
    def fetch_wind_only(self):
        df = pd.read_csv(f'../raw_data/lists/{self.image_type}.csv')
        
        if self.nb_lines:
            coordinates = df.head(self.nb_lines).copy()
        else:
            coordinates = df.copy()
        
        
        for index in coordinates.index.values:
            item = coordinates.iloc[index,:]
            print(f'Now fetching {item.region}')
            print(item)
            wind_data = WindData(file_=item.region,
                                 NWSE_coordinates = [item.north,
                                  item.west,
                                  item.south,
                                  item.east],
                                 image_type=self.image_type) \
                        .get_data_one_region() \
                        .prep_data()
        return self
    
    def fetch_images_only(self, file_name=None):
        wind_data = WindData(file_name).prep_data()
        for data_index in wind_data.index.values:
                data = wind_data.iloc[data_index,:]
                SatelliteData(data).get_image_per_coordinates()
        return self
    
    def rotate_images(self):
        base_path = '../raw_data/images/training'
        
        mypath=f'{base_path}/{self.image_type}'
        try:
            list_of_images =  [f for f in listdir(mypath) if isfile(join(mypath, f))]
            print(f'Now applying rotation to {len(list_of_images)} files')
        except:
            print(f'No rotation applied (check if the {mypath} is empty)')
            return self
        
        for image_name in list_of_images:
            if image_name.split('.')[-1]=='jpg':
                self.apply_rotation(f'{mypath}/{image_name}')
        return self
            
    def apply_rotation(self, image_name):
        image = cv2.imread(image_name)
        rotation_values = [(cv2.cv2.ROTATE_90_CLOCKWISE,90,'CW090'),
                           (cv2.cv2.ROTATE_180,180,'CW180'),
                           (cv2.cv2.ROTATE_90_COUNTERCLOCKWISE,270,'CW270')]
        
        label = image_name.split('/')[-1]
        original_sin = float(label.split('_')[-3])
        original_cos = float(label.split('_')[-2])
        original_strength = label.split('_')[-1]
        original_angle = math.atan2(original_sin, original_cos)
        original_angle *= 180 / math.pi
        if original_angle < 0: original_angle += 360
        
        for rotation in rotation_values:
            rota = rotation[1]
            new_angle = original_angle + rota
            if new_angle > 360: original_angle -= 360
            new_sin = np.sin(new_angle)
            new_cos = np.cos(new_angle)
            name_tags = label.split('_')
            name_tags[-4] = rotation[2]
            name_tags[-3] = str(new_sin)
            name_tags[-2] = str(new_cos)
            image_label = "_".join(name_tags)
            path_tags = image_name.split('/')
            path_tags[-1] = image_label
            image_path = '/'.join(path_tags)
            new_image = cv2.rotate(image, rotation[0])
            cv2.imwrite(image_path,new_image)
        return self
            
        
    
    
if __name__ == '__main__':
    data_handler = DataPreparation(image_type='dunes') #replace by 'no_dunes' to fetch rocks
    data_handler.fetch_all_data()
    #data_handler.rotate_images()
    