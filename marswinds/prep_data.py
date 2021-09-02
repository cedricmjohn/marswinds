import pandas as pd
import xarray as xr
import numpy as np
import math
import glob, os
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import cv2
from marswinds.satellite_data import SatelliteData
from marswinds.wind_data import WindData
from marswinds.utils import decode_angle, encode_angle
import datetime
    

class DataPreparation:
    
    def __init__(self,nb_lines=None, image_type='dunes', 
                 force_download=False,
                 resume_log=None):
        self.nb_lines=nb_lines
        self.image_type=image_type 
        current_time = datetime.datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
        self.force_download=force_download
        
        if resume_log != None:
            self.logfile = f'../raw_data/logs/{resume_log}.csv'
        else:
            if not os.path.exists('../raw_data/logs'):
                os.makedirs('../raw_data/logs')
            self.logfile = f'../raw_data/logs/{current_time}.csv'
            
        
            d = {"time":[],
                "image_latitude":[],
                "image_longitude":[],   
                "satellite":[],
                "label":[],
                "pixel_resolution":[],
                "file_name":[],
                "SENTINEL_available":[],
                "SENTINEL_complete":[],
                "LANDSAT_available":[],
                "LANDSAT_complete":[],
            }
            
            pd.DataFrame.from_dict(d).to_csv(self.logfile,index=False)
         
    def get_last_region(self, lat, long):
        regions = pd.read_csv(f'../raw_data/lists/{self.image_type}.csv')
        last_index = 0
        
        for idx in  regions.index.values:
            region = regions.iloc[idx,:]
            if (((region.north > lat) and (region.south < lat)) and ((region.west < long) and (region.east > long))):
                last_index=idx

        return regions.loc[last_index::,:]
    
    def get_last_cell(self, lat, long, wind_data):
        last_index = 0
        
        for idx in  wind_data.index.values:
            current_lat = wind_data.iloc[idx,:]['latitude']
            current_long = wind_data.iloc[idx,:]['longitude']
            if (((current_lat-0.05 < lat) and (current_lat+0.05 > lat))): 
                if ((current_long-0.05 < long) and (current_long+0.05 > long)):
                    last_index=idx
                    
        return wind_data.loc[last_index::,:]
                
    
    def fetch_all_data(self):
        df = pd.read_csv(f'../raw_data/lists/{self.image_type}.csv')
        
        log =pd.read_csv(self.logfile)
        
        if log.shape[0] > 0:
            lat = log.iloc[0,:].image_latitude
            long = log.iloc[0,:].image_longitude
            df = self.get_last_region(lat,long).reset_index()
            set_continue_download_flag = True
        else:
            set_continue_download_flag = False
        
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
            if set_continue_download_flag:
                last_lat = log.iloc[0,:].image_latitude
                last_long = log.iloc[0,:].image_longitude
                wind_data = self.get_last_cell(last_lat, last_long,wind_data).reset_index()
                start_lat = wind_data.iloc[0,:]['latitude']
                start_long = wind_data.iloc[0,:]['longitude']
                print(f'Starting to download from coordinates {start_lat}/{start_long}')
                set_continue_download_flag = False
                self.force_download=False
            
            for data_index in wind_data.index.values:
                data = wind_data.iloc[data_index,:]
                SatelliteData(data,self.logfile, force_download=self.force_download).get_image_per_coordinates()
            
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
            print(item.region)
            print(item.north, item.west,item.south, item.east)
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
        
        for image_name in tqdm(list_of_images):
            if image_name.split('.')[-1]=='jpg':
                self.apply_rotation(f'{mypath}/{image_name}')
        return self
            
    def apply_rotation(self, image_name):
        image = cv2.imread(image_name)
        rotation_values = [(cv2.cv2.ROTATE_90_CLOCKWISE,90,'CW090'),
                           (cv2.cv2.ROTATE_180,180,'CW180'),
                           (cv2.cv2.ROTATE_90_COUNTERCLOCKWISE,270,'CW270')]
        
        original_angle = decode_angle(image_name)
        label = image_name.split('/')[-1]
        
        for rotation in rotation_values:
            rota = rotation[1]
            new_angle = original_angle + rota
            new_sin, new_cos = encode_angle(new_angle)
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
    
    def flip_images(self):
        base_path = '../raw_data/images/training'
        
        mypath=f'{base_path}/{self.image_type}'
        try:
            list_of_images =  [f for f in listdir(mypath) if isfile(join(mypath, f))]
            print(f'Now flipping {len(list_of_images)} images')
        except:
            print(f'No images flipped (check if the {mypath} is empty)')
            return self
        
        for image_name in tqdm(list_of_images):
            if image_name.split('.')[-1]=='jpg':
                self.apply_flip(f'{mypath}/{image_name}')
        return self
    
    def apply_flip(self, image_name):
        image = cv2.imread(image_name)
        
        original_angle = decode_angle(image_name)
        label = image_name.split('/')[-1]

        new_angle = 360-original_angle
        new_sin, new_cos = encode_angle(new_angle)
        name_tags = label.split('_')
        name_tags[-5] = f'{name_tags[-5]}F'
        name_tags[-3] = str(new_sin)
        name_tags[-2] = str(new_cos)
        image_label = "_".join(name_tags)
        path_tags = image_name.split('/')
        path_tags[-1] = image_label
        image_path = '/'.join(path_tags)
        new_image = cv2.flip(image, 1)
        cv2.imwrite(image_path,new_image)
        
        return self
    
            
        
    
    
if __name__ == '__main__':
    data_handler = DataPreparation(image_type='earth_tests', # replace by no_dunes for rocks
                                   force_download=False) # replace by True if you want to delete image previously downloaded
    data_handler.fetch_all_data()
    #data_handler.flip_images()
    #data_handler.rotate_images()