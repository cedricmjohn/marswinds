import numpy as np
import pandas as pd
import os
import ee
import geemap
import cv2
import xarray as xr

class SatelliteData:
    
    def __init__(self, data):
        self.data = data 
            
    def get_image_per_coordinates(self, **kwargs) -> list:
    
            coordinates = [
              (self.data.longitude+0.025,self.data.latitude+0.025,1),
              (self.data.longitude-0.025,self.data.latitude+0.025,2),
              (self.data.longitude-0.025,self.data.latitude-0.025,3),
              (self.data.longitude+0.025,self.data.latitude-0.025,4)]
        
            for coordinate in coordinates:
                self.get_one_image(long=coordinate[0], lat=coordinate[1], data=self.data, quadrant=coordinate[2])
        
            return None
    
    def get_one_image(self,long:float, lat:float,quadrant:int, **kwargs) -> list:
    
        ee.Initialize()
    
        if 'resolution' in kwargs.keys():
            resolution=int(kwargs['resolution'])
        else:
            resolution = 512
    
        satellite_name='COPERNICUS/S2_SR'
    
        area = [(long-0.025,lat-0.025),
                (long+0.025,lat+0.025)]
        print(f'Fetching {lat}/{long}')
        roi = ee.Geometry.Rectangle(coords=area)
           
        collection = ee.ImageCollection(satellite_name) \
                .filterBounds(roi) \
                .filter('HIGH_PROBA_CLOUDS_PERCENTAGE < 1') \
                .filter('NODATA_PIXEL_PERCENTAGE < 1') \
                .limit(1)

        image = collection.first()
    
        vis_params = {
                  'bands': [ 'B4','B3','B2'],
                  'min': 0,
                  'max': 10000,
                  'gamma': 1.4}

        image_name = f"{lat}_{long}_0{quadrant}_CW000_{self.data.sin}_{self.data.cos}_{self.data.wind_strength}"
        file_name = f"../raw_data/images/{self.data.folder}/{self.data.image_type}/{image_name}.jpg"
        out_img = os.path.expanduser(file_name)

        try:
            geemap.get_image_thumbnail(image, out_img,vis_params,dimensions=(resolution, resolution),region=roi, format='jpg')
            image_grey = cv2.cvtColor(cv2.imread(out_img), 
                              cv2.COLOR_BGR2GRAY)
            cv2.imwrite(file_name,image_grey)
        except:
            print('No image corresponding to filter criteria')
        
        
        return None

    def prep_data(self, path):
        ds = xr.open_dataset(path)
        raw_data = ds.to_dataframe()
        clean_df = raw_data.dropna().reset_index()
        clean_df['wind_strength'] = (clean_df.u10**2 + clean_df.v10**2)**.5
        clean_df['sin'] = np.sin(np.arctan2(clean_df['u10'],clean_df['v10']))
        clean_df['cosin'] = np.cos(np.arctan2(clean_df['u10'],clean_df['v10']))
        wind_data = clean_df.groupby(by=['latitude','longitude']).mean().reset_index()
        return wind_data

