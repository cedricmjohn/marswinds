import numpy as np
import pandas as pd
import os
import ee
import geemap
import cv2
import xarray as xr

def get_region(data:pd.DataFrame) -> pd.DataFrame:
    pass

def get_image_per_coordinates(data:pd.Series, **kwargs) -> list:
    
        coordinates = [(data.longitude-0.025,data.latitude-0.025),
              (data.longitude-0.025,data.latitude+0.025),
              (data.longitude+0.025,data.latitude+0.025),
              (data.longitude+0.025,data.latitude-0.025)]
        
        for quadrant,coordinate in enumerate(coordinates):
            get_one_image(long=coordinate[0], lat=coordinate[1], data=data, quadrant=quadrant)
        
        return None
    
def get_one_image(self,long:float, lat:float,data:pd.Series, quadrant:int, **kwargs) -> list:
    
    ee.Initialize()
    
    if 'resolution' in kwargs.keys():
        resolution=int(kwargs['resolution'])
    else:
        resolution = 512
    
    satellite_name='COPERNICUS/S2_SR'
    
    area = [(long-0.025,lat-0.025),
            (long+0.025,lat+0.025)]
        
    roi = ee.Geometry.Rectangle(coords=area)
           
    collection = ee.ImageCollection(satellite_name) \
                .filterBounds(roi) \
                .sort("CLOUD_COVER") \
                .filter('HIGH_PROBA_CLOUDS_PERCENTAGE < 10') \
                .limit(1)

    image = collection.first()
    
    vis_params = {
                  'bands': [ 'B4','B3','B2'],
                  'min': 0,
                  'max': 10000,
                  'gamma': 1.4}

    file_name = f"{lat}_{long}_0{quadrant}_CW000_{data.sin}_{data.cosin}_{data.wind_strength}"
    out_img = os.path.expanduser(f"../raw_data/practice_dataset/{file_name}.jpg")

    geemap.get_image_thumbnail(image, out_img,vis_params,dimensions=(resolution, resolution),region=roi, format='jpg')
        
    image_grey = cv2.imread(out_img)
    image_grey = cv2.cvtColor(image_grey, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"../raw_data/practice_dataset/{file_name}.jpg",image_grey)
        
    return None

def prep_data(path):
    ds = xr.open_dataset(path)
    raw_data = ds.to_dataframe()
    clean_df = raw_data.dropna().reset_index()
    clean_df['wind_strength'] = (clean_df.u10**2 + clean_df.v10**2)**.5
    clean_df['sin'] = np.sin(np.arctan2(clean_df['u10'],clean_df['v10']))
    clean_df['cosin'] = np.cos(np.arctan2(clean_df['u10'],clean_df['v10']))
    wind_data = clean_df.groupby(by=['latitude','longitude']).mean().reset_index()
    return wind_data

if __name__ == '__main__':
    file_paths = ['../raw_data/practice_wind_dataset.nc',
                  '../raw_data/white_sands_winds.nc',
                  '../raw_data/practice_wind_dataset.nc',
                  '../raw_data/gobi_winds.nc']
    
    for path in file_paths:
        test_data = prep_data(path).iloc[1,:]
        get_image_per_coordinates(test_data)