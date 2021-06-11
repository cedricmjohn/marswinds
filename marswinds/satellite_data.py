import numpy as np
import pandas as pd
import os
import ee
import geemap
import cv2
import xarray as xr
import datetime

class SatelliteData:
    
    def __init__(self, data,logfile):
        self.data = data
        self.logfile = logfile
            
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
    
        area = [(long-0.025,lat-0.025),
            (long+0.025,lat+0.025)] 
        print(f'Fetching {lat}/{long}')     
        roi = ee.Geometry.Rectangle(coords=area)  
        vis_params = {'bands': [ 'B4','B3','B2'],
                  'min': 0,
                  'max': 10000,
                  'gamma': 1.4}
           
        collection = ee.ImageCollection('COPERNICUS/S2_SR') \
                .filterBounds(roi) \
                .filter('HIGH_PROBA_CLOUDS_PERCENTAGE < 1') \
                .filter('NODATA_PIXEL_PERCENTAGE < 5') \
                .filter('DEGRADED_MSI_DATA_PERCENTAGE == 0') \
                .filter('SATURATED_DEFECTIVE_PIXEL_PERCENTAGE == 0') \
                .filter('DARK_FEATURES_PERCENTAGE < 5') \
                .limit(10)

        image = collection.first()
        image_name = f"{lat}_{long}_0{quadrant}_CW000_{self.data.sin}_{self.data.cos}_{self.data.wind_strength}"
        folder_name = f"../raw_data/images/{self.data.folder}/{self.data.image_type}"
        file_name = f"{folder_name}/{image_name}.jpg"
        
        out_img = os.path.expanduser(file_name)
        tmp_img = os.path.expanduser('../raw_data/tmp/tmp.jpg')
        
        
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        if not os.path.exists('../raw_data/tmp'):
            os.makedirs('../raw_data/tmp')
    
        
        sentinel_available = False
        landsat_available = False
        sentinel_complete = False
        landsat_complete = False

        try:
            geemap.get_image_thumbnail(image, tmp_img,vis_params,dimensions=(resolution, resolution),region=roi, format='jpg')
            sentinel_available=True
        except:
            print("Cannot fetch image from Sentinel, attempting to fetch from Landsat")
            collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR') \
                .filterBounds(roi) \
                .filter('CLOUD_COVER < 1') \
                .filter('CLOUD_COVER_LAND < 1') \
                .limit(10)
            image = collection.first()
            try:
                geemap.get_image_thumbnail(image, tmp_img,vis_params,dimensions=(resolution, resolution),region=roi, format='jpg')
                landsat_available = True
            except:
                print("Cannot fetch image from either satellites")
        
        
        if(sentinel_available or landsat_available):
            im = cv2.imread(tmp_img)
            tot_pix = im.shape[0]*im.shape[1]*im.shape[2]
            missing_pix = im[np.where(im == 0)].shape[0]
            sentinel_complete=True
            if (missing_pix/tot_pix > 0.05):
                print("Too much information missing from picture, trying to fetch from Landsat 8")
                sentinel_complete=False
                collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR') \
                    .filterBounds(roi) \
                    .filter('CLOUD_COVER < 1') \
                    .filter('CLOUD_COVER_LAND < 1') \
                    .limit(10)
                image = collection.first()
                try:
                    geemap.get_image_thumbnail(image, tmp_img,vis_params,dimensions=(resolution, resolution),region=roi, format='jpg')
                    im = cv2.imread(tmp_img)
                    landsat_available = True
                    landsat_complete= True
                    tot_pix = im.shape[0]*im.shape[1]*im.shape[2]
                    missing_pix = im[np.where(im == 0)].shape[0]
                    if missing_pix/tot_pix > 0.05:
                        landsat_complete=False
                except:
                    print("Cannot fetch image from Landstat 8")

        if (sentinel_complete| landsat_complete ):
            im = cv2.imread(tmp_img)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(out_img,gray)
        else:
            file_name='none'
        
        log = pd.read_csv(self.logfile)
        
        if sentinel_complete:
            sat = 'Sentinel-2'
            pxres = '10m'
        else:
            sat = 'Landsat-8'
            pxres = '30m'
            
        d = {"time":[datetime.datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")],
            "image_latitude":[lat],
            "image_longitude":[long],
            "satellite":[sat], 
            "image_type":[self.data.image_type],
            "pixel_resolution":[pxres],
            "file_name":[file_name],
            "SENTINEL_available":[sentinel_available],
            "SENTINEL_complete":[sentinel_complete],
            "LANDSAT_available":[landsat_available],
            "LANDSAT_complete":[landsat_complete]
        }
        added_log = pd.concat([pd.DataFrame.from_dict(d),
                               log])
        added_log.to_csv(self.logfile,index=False)       
        
        return self
    
    

    def prep_data(self, path):
        ds = xr.open_dataset(path)
        raw_data = ds.to_dataframe()
        clean_df = raw_data.dropna().reset_index()
        clean_df['wind_strength'] = (clean_df.u10**2 + clean_df.v10**2)**.5
        clean_df['sin'] = np.sin(np.arctan2(clean_df['u10'],clean_df['v10']))
        clean_df['cosin'] = np.cos(np.arctan2(clean_df['u10'],clean_df['v10']))
        wind_data = clean_df.groupby(by=['latitude','longitude']).mean().reset_index()
        return wind_data

