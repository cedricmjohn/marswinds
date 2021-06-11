import pandas as pd
import xarray as xr
import cdsapi
import os
import numpy as np
from sklearn.model_selection import train_test_split


class WindData:
    def __init__(self, file_,NWSE_coordinates=None, 
                 outpath='../raw_data/wind/', 
                 image_type='dunes'):  #file_ to be given as a .nc file 
        self.outpath = outpath
        self.file_ = f'{file_}.nc'
        self.NWSE_coordinates = NWSE_coordinates
        self.image_type=image_type
            
    def get_data_one_region(self): #NWSE passed as a list and outpath is the directory in which it will be saved

        
        if os.path.isdir(self.outpath) == False:
                os.mkdir(self.outpath)


        c = cdsapi.Client()
        
        c.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'format': 'netcdf',
            'product_type': 'monthly_averaged_reanalysis',
            'variable': [
                '10m_u_component_of_wind', '10m_v_component_of_wind',
            ],
            'year': ['2011',
                '2012', '2013', '2014',
                '2015', '2016', '2017',
                '2018', '2019', '2020',
                '2021',
            ],
            'month': [
                '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            ],
            'time': '00:00',
            'area': self.NWSE_coordinates,
        },
        self.outpath + self.file_)
        
        return self
    
    def prep_data(self):
        
        ds = xr.open_dataset(self.outpath + self.file_)
        raw_data = ds.to_dataframe()
        clean_df = raw_data.dropna().reset_index()
        clean_df["wind_strength"] = (clean_df.u10**2 + clean_df.v10**2)**.5
        clean_df['sin'] = np.sin(np.arctan2(clean_df['u10'],clean_df['v10']))
        clean_df['cos'] = np.cos(np.arctan2(clean_df['u10'],clean_df['v10']))
        wind_data = clean_df.groupby(by=['latitude','longitude']).mean().reset_index()
        wind_data = self.train_test_geographic_split(wind_data)
        
        return wind_data
    
    def train_test_geographic_split(self, data:pd.DataFrame) -> pd.DataFrame:
        wind_data = data.copy()
        nb_long = wind_data.longitude.unique().shape[0]
        nb_lat = wind_data.latitude.unique().shape[0]
        nb_test_long = 0
        nb_test_lat = 0
        
        while nb_test_long * nb_test_lat < wind_data.shape[0]*.3:
            nb_test_long += 1
            if (nb_test_long * nb_test_lat < wind_data.shape[0]*.3):
                nb_test_lat += 1
        

        test_lat = np.sort(wind_data.latitude.unique())[:nb_test_lat]
        test_long = np.sort(wind_data.longitude.unique())[:nb_test_long]
        wind_data.loc[:,'image_type']=self.image_type
        wind_data.loc[:,'folder']='training'
        
        
        train_data = wind_data.copy()
        test_val_data = pd.DataFrame()

        for vlat in test_lat:
            for vlong in test_long:
                filter_ = ((train_data.latitude == vlat) & (train_data.longitude==vlong))
                test_val_data=pd.concat([test_val_data, train_data[filter_]])
                train_data =train_data[~filter_].copy()
        
        test_val_data.loc[:,'folder']='testing' 
        
        if (nb_test_long*nb_test_lat<3):
            print(f'Grid is too small to split: {nb_lat} x {nb_long}. Returning training and testing data only.')
            return pd.concat([train_data,test_val_data])
          
        test_data, val_data = train_test_split(test_val_data, test_size=0.3)   
        test_data.loc[:,'folder']='testing'
        val_data.loc[:,'folder']='validation'

        return pd.concat([train_data,test_data,val_data])
        


