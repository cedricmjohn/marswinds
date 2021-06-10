import pandas as pd
import xarray as xr
import cdsapi
import os
import numpy as np


class Wind:
    def __init__(self, outpath, file_,NWSE_coordinates):  #file_ to be given as a .nc file 
        self.outpath = outpath
        self.file_ = file_
        self.NWSE_coordinates = NWSE_coordinates
            
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
    
    def prep_data(self):
        
        ds = xr.open_dataset(self.outpath + self.file_)
        raw_data = ds.to_dataframe()
        clean_df = raw_data.dropna().reset_index()
        clean_df["wind strength"] = (clean_df.u10**2 + clean_df.v10**2)**.5
        clean_df['sin'] = np.sin(np.arctan2(clean_df['u10'],clean_df['v10']))
        clean_df['cos'] = np.cos(np.arctan2(clean_df['u10'],clean_df['v10']))
        wind_data = clean_df.groupby(by=['latitude','longitude']).mean().reset_index()
        return wind_data
        


### THIS WAS THE NOTEBOOK CODE TO LOOP OVER AND DOWNLOAD AND PREP THE DATA AUTOMATICALLY

# final_df = pd.DataFrame()        
# for i in data_list:
#     NWSE_coordinates = [i[1], i[2], i[3], i[4]]
#     wind = Wind(outpath = 'Downloads/', file_ = f'{i[0]}.nc', NWSE_coordinates = NWSE_coordinates)
#     wind.get_data_one_region()
#     clean_df = wind.prep_data()
    
#     final_df = pd.concat([final_df, clean_df], axis = 0)
#     print("done",i[0])

wind = Wind(outpath = 'marswinds/data/wind_data/', file_ = 'test4.nc', NWSE_coordinates = [90, -10, 80, 10,])
  
print(wind.get_data_one_region())

