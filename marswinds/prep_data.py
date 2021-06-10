import pandas as pd
import xarray as xr
import numpy as np
    
    
def prep_data(path):
        
    ds = xr.open_dataset(path)
    raw_data = ds.to_dataframe()
    clean_df = raw_data.dropna().reset_index()
    clean_df["wind strength"] = (clean_df.u10**2 + clean_df.v10**2)**.5
    clean_df['sin'] = np.sin(np.arctan2(clean_df['u10'],clean_df['v10']))
    clean_df['cos'] = np.cos(np.arctan2(clean_df['u10'],clean_df['v10']))
    wind_data = clean_df.groupby(by=['latitude','longitude']).mean().reset_index()
    return wind_data