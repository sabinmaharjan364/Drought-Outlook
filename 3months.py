import os
import numpy as np
import xarray as xr
import pandas as pd
from functools import partial
from joblib import Parallel, delayed
import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool
import matplotlib.pyplot as plt
from IPython.display import display
import netCDF4 as nc
import rasterio
from shapely.geometry import Point
import geopandas as gpd
from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio.transform import from_bounds

ncname_cdi = "/Users/sabinmaharjan/projects/python/do/static/file/cdi_1.nc"
ncname_rain = "/Users/sabinmaharjan/projects/python/do/static/file/3_average_final.nc"

ds1 = xr.open_dataset(ncname_cdi)
lon = ds1["longitude"].values
lat = ds1["latitude"].values
dname1 = "cdi"
cdi_array = ds1[dname1].values
cdi_slice = cdi_array[:, :, 305]
cdi_vec = np.ravel(cdi_slice)
lon_grid, lat_grid = np.meshgrid(lon, lat)
cdi_df = pd.DataFrame({'lon': lon_grid.flatten(), 'lat': lat_grid.flatten(), 'cdi': cdi_vec})
cdi_df_crop=cdi_df.dropna()

print(cdi_slice.shape)
print(cdi_df.shape)
print(cdi_df_crop.shape)
print(cdi_df_crop.head(10))

ds2 = xr.open_dataset(ncname_rain)
lon_r = ds2["longitude"].values
lat_r = ds2["latitude"].values
rain_avg=ds2["rain_avg"].values
time_r=ds2["time"].values
rain_vec = np.ravel(rain_avg)
lon_grid, lat_grid = np.meshgrid(lon_r, lat_r)
rain_df = pd.DataFrame({'lat': lat_grid.flatten(), 'lon': lon_grid.flatten(), 'rain': rain_vec})
time_r = time_r.astype('datetime64[ns]')  # Convert to datetime64
month_name = pd.to_datetime(time_r[0]).strftime('%B')
print(rain_df.shape)
print("First month:", month_name)

join_df = cdi_df.merge(rain_df, how='left', on=['lon', 'lat'])
print(join_df.shape)
print("join_df.head(10)")
display(join_df)
num_rows_with_nan = join_df.isna().sum().sum()
num_rows_without_nan = join_df.notna().sum().sum()
print(f"Number of rows with NaN values: {num_rows_with_nan}")
print(f"Number of rows without NaN values: {num_rows_without_nan}")
rmna_df=join_df.dropna()
print(rmna_df.shape)
nan_count = rmna_df.isna().sum().sum()
nan__cdi_count = rmna_df.notna().sum().sum()
nan__cdi_count = rmna_df.notna().sum()
print("Number of  NaN values:", nan_count)
print("Number of NaN values:", nan__cdi_count)

def classify_drought(row):
    cdi, rain = row['cdi'], row['rain']
    if cdi < 0.2:
        if rain < 50:
            if cdi < 0.02:
                return 5
            else:
                return 6
        elif rain < 70:
            return 5
        else:
            if 0.1 <= cdi < 0.2:
                return 2
            else:
                return 3
    else:
        if rain < 30:
            return 4
        else:
            return 1

ncell = len(rmna_df)
ncores = min(multiprocessing.cpu_count(), 4)
with Pool(ncores) as p:
    try:
        classified = p.map(classify_drought, [rmna_df.iloc[i] for i in range(rmna_df.shape[0])])
    except Exception as e:
        print(f"An error occurred during multiprocessing: {e}")
        p.close()
        p.join()
        raise
print(len(classified))

df_out = pd.DataFrame({ 'lat': cdi_df['lat'], 'lon': cdi_df['lon'],'outlook': np.nan})
print(len(rmna_df.index))
order = rmna_df.index.astype(int)
classified = np.array(classified)
df_out.loc[order, 'outlook'] = classified.astype(int)
print(df_out['outlook'].value_counts())

print(cdi_slice.shape)
print(len(lon))
print(len(lat))

da = xr.DataArray(df_out['outlook'].values.reshape(cdi_slice.shape),
                  coords=[('lat', lat), ('lon', lon)],
                  name='outlook')
da.attrs['varunit'] = ''
da.attrs['longname'] = 'drought outlook'
ds = da.to_dataset()
ds['time'] = (('time'), time_r)  # Add time variable to the dataset

month_name = "April 3 months forecast"
out_ncname = "/Users/sabinmaharjan/projects/python/do/static/result/nc/3_months/"+month_name+"_Final_2024.nc"
try:
    ds.to_netcdf(out_ncname)
    print(f"file saved with name: {out_ncname}")
except Exception as e:
    print(f"An error occurred while saving the Dataset: {e}")