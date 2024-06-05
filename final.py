# %%
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
# Load the ncdf4 package
import netCDF4 as nc
import rasterio
from shapely.geometry import Point
import geopandas as gpd
from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio.transform import from_bounds

# %% [markdown]
# ## File path and settings

# %%
# Set the file paths
ncname_cdi = "/Users/sabinmaharjan/projects/python/do/static/file/cdi_1.nc"
ncname_rain = "/Users/sabinmaharjan/projects/python/do/static/file/2024.forecast.nc"
ncname_rain_final="/Users/sabinmaharjan/projects/python/do/static/file/2.nc"
# pd.set_option('display.max_rows',None)
# pd.set_option('display.max_columns',None)

# pd.set_option('display.precision',2)
# pd.set_option('display.float_format','{:.2f}'.format)


# %%
try:
   
    ds = xr.open_dataset(ncname_rain)

    # Rename 'lat' to 'latitude' and 'lon' to 'longitude'
    ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})

    # Assuming latitude and longitude variables are now named 'latitude' and 'longitude'
    # Round latitude and longitude values
    ds['latitude'] = ds['latitude'].astype('double')
    ds['longitude'] = ds['longitude'].astype('double')

    # Round latitude values to two decimal places
    ds['latitude'] = ds['latitude'].round(decimals=2)
    ds['longitude'] = ds['longitude'].round(decimals=2)

    # Save the modified dataset to a new NetCDF file
    ds.to_netcdf(ncname_rain_final)

    print("New file saved as:", ncname_rain_final)

except Exception as e:
    print("An error occurred:", e)

# %% [markdown]
# ## Reading CDI data and creating dataframe

# %%

# Open the first NetCDF file

ds1 = xr.open_dataset(ncname_cdi)
 
# Get longitude and latitude
lon = ds1["longitude"].values
lat = ds1["latitude"].values
 
# Get cdi
dname1 = "cdi"
cdi_array = ds1[dname1].values
 
# Get a single slice or layer
cdi_slice = cdi_array[:, :, 305]  # Assuming 305 is the index of the slice you want


cdi_vec = np.ravel(cdi_slice)
 
# Create DataFrame for cdi data
lon_grid, lat_grid = np.meshgrid(lon, lat)
cdi_df = pd.DataFrame({'lon': lon_grid.flatten(), 'lat': lat_grid.flatten(), 'cdi': cdi_vec})
cdi_df_crop=cdi_df.dropna()


print(cdi_slice.shape)
print(cdi_df.shape)
print(cdi_df_crop.shape)

# Open the second NetCDF file


# %%

ds2 = xr.open_dataset(ncname_rain_final)
 
# Get longitude and latitude
lon_r = ds2["longitude"].values
lat_r = ds2["latitude"].values
 
# Get rain
dname2 = "percentage_of_ensembles"
rain_array = ds2[dname2].values
 
nbins=2
time=1
# Get slices
rain_slice = rain_array[nbins-1,time-1,:,:]
rain_vec = np.ravel(rain_slice)
 
# Create DataFrame for rain data
lon_grid, lat_grid = np.meshgrid(lon_r, lat_r)

rain_df = pd.DataFrame({'lat': lat_grid.flatten(), 'lon': lon_grid.flatten(), 'rain': rain_vec})

# rain_df= rain_df.replace(np.nan, 'NA', regex=True)

time_r = ds2["time"][time-1].values


# Convert the time value to a datetime object
time_r = pd.to_datetime(time_r)


# Extract the month name from the datetime object

month_name = time_r.strftime('%B')
print(rain_slice.shape)
print(rain_df.shape)

print("First month:", month_name)
rain_df_crop=rain_df.dropna()
print(rain_df_crop.head(10))
print(rain_df_crop.shape)


# %% [markdown]
# ## merging CDI and forecast using lat and lon

# %%

# Perform the merge
join_df = cdi_df.merge(rain_df, how='left', on=['lon', 'lat'])
# Merge DataFrames on longitude and latitude
print(join_df.shape)
# join_dims={"lon":"lon","lat":"lat"}
# tes=xr.merge([cdi_df,rain_df],join='left',compat="override")
print("join_df.head(10)")
display(join_df)
# Count the number of rows with NaN values
num_rows_with_nan = join_df.isna().sum().sum()

# Count the number of rows without NaN values
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



# %%
def classify_drought(row):
    cdi, rain = row['cdi'], row['rain']
    if cdi < 0.2:
        if rain < 50:
            if cdi < 0.02:
                return 5  # Persists
            else:
                return 6  # Worsens
        elif rain < 70:
            return 5  # Persists
        else:
            if 0.1 <= cdi < 0.2:
                return 2  # Removed
            else:
                return 3  # Improved
    else:
        if rain < 30:
            return 4  # Develops
        else:
            return 1  # No drought

# %%
ncell = len(rmna_df)

ncores = min(multiprocessing.cpu_count(), 4) 
# Use pathos for multiprocessing
with Pool(ncores) as p:
    try:
        classified = p.map(classify_drought, [rmna_df.iloc[i] for i in range(rmna_df.shape[0])])
    except Exception as e:
        print(f"An error occurred during multiprocessing: {e}")
        p.close() # Close the pool
        p.join() # Wait for the worker processes to exit
        raise # Re-raise the exception

print(len(classified))

# %%
# Create the dataframe df_out
df_out = pd.DataFrame({ 'lat': cdi_df['lat'], 'lon': cdi_df['lon'],'outlook': np.nan})
print(len(rmna_df.index))

# Get the rows where NAs were removed
order = rmna_df.index.astype(int)

# Replace the category value
classified = np.array(classified)
df_out.loc[order, 'outlook'] = classified.astype(int)
print(df_out['outlook'].value_counts())
outlook_counts = df_out['outlook'].value_counts()
for value in range(1, 7):
    if value in outlook_counts:
        print(f"Value {value}: {outlook_counts[value]}")
    else:
        print(f"Value {value}: 0")


# %%
print(cdi_slice.shape)
print(len(lon))
print(len(lat))

# %%
# Create a DataArray from the DataFrame
da = xr.DataArray(df_out['outlook'].values.reshape(cdi_slice.shape),
                  coords=[('lat', lat),('lon', lon)],
                  name='outlook')

# da = xr.DataArray(df_out['outlook'].values, coords=[('lon', cdi_df['lon']), ('lat', cdi_df['lat'])], dims=['lon', 'lat'])

# Add attributes
da.attrs['varunit'] = ''
da.attrs['longname'] = 'drought outlook'

# Create a Dataset from the DataArray
ds = da.to_dataset()
# Add the time variable to the Dataset
ds['time'] = (('time'), [time_r])

# %%
# Save the Dataset as a NetCDF file
out_ncname = "/Users/sabinmaharjan/projects/python/do/static/result/nc/1_months/"+month_name+"_Final_2024.nc"

try:
    ds.to_netcdf(out_ncname)
    print(f"file saved with name: {out_ncname}")
except Exception as e:
    print(f"An error occurred while saving the Dataset: {e}")


