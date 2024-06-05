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
import rasterio
from shapely.geometry import Point
import geopandas as gpd
from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import matplotlib.colors as colors
import datetime

# %% [markdown]
# ## File path and settings

# %%
# Set the file paths
ncname_cdi = "/Users/sabinmaharjan/projects/python/do/static/file/cdi_1.nc"
ncname_rain = "/Users/sabinmaharjan/projects/python/do/static/file/2024.forecast.nc"
ncname_rain_final="/Users/sabinmaharjan/projects/python/do/static/file/2.nc"
thong_3="/Users/sabinmaharjan/projects/python/do/static/file/thong/droughtoutlook_3m.nc"
thong_1="/Users/sabinmaharjan/projects/python/do/static/file/thong/droughtoutlook_1m.nc"
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

# %%
import numpy as np

# Example array
arr = np.array([1, 2, 3, 2, 4, 1, 3, 4, 4, 4])

# Get unique values and their counts
unique_values, counts = np.unique(arr, return_counts=True)

# Print the results
for value, count in zip(unique_values, counts):
    print(f"{value} occurs {count} times")


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
print(cdi_df_crop.head(10))
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


# %%
import netCDF4 as nc
import pandas as pd

# Open an existing NetCDF file in read/write mode
rootgrp = nc.Dataset(thong_3, 'r+', format='NETCDF4')

# Check if the 'time' dimension exists
if 'time' not in rootgrp.dimensions:
    # Create a new 'time' dimension with length 1
    rootgrp.createDimension('time', 1)

# Check if the 'time' variable exists
if 'time' in rootgrp.variables:
    # Get the existing 'time' variable
    time_var = rootgrp.variables['time']
else:
    # Create a new 'time' variable
    time_var = rootgrp.createVariable('time', 'f8', ('time',))
    time_var.units = "days since 2024-04-16 00:00:00"
    time_var.calendar = "proleptic_gregorian"

# Assume time_r is a Timestamp object
time_r = pd.Timestamp('2024-05-01')  # Example timestamp

# Convert the Timestamp object to a number of days since the reference time
reference_date = pd.Timestamp('2024-04-16')
days_since_ref = (time_r - reference_date).days

# Update the 'time' variable with the number of days
time_var[:] = [days_since_ref]

# Close the NetCDF file
rootgrp.close()

# %% [markdown]
# 

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
# import pandas as pd

# # Create sample DataFrames
# cdi_df = pd.DataFrame({
#     'lon': [1, 2, 3, 4, 5],
#     'lat': [10, 20, 30, 40, 50],
#     'cdi_val': [1.1, 2.2, 3.3, 4.4, 5.5]
# })

# rain_df = pd.DataFrame({
#     'lon': [2, 3, 4, 6],
#     'lat': [20, 30, 40, 60],
#     'rain_val': [100, 200, 300, 400]
# })

# # Perform left join
# join_df_py = pd.merge(cdi_df, rain_df, how='left', on=['lon', 'lat'])

# # Print the joined DataFrame
# print(join_df_py)

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
import pandas as pd
import numpy as np

# Assuming cdi_df and rmna_df are defined elsewhere in your code
# Create the dataframe df_out
df_out = pd.DataFrame({'lat': cdi_df['lat'], 'lon': cdi_df['lon'], 'outlook': np.nan})

# Get the rows where NAs were removed
order = rmna_df.index.astype(int)

# Replace the category value
classified = np.array(classified)
df_out.loc[order, 'outlook'] = classified.astype(int)


# Fill NaN values in 'outlook' column with NaN
df_out['outlook'].fillna(np.nan, inplace=True)
# Explicitly convert the 'outlook' column to float32
df_out['outlook'] = df_out['outlook'].astype(np.float32)
# Keep 'outlook' as float to accommodate NaN values
# df_out['outlook'] = df_out['outlook'].astype(int) # This line is commented out

# Print the value counts of 'outlook'
print(df_out['outlook'].value_counts())
print(df_out['outlook'].unique())
print(df_out['outlook'].isna().sum())

print(df_out['outlook'].dtypes)




# %%
# import geopandas as gpd
# from shapely.geometry import Point
# import xarray as xr
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import rasterio
# from rasterio.crs import CRS

# # Create a GeoDataFrame from the df_out DataFrame
# geometry = [Point(xy) for xy in zip(df_out.lon, df_out.lat)]
# gdf = gpd.GeoDataFrame(df_out, geometry=geometry, crs="EPSG:4326")

# # Create a DataArray from the GeoDataFrame
# da = gdf.set_index(['lat', 'lon'])['outlook'].to_xarray()

# # Add attributes
# da.attrs['varunit'] = ''
# da.attrs['longname'] = 'drought outlook'

# # Create a Dataset from the DataArray
# ds = da.to_dataset(name='outlook')
# print(ds)

# # Add the time variable to the Dataset
# ds['time'] = (('time'), [time_r])

# # Save the Dataset as a NetCDF file
# out_ncname = "/Users/sabinmaharjan/projects/python/do/static/result/nc/1_months/" + month_name + "_Final_2024.nc"
# try:
   
#     ds.to_netcdf(out_ncname)
#     print(f"File saved with name: {out_ncname}")
# except Exception as e:
#     print(f"An error occurred while saving the Dataset: {e}")

# # Optionally, plot the points on a map
# fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
# cmap = colors.ListedColormap(['#E0E0E0', '#016838', '#5BB75B', '#FCDE66', '#B079D1', '#59207D'])
# scatter = gdf.plot(column='outlook', cmap=cmap, legend=True, ax=ax, transform=ccrs.PlateCarree())
# mappable = scatter.collections[0]
# plt.colorbar(mappable, ax=ax, orientation='horizontal', pad=0.05, aspect=50, shrink=0.5)

# plt.show()

# %%

# Create a DataArray from the DataFrame
da = xr.DataArray(df_out['outlook'].values.reshape(cdi_slice.shape),
                  coords=[('lat', lat),('lon', lon)],
                  dims=['lat', 'lon'],
                  name='outlook')

# da = xr.DataArray(df_out['outlook'].values, coords=[('lon', cdi_df['lon']), ('lat', cdi_df['lat'])], dims=['lon', 'lat'])

# Add attributes
da.attrs['varunit'] = ''
da.attrs['longname'] = 'drought outlook'

# Create a Dataset from the DataArray
ds = da.to_dataset()
# Add the time variable to the Dataset
ds['time'] = (('time'), [time_r])
print(ds.dims)  # Print the dimensions of the Dataset
print(ds['outlook'].dims)  # Print the dimensions of the 'outlook' variable

# %%
# Save the Dataset as a NetCDF file
out_ncname = "/Users/sabinmaharjan/projects/python/do/static/result/nc/1_months/"+month_name+"_Final_2024.nc"

try:
    ds.to_netcdf(out_ncname, mode='w')
    print(f"file saved with name: {out_ncname}")
except Exception as e:
    print(f"An error occurred while saving the Dataset: {e}")

# %%
import numpy as np
import xarray as xr
ds_outncname = xr.open_dataset(out_ncname)
# Open the dataset

# Flatten the 'outlook' variable into a 1D array
outlook_values = ds_outncname['outlook'].values.flatten()
print(outlook_values)
# Use numpy.unique() to get unique values and their counts
unique_values, counts = np.unique(outlook_values, return_counts=True)

# Display the unique values and their counts
for value, count in zip(unique_values, counts):
    print(f"Value: {value}, Count: {count}")

# %%
[nan nan nan ... nan nan nan]
Value: 1.0, Count: 115669
Value: 4.0, Count: 140277
Value: 5.0, Count: 1728
Value: 6.0, Count: 15918
Value: nan, Count: 299129


