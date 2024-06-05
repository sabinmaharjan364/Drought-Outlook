
import os
import numpy as np
import xarray as xr
import pandas as pd
from functools import partial
from joblib import Parallel, delayed
import netCDF4 as nc
# Set the file paths
ncname_cdi = "/Users/sabinmaharjan/projects/python/do/static/do/cdi_1.nc"
ncname_rain = "/Users/sabinmaharjan/projects/python/do/static/do/2024.forecast.nc"
out_ncname = "/Users/sabinmaharjan/projects/python/do/static/do/results/nc/final_mn_2024.nc"
# Open the CDI file
with nc.Dataset(ncname_cdi, 'r') as ncvar:
    lon = ncvar.variables['longitude'][:]
    lat = ncvar.variables['latitude'][:]
    m=305
    cdi_array = ncvar.variables['cdi'][:, :, -1]
    cdi_vec = cdi_array.ravel()
    cdi_df = pd.DataFrame({'lon': np.repeat(lon, len(lat)), 'lat': np.tile(lat, len(lon)), 'cdi': cdi_vec})
    print(cdi_df.head(10))
# Open the rainfall file
with nc.Dataset(ncname_rain, 'r') as ncvar:
    lon_r = ncvar.variables['lon'][:]
    lat_r = ncvar.variables['lat'][:]
    n=0
    m=1
    rain_array = ncvar.variables['percentage_of_ensembles'][m,n, :, :]
    rain_vec = rain_array.ravel()
    rain_df = pd.DataFrame({'lon': np.round(np.repeat(lon_r, len(lat_r)), 2), 'lat': np.round(np.tile(lat_r, len(lon_r)), 2), 'rain': rain_vec})
    print(rain_df.head(10))
# Join the CDI and rainfall dataframes
join_df = pd.merge(cdi_df, rain_df, on=['lon', 'lat'], how='left')
print("join_df.head(10)")
print(join_df.head(10))
# Define the drought classification function
def classify_drought(row):
    cdi, rain = row['cdi'], row['rain']
    if cdi < 0.2:
        if rain < 50:
            if cdi < 0.02:
                return 5 # Persists
            else:
                return 6 # Worsens
        elif rain < 70:
            return 5 # Persists
        else:
            if 0.1 <= cdi < 0.2:
                return 2 # Removed
            else:
                return 3 # Improved
    else:
        if rain < 30:
            return 4 # Develops
        else:
            return 1 # No drought
# Apply the drought classification in parallel
join_df['drought_outlook'] = Parallel(n_jobs=-1)(delayed(classify_drought)(row) for _, row in join_df.iterrows())
print("join_df.head(10)")
print(join_df.head(10))
not_nan_mask = pd.notna(join_df['drought_outlook'])
# Filter the DataFrame to get the first 10 rows with non-NaN values in 'drought_outlook' column
non_nan_rows = join_df[not_nan_mask].head(10)
# Print the resulting DataFrame
print(non_nan_rows)
# Create the output raster
out_df = pd.DataFrame({'longitude': cdi_df['lon'], 'latitude': cdi_df['lat'], 'drought_outlook': np.nan})
print("out_df.head(10)")
print(out_df.head(10))# Create a mask to filter out NaN values in the 'drought_outlook' column
not_nan_mask = pd.notna(out_df['drought_outlook'])
# Filter the DataFrame to get the first 10 rows with non-NaN values in 'drought_outlook' column
non_nan_rows = out_df[not_nan_mask].head(10)
# Print the resulting DataFrame
print(non_nan_rows)

out_df.loc[join_df.index, 'drought_outlook'] = join_df['drought_outlook']
print(out_df.head(10))
out_xr = xr.Dataset({'drought_outlook': (['latitude', 'longitude'], out_df['drought_outlook'].values.reshape(len(lat), len(lon)))},
                    coords={'longitude': lon, 'latitude': lat})
# Filter the DataFrame to get the first 10 rows with NaN values in 'drought_outlook' column
nan_rows = out_df[pd.isna(out_df['drought_outlook'])].head(10)
# Print the resulting DataFrame
print(nan_rows)

# Save the output to a netCDF file
out_xr.to_netcdf(out_ncname, encoding={'drought_outlook': {'_FillValue': np.nan}})
print("Done")
