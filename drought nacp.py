import os
import numpy as np
import xarray as xr
import pandas as pd
from pathos.multiprocessing import ProcessingPool as Pool
import datetime

# File paths
NCNAME_CDI = "/Users/sabinmaharjan/projects/python/do/static/file/cdi_1.nc"
NCNAME_RAIN = "/Users/sabinmaharjan/projects/python/do/static/file/p_atmos_q5_pr_s_maq5_pumedian_20240414_rt.nc"

def load_and_process_ncfile(ncfile):
    """
    Load and process the NetCDF file.
    
    Args:
        ncfile (str): Path to the NetCDF file.
        
    Returns:
        xarray.Dataset: Processed dataset.
    """
    ds = xr.open_dataset(ncfile)
    ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})
    ds['latitude'] = ds['latitude'].astype('double').round(decimals=2)
    ds['longitude'] = ds['longitude'].astype('double').round(decimals=2)
    return ds

def create_dataframe_from_nc(ds, var_name, slice_index=None):
    """
    Create a DataFrame from a NetCDF variable.
    
    Args:
        ds (xarray.Dataset): Dataset containing the variable.
        var_name (str): Name of the variable.
        slice_index (int, optional): Index of the slice to extract (for 3D variables).
        
    Returns:
        pd.DataFrame: DataFrame with latitude, longitude, and variable values.
    """
    lon, lat = ds["longitude"].values, ds["latitude"].values
    var_array = ds[var_name].values
    if slice_index is not None:
        var_slice = var_array[:, :, slice_index]
    else:
        var_slice = var_array
    var_vec = np.ravel(var_slice)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    return pd.DataFrame({'lon': lon_grid.flatten(), 'lat': lat_grid.flatten(), var_name: var_vec})

def classify_drought(row):
    """
    Classify drought based on CDI and rain values.
    
    Args:
        row (pd.Series): Row of the DataFrame.
        
    Returns:
        int: Drought classification.
    """
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

def classify_drought_parallel(df, ncores=4):
    """
    Classify drought in parallel using multiprocessing.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        ncores (int): Number of cores to use for parallel processing.
        
    Returns:
        np.ndarray: Array of drought classifications.
    """
    with Pool(ncores) as p:
        try:
            classified = p.map(classify_drought, [df.iloc[i] for i in range(df.shape[0])])
            return np.array(classified)
        except Exception as e:
            print(f"An error occurred during multiprocessing: {e}")
            p.close()
            p.join()
            raise

def create_and_save_dataset(cdi_df, rmna_df, classified, lat, lon, time_r, out_ncname):
    """
    Create and save a dataset from classified drought data.
    
    Args:
        cdi_df (pd.DataFrame): Original CDI DataFrame.
        rmna_df (pd.DataFrame): DataFrame with NaNs removed.
        classified (np.ndarray): Classified drought data.
        lat (np.ndarray): Latitude values.
        lon (np.ndarray): Longitude values.
        time_r (datetime): Time reference.
        out_ncname (str): Output NetCDF file name.
    """
    df_out = pd.DataFrame({'lat': cdi_df['lat'], 'lon': cdi_df['lon'], 'outlook': np.nan})
    order = rmna_df.index.astype(int)
    df_out.loc[order, 'outlook'] = classified.astype(int)
    df_out['outlook'] = df_out['outlook'].astype(np.float32)
    
    da = xr.DataArray(df_out['outlook'].values.reshape((len(lat), len(lon))),
                      coords=[('lat', lat), ('lon', lon)],
                      dims=['lat', 'lon'],
                      name='outlook')
    da.attrs['varunit'] = ''
    da.attrs['longname'] = 'drought outlook'
    ds = da.to_dataset()
    ds['time'] = (('time'), [time_r])
    
    try:
        ds.to_netcdf(out_ncname, mode='w')
        print(f"File saved with name: {out_ncname}")
    except Exception as e:
        print(f"An error occurred while saving the Dataset: {e}")

def main():
    try:
        ds_rain = load_and_process_ncfile(NCNAME_RAIN)
        ds_cdi = xr.open_dataset(NCNAME_CDI)
        
        cdi_df = create_dataframe_from_nc(ds_cdi, "cdi", slice_index=305)
        rain_df = create_dataframe_from_nc(ds_rain, "percentage_of_ensembles", slice_index=None)
        
        join_df = cdi_df.merge(rain_df, how='left', on=['lon', 'lat'])
        rmna_df = join_df.dropna()
        
        time_r = pd.to_datetime(ds_rain["time"].values[0])
        classified = classify_drought_parallel(rmna_df)
        
        month_name = time_r.strftime('%B')
        out_ncname = f"/Users/sabinmaharjan/projects/python/do/static/result/nc/1_months/nacp_{month_name}_Final_2024.nc"
        
        create_and_save_dataset(cdi_df, rmna_df, classified, ds_cdi["latitude"].values, ds_cdi["longitude"].values, time_r, out_ncname)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
