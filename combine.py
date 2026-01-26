# %% [markdown]
# 
# =====================================================================
# Australian Drought Monitoring and Outlook Generation System
# =====================================================================
# 
# This script processes CDI (Combined Drought Indicator) data and rainfall forecasts
# to generate drought outlook maps for Australia. It calculates both 1-month and 
# 3-month outlooks based on current conditions and forecast rainfall.
# 
# Author: Sabin Maharjan
# Last Updated: March 2025
# 

# %%
import os
import sys
import glob
import logging
import numpy as np
import pandas as pd
import xarray as xr
import datetime
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

# Cartopy imports for map generation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader


# %%
#=====================================================================
# SETUP LOGGING
#=====================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("drought_monitor.log")
    ]
)
logger = logging.getLogger(__name__)

#=====================================================================
# IMPORT OPTIONAL DEPENDENCIES
#=====================================================================

# Try to import pathos for parallel processing
try:
    from pathos.multiprocessing import ProcessingPool as Pool
    import multiprocessing
    HAS_PATHOS = True
    logger.info("Pathos available for parallel processing")
except ImportError:
    HAS_PATHOS = False
    logger.warning("Pathos not available, will use serial processing")

# %% [markdown]
# #=====================================================================
# #CONFIGURATION
# #=====================================================================

# %%
#=====================================================================
# CONFIGURATION
#=====================================================================

CONFIG = {
    'cdi_file': "/Volumes/data/nacp/results/netcdf/cdi_1.nc",  # Path to CDI file
    'forecast_dir': "/Volumes/data/do/static/file/",  # Directory with rainfall forecasts
    'output_dir': "/Volumes/data/do/static/",  # Base output directory
    'shapefile': "/Volumes/data/nacp/DroughtMonitor/shapes/gadm36_AUS_1.shp",  # Shapefile path
    'offset': -1,  # Use -1 to use previous month's CDI
    'forecast_pattern': "p_atmos_q5_pr_s_maq5_pumedian_*.nc",  # Forecast filename pattern
    'overwrite': True,  # Whether to overwrite existing files
    'process_single_file': False,  # Process only one file for testing
    'file_index': 0,  # Index of file to process if process_single_file is True
    'debug': True,  # Enable additional debug info
    'chunk_size': 100,  # Chunk size for xarray operations
    'use_dask': False,  # Use dask for parallel processing
    'max_cores': 4,  # Maximum number of cores to use for parallel processing
    'batch_size': 10000  # Data batch size for classification
}

# %%
#=====================================================================
# HELPER FUNCTIONS
#=====================================================================

def setup_dirs(date_str, offset_str):
    """
    Create and return all necessary output directories.
    
    Args:
        date_str (str): Date string in YYYY-MM format
        offset_str (str): String indicating CDI offset for directory naming
        
    Returns:
        dict: Dictionary containing paths for all output files
    """
    dirs = {
        'nc_1month': os.path.join(CONFIG['output_dir'], f"result/nc/1_months/{offset_str}"),
        'nc_3month': os.path.join(CONFIG['output_dir'], f"result/nc/3_months/{offset_str}"),
        'map_1month': os.path.join(CONFIG['output_dir'], f"result/maps/1_months/{offset_str}"),
        'map_3month': os.path.join(CONFIG['output_dir'], f"result/maps/3_months/{offset_str}"),
        'map_rainfall': os.path.join(CONFIG['output_dir'], f"result/maps/forecast/{offset_str}")
    }
    
    # Create all directories
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    # Create output file paths
    files = {
        'nc_1month': os.path.join(dirs['nc_1month'], f"{date_str}_outlook_1month.nc"),
        'nc_3month': os.path.join(dirs['nc_3month'], f"{date_str}_outlook_3month.nc"),
        'map_1month': os.path.join(dirs['map_1month'], f"drought-outlook_1_{date_str}.jpg"),
        'map_3month': os.path.join(dirs['map_3month'], f"drought-outlook_3_{date_str}.jpg"),
        'map_rainfall': os.path.join(dirs['map_rainfall'], f"rainfall-forecast_{date_str}.jpg")
    }
    
    return files

def load_netcdf(file_path, use_dask=False):
    """
    Load a NetCDF file and return as xarray Dataset.
    
    Args:
        file_path (str): Path to the NetCDF file
        use_dask (bool): Whether to use dask for chunked loading
        
    Returns:
        xarray.Dataset or None: The loaded dataset or None if an error occurs
    """
    try:
        logger.info(f"Loading NetCDF file: {file_path}")
        if use_dask:
            # Use dask for chunked loading
            return xr.open_dataset(file_path, chunks={'time': CONFIG['chunk_size']})
        else:
            return xr.open_dataset(file_path)
    except Exception as e:
        logger.error(f"Error loading NetCDF file {file_path}: {e}")
        return None

def preprocess_dataset(ds, rename_dict=None):
    """
    Preprocess the dataset by renaming variables and rounding coordinates.
    
    Args:
        ds (xarray.Dataset): Dataset to preprocess
        rename_dict (dict): Dictionary mapping old variable names to new ones
        
    Returns:
        xarray.Dataset: Preprocessed dataset
    """
    if rename_dict:
        ds = ds.rename(rename_dict)
    
    # Round coordinates to 2 decimal places for consistency
    if 'latitude' in ds:
        ds['latitude'] = ds['latitude'].astype('float64').round(2)
    if 'longitude' in ds:
        ds['longitude'] = ds['longitude'].astype('float64').round(2)
    
    return ds

def extract_date_from_forecast_file(filepath):
    """
    Extract date information from forecast filename.
    
    Args:
        filepath (str): Path to the forecast file
        
    Returns:
        datetime: Extracted date or current date if extraction fails
    """
    try:
        # Try to extract a date-like pattern from the filename (format like *20240101*.nc with YYYYMMDD)
        filename = os.path.basename(filepath)
        date_parts = [part for part in filename.split('_') if len(part) == 8 and part.isdigit()]
        
        if date_parts:
            date_str = date_parts[0]
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            return datetime(year, month, day)
        else:
            # If no date in filename, try to get date from file content
            ds = xr.open_dataset(filepath)
            if 'time' in ds:
                return pd.to_datetime(ds.time.values[0])
            ds.close()
    except Exception as e:
        logger.error(f"Error extracting date from {filepath}: {e}")
    
    # Default to current date if extraction fails
    logger.warning(f"Could not extract date from {filepath}, using current date")
    return datetime.now()

def find_time_index_in_cdi(ds_cdi, target_date, offset_months=0):
    """
    Find the time index in CDI dataset that corresponds to target_date with optional offset.
    
    Args:
        ds_cdi (xarray.Dataset): CDI dataset
        target_date (datetime or np.datetime64): Target date
        offset_months (int): Offset in months to apply to target date
        
    Returns:
        int: Index in the CDI dataset time dimension
    """
    if isinstance(target_date, np.datetime64):
        target_date = pd.to_datetime(target_date)
        
    # Apply offset
    target_date_with_offset = target_date + pd.DateOffset(months=offset_months)
    
    # Convert target date to year-month format for comparison
    target_ym = target_date_with_offset.strftime('%Y-%m')
    logger.info(f"Finding CDI time index for target date: {target_ym}")
    
    # Convert all times in CDI to year-month format
    cdi_times = pd.to_datetime(ds_cdi.time.values)
    cdi_ym = [dt.strftime('%Y-%m') for dt in cdi_times]
    
    # Print sample dates for debugging
    logger.info(f"First available CDI date: {cdi_ym[0]}")
    logger.info(f"Last available CDI date: {cdi_ym[-1]}")
    
    # Find index of matching year-month
    try:
        matching_idx = cdi_ym.index(target_ym)
        logger.info(f"Found exact match at index {matching_idx}: {cdi_ym[matching_idx]}")
        return matching_idx
    except ValueError:
        # If exact match not found, find the closest date
        logger.warning(f"Exact match for {target_ym} not found in CDI data. Finding closest date.")
        
        # Convert dates to numbers for comparison
        target_num = target_date_with_offset.timestamp()
        cdi_nums = [dt.timestamp() for dt in cdi_times]
        time_diffs = [abs(cdi_num - target_num) for cdi_num in cdi_nums]
        closest_idx = time_diffs.index(min(time_diffs))
        
        logger.info(f"Using {cdi_times[closest_idx].strftime('%Y-%m')} instead (closest available).")
        return closest_idx

def extract_data_slice(ds, var_name, time_index=None, nbins=2, time=1):
    """
    Extract a 2D slice from the dataset with better error handling.
    
    Args:
        ds (xarray.Dataset): Dataset to extract from
        var_name (str): Variable name
        time_index (int, optional): Time index to use
        nbins (int): Bin index for 4D data
        time (int): Alternative time index
        
    Returns:
        numpy.ndarray: 2D data slice
    """
    try:
        data = ds[var_name].values
        logger.info(f"Extracting data slice from {var_name}, shape: {data.shape}")
        
        if data.ndim == 3:
            if time_index is None:
                time_index = -1
            logger.info(f"Using time index {time_index} for 3D data")
            # Handle different dimension ordering
            if 'time' in ds[var_name].dims and ds[var_name].dims[0] == 'time':
                return data[time_index, :, :]
            else:
                return data[:, :, time_index]
        elif data.ndim == 4:
            if time_index is None:
                time_index = 0
            logger.info(f"Using bin {nbins-1}, time index {time_index} for 4D data")
            return data[nbins-1, time_index, :, :]
        else:
            raise ValueError(f"Unexpected number of dimensions: {data.ndim}")
    except Exception as e:
        logger.error(f"Error extracting data slice: {e}")
        raise

def create_dataframe(lon, lat, data, columns):
    """
    Create a DataFrame from grid data.
    
    Args:
        lon (numpy.ndarray): Longitude values
        lat (numpy.ndarray): Latitude values
        data (numpy.ndarray or xarray.DataArray): Data values
        columns (list): Column names for the DataFrame
        
    Returns:
        pandas.DataFrame: DataFrame containing grid data
    """
    try:
        logger.info(f"Creating dataframe from grid data, shape: {data.shape if hasattr(data, 'shape') else 'unknown'}")
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        if isinstance(data, xr.DataArray):
            data_values = data.values.flatten()
        else:
            data_values = data.flatten()
        
        df = pd.DataFrame({
            'lon': lon_grid.flatten(),
            'lat': lat_grid.flatten(),
            columns[-1]: data_values
        })
        logger.info(f"Created dataframe with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error creating dataframe: {e}")
        raise

def classify_drought(row):
    """
    Classify drought conditions based on CDI and rainfall.
    
    Args:
        row (pandas.Series): Row containing 'cdi' and 'rain' values
        
    Returns:
        float: Classification value (1-5) or NaN on error
        
    Classification values:
        1: No Drought
        2: Drought Removed
        3: Drought Improves
        4: Drought Persists
        5: Drought Worsens
    """
    try:
        cdi, rain = row['cdi'], row['rain']
        
        if cdi < 0.2:
            if rain < 50:
                return 5 if cdi < 0.02 else 4  # 5=Drought Worsens, 4=Drought Persists
            elif rain < 70:
                return 4  # Drought Persists
            else:
                return 2 if 0.1 <= cdi < 0.2 else 3  # 2=Drought Removed, 3=Drought Improves
        else:
            return 3 if rain < 30 else 1  # 3=Drought Develops, 1=No Drought
    except Exception as e:
        logger.error(f"Error in classify_drought: {e}")
        return np.nan  # Return NaN on error

def create_output_dataset(df_out, lat, lon, time):
    """
    Create xarray Dataset from output DataFrame.
    
    Args:
        df_out (pandas.DataFrame): DataFrame with 'lat', 'lon', and 'outlook' columns
        lat (numpy.ndarray): Latitude values
        lon (numpy.ndarray): Longitude values
        time (numpy.datetime64): Time value
        
    Returns:
        xarray.Dataset: Output dataset
    """
    try:
        logger.info(f"Creating output dataset with shape: ({len(lat)}, {len(lon)})")
        
        # Create a 2D array filled with NaN
        outlook_values = np.full((len(lat), len(lon)), np.nan, dtype=np.float32)
        
        # Map DataFrame values to the 2D array
        for i, row in df_out.iterrows():
            # Find the indices that match lat/lon values
            lat_idx = np.where(np.isclose(lat, row['lat'], rtol=1e-5))[0]
            lon_idx = np.where(np.isclose(lon, row['lon'], rtol=1e-5))[0]
            
            if len(lat_idx) > 0 and len(lon_idx) > 0:
                outlook_values[lat_idx[0], lon_idx[0]] = row['outlook']
        
        # Create DataArray and Dataset
        da = xr.DataArray(
            outlook_values,
            coords=[('lat', lat), ('lon', lon)],
            dims=['lat', 'lon'],
            name='outlook'
        )
        da.attrs['varunit'] = ''
        da.attrs['longname'] = 'drought outlook'
        
        ds = da.to_dataset()
        ds['time'] = ('time', [time])
        
        return ds
    except Exception as e:
        logger.error(f"Error creating output dataset: {e}")
        raise

def save_netcdf(ds, file_path):
    """
    Save xarray Dataset as NetCDF file.
    
    Args:
        ds (xarray.Dataset): Dataset to save
        file_path (str): Path to save the file to
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        logger.info(f"Saving NetCDF to {file_path}")
        ds.to_netcdf(file_path, mode='w')
        logger.info(f"File saved successfully: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving NetCDF file {file_path}: {e}")
        return False

def print_value_counts(da):
    """
    Print value counts of the DataArray.
    
    Args:
        da (xarray.DataArray): DataArray to count values in
    """
    try:
        # Filter out NaN values
        non_nan_values = da.values[~np.isnan(da.values)]
        unique_values, counts = np.unique(non_nan_values, return_counts=True)
        
        logger.info("Value counts:")
        for value, count in zip(unique_values, counts):
            logger.info(f"  Value: {value}, Count: {count}")
    except Exception as e:
        logger.error(f"Error in print_value_counts: {e}")

def calculate_3month_average(ds, var_name):
    """
    Calculate 3-month average from the dataset.
    
    Args:
        ds (xarray.Dataset): Dataset containing rainfall data
        var_name (str): Variable name
        
    Returns:
        xarray.DataArray: 3-month average data
    """
    try:
        rain_var = ds[var_name]
        logger.info(f"Calculating 3-month average from {var_name}, shape: {rain_var.shape}")
        
        if rain_var.ndim == 4 and rain_var.shape[1] >= 3:
            logger.info("Taking average of first three time slices")
            first_three_months = rain_var[1, :3, :, :]
            return first_three_months.mean(dim="time")
        else:
            logger.warning("Not enough time slices for 3-month average, using first slice")
            if rain_var.ndim == 4:
                return rain_var[1, 0, :, :]
            else:
                raise ValueError(f"Unexpected dimensions for rainfall data: {rain_var.ndim}")
    except Exception as e:
        logger.error(f"Error calculating 3-month average: {e}")
        raise


# %%
#=====================================================================
# MAIN PROCESSING FUNCTIONS
#=====================================================================

def process_outlook(cdi_df, rain_df, ds_cdi, time, output_path, outlook_type):
    """
    Process outlook for either 1-month or 3-month average.
    
    Args:
        cdi_df (pandas.DataFrame): DataFrame with CDI data
        rain_df (pandas.DataFrame): DataFrame with rainfall data
        ds_cdi (xarray.Dataset): CDI dataset
        time (numpy.datetime64): Time value
        output_path (str): Path to save output NetCDF
        outlook_type (str): Type of outlook ('1-month' or '3-month average')
        
    Returns:
        xarray.Dataset or None: Output dataset or None on error
    """
    try:
        logger.info(f"Processing {outlook_type} outlook")
        logger.info(f"CDI DataFrame shape: {cdi_df.shape}")
        logger.info(f"Rain DataFrame shape: {rain_df.shape}")
        
        # Merge dataframes
        logger.info("Merging dataframes...")
        join_df = cdi_df.merge(rain_df, how='left', on=['lon', 'lat'])
        logger.info(f"Merged DataFrame shape: {join_df.shape}")
        
        # Drop NaN values
        logger.info("Dropping NaN values...")
        rmna_df = join_df.dropna()
        logger.info(f"DataFrame after dropping NaNs: {rmna_df.shape}")
        
        if rmna_df.empty:
            logger.warning(f"No valid data after merging for {outlook_type}. Skipping.")
            return None
        
        # Classify drought conditions
        logger.info("Classifying drought conditions...")
        
        # Process in smaller batches to avoid memory issues
        classified = []
        batch_size = CONFIG['batch_size']
        total_batches = (len(rmna_df) + batch_size - 1) // batch_size
        
        for i in range(0, len(rmna_df), batch_size):
            logger.info(f"Processing batch {i//batch_size + 1}/{total_batches}")
            batch = rmna_df.iloc[i:min(i+batch_size, len(rmna_df))]
            
            if HAS_PATHOS:
                num_cores = min(multiprocessing.cpu_count(), CONFIG['max_cores'])
                with Pool(num_cores) as p:
                    batch_results = p.map(classify_drought, [batch.iloc[j] for j in range(len(batch))])
            else:
                batch_results = [classify_drought(batch.iloc[j]) for j in range(len(batch))]
                
            classified.extend(batch_results)
        
        logger.info(f"Classification complete. Processed {len(classified)} data points.")
        
        # Create output DataFrame with the classifications
        logger.info("Creating output DataFrame...")
        df_out = pd.DataFrame({
            'lat': rmna_df['lat'].values,
            'lon': rmna_df['lon'].values,
            'outlook': classified
        })
        
        # Create output dataset
        logger.info("Creating output dataset...")
        ds_out = create_output_dataset(df_out, ds_cdi.latitude.values, ds_cdi.longitude.values, time)
        
        # Save to NetCDF
        logger.info("Saving output to NetCDF...")
        save_netcdf(ds_out, output_path)
        
        # Print value counts
        logger.info(f"Value counts for {outlook_type} outlook:")
        print_value_counts(ds_out.outlook)
        
        return ds_out
        
    except Exception as e:
        logger.error(f"Error in process_outlook: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def process_drought_outlook(cdi_path, rain_path, output_path_1month, output_path_3month, cdi_offset_months=0, overwrite=True):
    """
    Main function to process drought outlook for both 1-month and 3-month average.
    
    Args:
        cdi_path (str): Path to CDI NetCDF file
        rain_path (str): Path to rainfall forecast NetCDF file
        output_path_1month (str): Path to save 1-month outlook NetCDF
        output_path_3month (str): Path to save 3-month outlook NetCDF
        cdi_offset_months (int): Offset in months to apply to CDI data (e.g., -1 for previous month)
        overwrite (bool): Whether to overwrite existing files
        
    Returns:
        tuple: (ds_1month, ds_3month, ds_rain) - Output datasets
    """
    try:
        # Check if output files already exist and we're not overwriting
        if not overwrite and os.path.exists(output_path_1month) and os.path.exists(output_path_3month):
            logger.info(f"Output files already exist and overwrite=False. Skipping processing.")
            return None, None, None
        
        # Load CDI data
        logger.info("Loading CDI data...")
        ds_cdi = load_netcdf(cdi_path, CONFIG['use_dask'])
        if ds_cdi is None:
            logger.error("Failed to load CDI data.")
            return None, None, None
            
        logger.info("Preprocessing CDI data...")
        ds_cdi = preprocess_dataset(ds_cdi)
        
        # Load rainfall data
        logger.info("Loading rainfall forecast data...")
        ds_rain = load_netcdf(rain_path)
        if ds_rain is None:
            logger.error("Failed to load rainfall forecast data.")
            return None, None, None
            
        logger.info("Preprocessing rainfall data...")
        ds_rain = preprocess_dataset(ds_rain, {'lat': 'latitude', 'lon': 'longitude'})
        
        # Extract rainfall forecast date
        forecast_date = pd.to_datetime(ds_rain.time.values[0])
        forecast_date_str = forecast_date.strftime('%Y-%m')
        rain_filename = os.path.basename(rain_path)
        
        # Find the appropriate CDI time index based on rainfall forecast date and offset
        logger.info("Finding appropriate CDI time index...")
        cdi_time_idx = find_time_index_in_cdi(ds_cdi, forecast_date, cdi_offset_months)
        cdi_date = pd.to_datetime(ds_cdi.time.values[cdi_time_idx])
        cdi_date_str = cdi_date.strftime('%Y-%m')
        
        # Print detailed information about the calculation being performed
        logger.info("\n" + "="*80)
        logger.info(f"CALCULATION DETAILS:")
        logger.info(f"Rain forecast file: {rain_filename}")
        logger.info(f"Rain forecast date: {forecast_date_str}")
        logger.info(f"CDI offset applied: {cdi_offset_months} months")
        logger.info(f"CDI date used: {cdi_date_str} (index {cdi_time_idx})")
        logger.info(f"Output 1-month: {os.path.basename(output_path_1month)}")
        logger.info(f"Output 3-month: {os.path.basename(output_path_3month)}")
        logger.info("="*80 + "\n")
        
        # Extract CDI slice for the appropriate time
        logger.info("Extracting CDI data slice...")
        cdi_slice = extract_data_slice(ds_cdi, "cdi", time_index=cdi_time_idx)
        
        logger.info("Creating CDI DataFrame...")
        cdi_df = create_dataframe(ds_cdi.longitude.values, ds_cdi.latitude.values, cdi_slice, ['lon', 'lat', 'cdi'])

        # Process 1-month outlook
        logger.info("\n=== Processing 1-month outlook ===")
        rain_slice_1month = extract_data_slice(ds_rain, "percentage_of_ensembles")
        rain_df_1month = create_dataframe(ds_rain.longitude.values, ds_rain.latitude.values, rain_slice_1month, ['lon', 'lat', 'rain'])
        ds_1month = process_outlook(cdi_df, rain_df_1month, ds_cdi, ds_rain.time[0].values, output_path_1month, "1-month")

        # Process 3-month average outlook
        logger.info("\n=== Processing 3-month outlook ===")
        try:
            rain_avg_3month = calculate_3month_average(ds_rain, "percentage_of_ensembles")
            rain_df_3month = create_dataframe(ds_rain.longitude.values, ds_rain.latitude.values, rain_avg_3month, ['lon', 'lat', 'rain'])
            ds_3month = process_outlook(cdi_df, rain_df_3month, ds_cdi, ds_rain.time[0].values, output_path_3month, "3-month average")
        except Exception as e:
            logger.error(f"Error processing 3-month outlook: {e}")
            import traceback
            logger.error(traceback.format_exc())
            ds_3month = None
        
        return ds_1month, ds_3month, ds_rain
    
    except Exception as e:
        logger.error(f"Error in process_drought_outlook: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None

# %%
#=====================================================================
# MAP GENERATION FUNCTIONS
#=====================================================================

def generate_drought_outlook_map(ncname, shp_path, output_path, map_type='1_month'):
    """
    Generate a drought outlook map.
    
    Args:
        ncname (str or xarray.Dataset): Path to the NetCDF file or xarray Dataset
        shp_path (str): Path to the shapefile
        output_path (str): Path to save the output map
        map_type (str): '1_month' or '3_months'
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Open the NetCDF file or use the provided dataset
        logger.info(f"Generating {map_type} drought outlook map...")
        
        if isinstance(ncname, str):
            if not os.path.exists(ncname):
                logger.warning(f"NetCDF file not found: {ncname}")
                return False
            ds = xr.open_dataset(ncname)
        else:
            ds = ncname
            
        if ds is None:
            logger.warning(f"Could not open NetCDF data for {map_type} map")
            return False
        
        # Get the 2D data array for outlook
        try:
            outlook_data = ds['outlook'].values
            lat = ds['lat'].values
            lon = ds['lon'].values
        except KeyError as e:
            logger.error(f"Error: Required variable not found in dataset: {e}")
            return False
        
        # Define color levels and corresponding category colors
        colors = ['#FFFFFF', '#016838', '#5BB75B', '#FCDE66', '#B079D1', '#59207D']
        categories = ['No Drought', 'Drought Removed', 'Drought Improves', 'Drought Develops', 'Drought Persists', 'Drought Worsens']
        
        # Create a ListedColormap
        cmap = mcolors.ListedColormap(colors)
        
        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection=ccrs.PlateCarree()))
        
        # Plot the pseudocolor map
        im = ax.pcolormesh(lon, lat, outlook_data, cmap=cmap, shading='auto')
        
        # Load and add shapefile geometries
        if os.path.exists(shp_path):
            logger.info(f"Adding shapefile geometries from {shp_path}")
            shp_feat = shpreader.Reader(shp_path).geometries()
            ax.add_geometries(shp_feat, ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.3)
        else:
            logger.warning(f"Shapefile not found: {shp_path}")
        
        # Create custom legend
        legend_entries = [Line2D([0], [0], marker='s', color='none', label=category, markerfacecolor=color, markersize=10, linestyle='None') 
                          for color, category in zip(colors, categories)]
        ax.legend(handles=legend_entries, loc='upper right', title='Drought Outlook', framealpha=0, 
                  bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes, borderaxespad=0.5)
        
        # Extract the forecast date from the dataset
        if 'time' in ds:
            forecast_date = pd.to_datetime(ds.time.values[0])
            current_month_name = forecast_date.strftime("%B")
            current_year = forecast_date.year
        else:
            # If time not in dataset, use current date
            forecast_date = datetime.now()
            current_month_name = forecast_date.strftime("%B")
            current_year = forecast_date.year
        
        if map_type == '1_month':
            subtitle_text = f"{current_month_name} {current_year}"
        else:
            if pd.Timestamp(forecast_date).month + 2 > 12:
                end_month = pd.Timestamp(forecast_date) + pd.DateOffset(months=2)
            else:
                end_month = datetime(
                    year=forecast_date.year,
                    month=(forecast_date.month + 2) % 12 or 12,
                    day=1
                )
            subtitle_text = f"{current_month_name} {current_year} - {end_month.strftime('%B')} {end_month.year}"
        
        title = TextArea('Drought Outlook', textprops=dict(color='black', size=24, weight='bold'))
        subtitle = TextArea(subtitle_text, textprops=dict(color='black', size=18))
        anchored_box = AnchoredOffsetbox(loc='lower left', child=VPacker(children=[title, subtitle], align='left', pad=5, sep=5),
                                        frameon=False, bbox_to_anchor=(0.08, 0.06), bbox_transform=ax.transAxes,
                                        borderpad=0.1)
        ax.add_artist(anchored_box)
        
        # Add watermark
        ax.text(0.5, 0.5, 'Prototype', transform=ax.transAxes, fontsize=80, color='gray', alpha=0.2, ha='center', va='center', rotation=4)
        
        # Set extent to focus on Australia
        ax.set_extent([110, 155, -45, -10], crs=ccrs.PlateCarree())
        
        # Make sure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the plot
        logger.info(f"Saving map to {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
        logger.info(f"Drought outlook map saved: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error generating drought outlook map: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def generate_rainfall_forecast_map(ncname, shp_path, output_path):
    """
    Generate a rainfall forecast map.
    
    Args:
        ncname (str or xarray.Dataset): Path to the NetCDF file or xarray Dataset
        shp_path (str): Path to the shapefile
        output_path (str): Path to save the output map
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Generating rainfall forecast map...")
        
        # Open the NetCDF file
        if isinstance(ncname, str):
            if not os.path.exists(ncname):
                logger.warning(f"NetCDF file not found: {ncname}")
                return False
            ds = xr.open_dataset(ncname)
        else:
            ds = ncname
            
        if ds is None:
            logger.warning("Could not open NetCDF data for rainfall forecast map")
            return False
        
        try:
            # Get the rainfall forecast data
            rain_data = ds['percentage_of_ensembles'].values[1, 0, :, :]  # Assuming we want the second bin and first time step
            lat = ds['latitude'].values
            lon = ds['longitude'].values
            
            # Get the forecast time
            forecast_time = ds['time'].values[0]
            forecast_date = pd.to_datetime(forecast_time)
            forecast_month = forecast_date.strftime('%B %Y')
        except (KeyError, IndexError) as e:
            logger.error(f"Error accessing rainfall forecast data: {e}")
            return False
        
        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection=ccrs.PlateCarree()))
        
        # Create a custom colormap
        colors = ['#a63603', '#e6550d', '#fdae6b', '#fee6ce', '#efedf5', '#bcbddc', '#807dba', '#4a1486']
        cmap = plt.cm.colors.ListedColormap(colors)
        bounds = [20, 30, 40, 50, 60, 70, 80, 90, 100]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        # Plot the rainfall forecast data
        im = ax.pcolormesh(lon, lat, rain_data, cmap=cmap, norm=norm, shading='auto')
        
        # Add features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        
        # Load and add shapefile geometries
        if os.path.exists(shp_path):
            shp_feat = shpreader.Reader(shp_path).geometries()
            ax.add_geometries(shp_feat, ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.5)
        else:
            logger.warning(f"Shapefile not found: {shp_path}")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, aspect=30, shrink=0.6)
        cbar.set_label('Chance of exceeding median rainfall (%)', fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        
        # Add title and subtitle
        title = TextArea('Chance of exceeding the median rainfall', textprops=dict(color='black', size=16, weight='bold'))
        subtitle = TextArea(forecast_month, textprops=dict(color='black', size=14))
        anchored_box = AnchoredOffsetbox(loc='lower left', child=VPacker(children=[title, subtitle], align='left', pad=5, sep=5),
                                        frameon=False, bbox_to_anchor=(0.05, 0.05), bbox_transform=ax.transAxes,
                                        borderpad=0)
        ax.add_artist(anchored_box)
        
        # Set extent to focus on Australia
        ax.set_extent([110, 155, -45, -10], crs=ccrs.PlateCarree())
        
        # Make sure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
        logger.info(f"Rainfall forecast map saved: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error generating rainfall forecast map: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# %%
#=====================================================================
# FILE PROCESSING FUNCTION
#=====================================================================

def process_file(file_to_process):
    """
    Process a single forecast file.
    
    Args:
        file_to_process (str): Path to the forecast file to process
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Extract date from filename
        date = extract_date_from_forecast_file(file_to_process)
        date_str = date.strftime("%Y-%m")
        
        # Create offset string for directory names
        offset_str = f"offset{CONFIG['offset']}" if CONFIG['offset'] != 0 else "current"
        
        # Setup output directories and file paths
        files = setup_dirs(date_str, offset_str)
        
        # Process drought outlook
        ds_1month, ds_3month, ds_rain = process_drought_outlook(
            CONFIG['cdi_file'],
            file_to_process,
            files['nc_1month'],
            files['nc_3month'],
            CONFIG['offset'],
            CONFIG['overwrite']
        )
        
        # Generate maps
        success = True
        if ds_1month is not None:
            success = success and generate_drought_outlook_map(ds_1month, CONFIG['shapefile'], files['map_1month'], map_type='1_month')
        if ds_3month is not None:
            success = success and generate_drought_outlook_map(ds_3month, CONFIG['shapefile'], files['map_3month'], map_type='3_months')
        if ds_rain is not None:
            success = success and generate_rainfall_forecast_map(ds_rain, CONFIG['shapefile'], files['map_rainfall'])
            
        return success
    except Exception as e:
        logger.error(f"Error processing file {file_to_process}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# %%
# def find_missing_first_of_month_dates(cdi_path, start_date="1998-04-01"):
#     ds = xr.open_dataset(cdi_path)

#     # Decode time
#     try:
#         ds_decoded = xr.decode_cf(ds)
#         time_values = ds_decoded.time.values
#     except Exception:
#         time_units = ds['time'].attrs.get('units', 'days since 1998-04-01')
#         time_values = pd.to_datetime(ds['time'].values, origin=pd.Timestamp(start_date), unit='D')

#     actual_dates = pd.to_datetime(time_values)
#     expected_dates = pd.date_range(start=start_date, end=datetime.today(), freq='MS')  # 1st of each month

#     actual_set = set(actual_dates)
#     missing_dates = [d for d in expected_dates if d not in actual_set]

#     return missing_dates

# def check_missing_cdi_dates():
#     """Check missing YYYY-MM-01 dates in multiple CDI files."""
#     CDI_SUFFIXES = ["1", "3", "6", "9", "12", "24", "36"]

#     print("\n🔍 Checking missing first-of-month dates in all CDI files...")
    
#     for suffix in CDI_SUFFIXES:
#         cdi_file = os.path.join(CONFIG['output_dir'], f"file/cdi_{suffix}.nc")
#         output_file = os.path.join(CONFIG['output_dir'], f"missing_first_of_months_cdi_{suffix}.txt")

#         if not os.path.exists(cdi_file):
#             print(f"⚠️ File not found: {cdi_file}")
#             continue

#         print(f"\n📂 Processing: {os.path.basename(cdi_file)}")
#         missing = find_missing_first_of_month_dates(cdi_file)

#         if CONFIG['debug']:
#             for d in missing:
#                 print(f"🚫 {d.strftime('%Y-%m-%d')}")

#         with open(output_file, "w") as f:
#             for d in missing:
#                 f.write(d.strftime("%Y-%m-%d") + "\n")

#         print(f"✅ Missing dates saved to: {output_file}")

# %%
#=====================================================================
# MAIN EXECUTION
#=====================================================================

def main():
    """Main execution function."""
    logger.info("Starting drought monitoring and outlook generation...")
    
    # Find forecast files
    forecast_files = sorted(glob.glob(os.path.join(CONFIG['forecast_dir'], CONFIG['forecast_pattern'])))
    logger.info(f"Found {len(forecast_files)} forecast files:")
    
    for i, file in enumerate(forecast_files):
        logger.info(f" {i}: {os.path.basename(file)}")
    
    # Process files
    if CONFIG['process_single_file'] and forecast_files:
        # Process a single file (for testing)
        file_to_process = forecast_files[CONFIG['file_index']]
        logger.info(f"\n===== Processing single file: {os.path.basename(file_to_process)} =====")
        process_file(file_to_process)
    else:
        # Process all files
        for i, file_to_process in enumerate(forecast_files):
            logger.info(f"\n===== Processing file {i+1}/{len(forecast_files)}: {os.path.basename(file_to_process)} =====")
            process_file(file_to_process)
    
    logger.info("Drought monitoring and outlook generation completed.")

   

if __name__ == "__main__":
    main()


