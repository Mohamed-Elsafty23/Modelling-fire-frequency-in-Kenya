#!/usr/bin/env python3
"""
Import and merge files - Python version of 1import_merge.R
Processes MODIS fire data and merges with climate data (rainfall, tmax, tmin)
Enhanced to store climate data in separate folders before merging
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
import os
import glob
from pathlib import Path
from datetime import datetime
from rasterio.features import rasterize
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')
from output_utils import get_output_path, ensure_output_dirs, get_fire_data_path, get_climate_data_path

def create_climate_folders():
    """Create separate folders for climate data storage"""
    climate_folders = ['climate_tmax_csv', 'climate_tmin_csv', 'climate_rainfall_csv']
    for folder in climate_folders:
        os.makedirs(folder, exist_ok=True)
    return climate_folders

def extract_climate_to_csv():
    """Extract climate data from TIFF files and store as CSV in separate folders"""
    print("Extracting climate data from TIFF files...")
    
    # Create climate folders
    create_climate_folders()
    
    # Get climate files
    tmax_files = sorted(glob.glob("./tmax/*.tif"))
    tmin_files = sorted(glob.glob("./tmin/*.tif"))
    rain_files = sorted(glob.glob("./rain/*.tif"))
    
    print(f"Processing {len(tmax_files)} tmax, {len(tmin_files)} tmin, {len(rain_files)} rain files")
    
    # Process each climate variable
    for file_list, folder, var_name in [
        (tmax_files, 'climate_tmax_csv', 'tmax'),
        (tmin_files, 'climate_tmin_csv', 'tmin'), 
        (rain_files, 'climate_rainfall_csv', 'rainfall')
    ]:
        print(f"Processing {var_name} files...")
        for tif_file in file_list:
            try:
                # Extract date from filename
                basename = os.path.basename(tif_file)
                date_part = basename.split('_')[1].replace('.tif', '')
                
                # Read raster and convert to CSV
                with rasterio.open(tif_file) as src:
                    # Read the raster data
                    data = src.read(1)
                    transform = src.transform
                    
                    # Get coordinates for each pixel
                    rows, cols = np.where(~np.isnan(data))
                    
                    # Convert pixel coordinates to geographic coordinates
                    lons, lats = rasterio.transform.xy(transform, rows, cols)
                    values = data[rows, cols]
                    
                    # Create DataFrame
                    climate_df = pd.DataFrame({
                        'latitude': lats,
                        'longitude': lons,
                        var_name: values,
                        'date': date_part
                    })
                    
                    # Filter out invalid values
                    climate_df = climate_df[climate_df[var_name] > 0]
                    
                    # Save to CSV
                    output_file = os.path.join(folder, f"{var_name}_{date_part}.csv")
                    climate_df.to_csv(output_file, index=False)
                    
            except Exception as e:
                print(f"Error processing {tif_file}: {e}")
                continue
    
    print("Climate data extraction completed!")

def import_merge_data():
    """Main function to import and merge MODIS and climate data"""
    
    # Create directories
    output_dir = ensure_output_dirs()
    os.makedirs('fire', exist_ok=True)
    os.makedirs('climate', exist_ok=True)
    
    # Extract climate data first if not already done
    if not os.path.exists('climate_tmax_csv') or len(os.listdir('climate_tmax_csv')) == 0:
        extract_climate_to_csv()
    
    # Import all MODIS files
    print("Importing MODIS fire data...")
    modis_files = glob.glob("./modis_data/*.csv")
    
    modis_all = []
    for file in modis_files:
        df = pd.read_csv(file)
        modis_all.append(df)
    
    # Combine all MODIS data
    modis_all = pd.concat(modis_all, ignore_index=True)
    
    # Add year and month columns
    modis_all['acq_date'] = pd.to_datetime(modis_all['acq_date'])
    modis_all['year'] = modis_all['acq_date'].dt.year
    modis_all['month'] = modis_all['acq_date'].dt.month
    
    print(f"Total MODIS records: {len(modis_all)}")
    print(modis_all.head())
    
    # Create fire data files by year-month
    print("Creating monthly fire files...")
    grouped = modis_all.groupby(['year', 'month'])
    
    for (year, month), group in grouped:
        filename = get_fire_data_path(f"fire_year_{year}-{month}.csv")
        group.to_csv(filename, index=False)
    
    # Get file lists for processing
    fire_files = glob.glob(get_fire_data_path("*.csv"))
    tmax_files = sorted(glob.glob("./tmax/*.tif"))
    tmin_files = sorted(glob.glob("./tmin/*.tif"))
    rain_files = sorted(glob.glob("./rain/*.tif"))
    
    print(f"Found {len(fire_files)} fire files")
    print(f"Found {len(tmax_files)} tmax files")
    print(f"Found {len(tmin_files)} tmin files") 
    print(f"Found {len(rain_files)} rain files")
    
    # Create date-based mapping for proper file matching
    def extract_date_from_climate_file(filename):
        basename = os.path.basename(filename)
        parts = basename.split('_')
        if len(parts) > 1:
            return parts[1].replace('.tif', '')
        return None
    
    # Create mappings: date -> filepath
    tmax_map = {extract_date_from_climate_file(f): f for f in tmax_files if extract_date_from_climate_file(f)}
    tmin_map = {extract_date_from_climate_file(f): f for f in tmin_files if extract_date_from_climate_file(f)}
    rain_map = {extract_date_from_climate_file(f): f for f in rain_files if extract_date_from_climate_file(f)}
    
    # Find common dates with all climate variables
    common_dates = set(tmax_map.keys()) & set(tmin_map.keys()) & set(rain_map.keys())
    print(f"Climate data available for {len(common_dates)} months")
    
    # Process each fire file with date-based matching
    print("Processing fire-climate intersections with date matching...")
    processed_count = 0
    
    for fire_file in fire_files:
        # Extract date from fire filename: fire_year_2000-11.csv
        basename = os.path.basename(fire_file)
        parts = basename.replace('.csv', '').split('_')
        if len(parts) >= 3:
            fire_date = parts[2]  # e.g., "2000-11" or "2001-1"
            
            # Normalize fire date to match climate date format (YYYY-MM)
            if '-' in fire_date:
                year, month = fire_date.split('-')
                fire_date_normalized = f"{year}-{month.zfill(2)}"
            else:
                fire_date_normalized = fire_date
            
            if fire_date_normalized in common_dates:
                try:
                    process_fire_climate(fire_file, tmax_map[fire_date_normalized], tmin_map[fire_date_normalized], rain_map[fire_date_normalized])
                    processed_count += 1
                    if processed_count % 20 == 0:
                        print(f"Processed {processed_count} date-matched files...")
                except Exception as e:
                    print(f"Error processing {fire_date}: {e}")
                    continue
            else:
                print(f"No climate data for {fire_date}")
    
    print(f"Successfully processed {processed_count} files with proper date matching")

def process_fire_climate(fire_file, tmax_file, tmin_file, rain_file):
    """Process individual fire file with climate data"""
    
    # Read fire data
    fire_df = pd.read_csv(fire_file)
    
    if len(fire_df) == 0:
        return
    
    # Create GeoDataFrame from fire points
    geometry = [Point(xy) for xy in zip(fire_df['longitude'], fire_df['latitude'])]
    fire_gdf = gpd.GeoDataFrame(fire_df, geometry=geometry, crs='EPSG:4326')
    
    # Extract climate values for each fire point
    tmax_values = extract_raster_values(fire_gdf, tmax_file)
    tmin_values = extract_raster_values(fire_gdf, tmin_file)
    rain_values = extract_raster_values(fire_gdf, rain_file)
    
    # Add climate data to fire dataframe
    fire_df['max_temp'] = tmax_values
    fire_df['min_temp'] = tmin_values
    fire_df['rainfall'] = rain_values
    
    # Clean values (set invalid values to NaN)
    fire_df['max_temp'] = np.where(fire_df['max_temp'] > 0, fire_df['max_temp'], np.nan)
    fire_df['min_temp'] = np.where(fire_df['min_temp'] > 0, fire_df['min_temp'], np.nan)
    fire_df['rainfall'] = np.where(fire_df['rainfall'] > 0, fire_df['rainfall'], np.nan)
    
    # Generate output filename
    base_name = os.path.basename(tmax_file).replace('.tif', '')
    output_file = get_climate_data_path(f"fire-tmax_{base_name}.csv")
    
    # Save processed data
    fire_df.to_csv(output_file, index=False)

def extract_raster_values(gdf, raster_file):
    """Extract raster values at point locations"""
    try:
        with rasterio.open(raster_file) as src:
            # Reproject points to raster CRS if needed
            if gdf.crs != src.crs:
                gdf_reproj = gdf.to_crs(src.crs)
            else:
                gdf_reproj = gdf
            
            # Extract values
            coords = [(point.x, point.y) for point in gdf_reproj.geometry]
            values = list(src.sample(coords))
            
            # Convert to flat array
            values = [val[0] if val.size > 0 else np.nan for val in values]
            
        return values
        
    except Exception as e:
        print(f"Error reading {raster_file}: {e}")
        return [np.nan] * len(gdf)

if __name__ == "__main__":
    import_merge_data()
    print("Import and merge completed!") 