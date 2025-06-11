#!/usr/bin/env python3
"""
Import and merge files - Python version of 1import_merge.R
Processes MODIS fire data and merges with climate data (rainfall, tmax, tmin)
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
from output_utils import get_output_path, ensure_output_dirs

def import_merge_data():
    """Main function to import and merge MODIS and climate data"""
    
    # Create directories
    output_dir = ensure_output_dirs()
    os.makedirs(get_output_path("fire"), exist_ok=True)
    os.makedirs(get_output_path("climate"), exist_ok=True)
    
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
        filename = get_output_path(f"fire/fire_year_{year}-{month}.csv")
        group.to_csv(filename, index=False)
    
    # Get file lists for processing
    fire_files = glob.glob(get_output_path("fire/*.csv"))
    tmax_files = sorted(glob.glob("./tmax/*.tif"))
    tmin_files = sorted(glob.glob("./tmin/*.tif"))
    rain_files = sorted(glob.glob("./rain/*.tif"))
    
    print(f"Found {len(fire_files)} fire files")
    print(f"Found {len(tmax_files)} tmax files")
    print(f"Found {len(tmin_files)} tmin files") 
    print(f"Found {len(rain_files)} rain files")
    
    # Process each fire file with corresponding climate data
    print("Processing fire-climate intersections...")
    
    for i, fire_file in enumerate(fire_files):
        if i >= len(tmax_files) or i >= len(tmin_files) or i >= len(rain_files):
            print(f"Skipping {fire_file} - no corresponding climate data")
            continue
            
        try:
            process_fire_climate(fire_file, tmax_files[i], tmin_files[i], rain_files[i])
            if i % 10 == 0:
                print(f"Processed {i+1}/{len(fire_files)} files")
        except Exception as e:
            print(f"Error processing {fire_file}: {e}")
            continue

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
    output_file = get_output_path(f"climate/fire-tmax_{base_name}.csv")
    
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