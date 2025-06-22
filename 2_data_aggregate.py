#!/usr/bin/env python3
"""
Data aggregation script - Python version of 2data_aggregate.R
Merges climate and fire data into final analysis dataset
"""

import pandas as pd
import numpy as np
import glob
from pathlib import Path
from output_utils import get_output_path, ensure_output_dirs, get_climate_data_path

def aggregate_data():
    """Aggregate all climate-fire data into single dataset"""
    
    print("Reading climate-fire datasets...")
    
    # Ensure output directory exists
    ensure_output_dirs()
    
    # Read all climate files
    climate_files = glob.glob(get_climate_data_path("*.csv"))
    
    if not climate_files:
        print("No climate files found! Run 1_import_merge.py first.")
        return
    
    print(f"Found {len(climate_files)} climate files to process")
    
    # Read and combine all files
    fire_all = []
    processed_files = 0
    
    for file in climate_files:
        try:
            df = pd.read_csv(file)
            if len(df) > 0:
                fire_all.append(df)
                processed_files += 1
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
    
    print(f"Successfully read {processed_files} files")
    
    if not fire_all:
        print("No valid data files found!")
        return
    
    # Combine all datasets
    fire_all = pd.concat(fire_all, ignore_index=True)
    
    print(f"Total combined records: {len(fire_all):,}")
    
    # Check data quality
    print(f"Records with max_temp: {(~fire_all['max_temp'].isna()).sum():,}")
    print(f"Records with min_temp: {(~fire_all['min_temp'].isna()).sum():,}")
    print(f"Records with rainfall: {(~fire_all['rainfall'].isna()).sum():,}")
    
    # Group by month and year and calculate summary statistics
    print("Calculating monthly aggregates...")
    
    fire_data = (fire_all.groupby(['month', 'year'])
                .agg({
                    'brightness': ['count', 'mean'],
                    'bright_t31': 'mean',
                    'frp': 'mean',
                    'max_temp': 'mean',
                    'min_temp': 'mean',
                    'rainfall': 'mean'
                })
                .reset_index())
    
    # Flatten column names
    fire_data.columns = ['month', 'year', 'count', 'mean_brightness', 
                        'mean_bright31', 'mean_frp', 'mean_max_temp', 
                        'mean_min_temp', 'mean_rainfall']
    
    # Calculate derived variables
    fire_data['anomaly'] = fire_data['mean_max_temp'] - fire_data['mean_min_temp']
    fire_data['average_temp'] = (fire_data['mean_max_temp'] + fire_data['mean_min_temp']) / 2
    
    # Sort by year and month
    fire_data = fire_data.sort_values(['year', 'month']).reset_index(drop=True)
    
    print(f"Final aggregated dataset shape: {fire_data.shape}")
    
    # Show data coverage by year
    print("\nData coverage by year:")
    year_coverage = fire_data.groupby('year').size()
    for year, months in year_coverage.items():
        print(f"  {year}: {months} months")
    
    print(f"\nDate range: {fire_data['year'].min()}-{fire_data['month'].min():02d} to {fire_data['year'].max()}-{fire_data['month'].max():02d}")
    
    print("\nSummary statistics:")
    print(fire_data[['count', 'mean_max_temp', 'mean_min_temp', 'mean_rainfall']].describe())
    
    # Save final dataset to root directory
    output_file = "fire_data_2000-18.csv"
    fire_data.to_csv(output_file, index=False)
    
    print(f"\nFinal dataset saved as: {output_file}")
    
    return fire_data

if __name__ == "__main__":
    fire_data = aggregate_data()
    print("Data aggregation completed!") 