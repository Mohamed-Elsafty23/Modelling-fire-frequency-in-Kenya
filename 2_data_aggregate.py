#!/usr/bin/env python3
"""
Data aggregation script - Python version of 2data_aggregate.R
Merges climate and fire data into final analysis dataset
"""

import pandas as pd
import numpy as np
import glob
from pathlib import Path
from output_utils import get_output_path, ensure_output_dirs

def aggregate_data():
    """Aggregate all climate-fire data into single dataset"""
    
    print("Reading climate-fire datasets...")
    
    # Ensure output directory exists
    ensure_output_dirs()
    
    # Read all climate files
    climate_files = glob.glob(get_output_path("climate/*.csv"))
    
    if not climate_files:
        print("No climate files found! Run 1_import_merge.py first.")
        return
    
    # Read and combine all files
    fire_all = []
    for file in climate_files:
        try:
            df = pd.read_csv(file)
            fire_all.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
    
    if not fire_all:
        print("No valid data files found!")
        return
    
    # Combine all datasets
    fire_all = pd.concat(fire_all, ignore_index=True)
    
    print(f"Total combined records: {len(fire_all)}")
    print(fire_all.head())
    
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
    print("\nSummary statistics:")
    print(fire_data[['count', 'mean_max_temp', 'mean_min_temp', 'mean_rainfall']].describe())
    
    # Save final dataset
    output_file = get_output_path("fire_data_2000-18.csv")
    fire_data.to_csv(output_file, index=False)
    
    print(f"\nFinal dataset saved as: {output_file}")
    
    return fire_data

if __name__ == "__main__":
    fire_data = aggregate_data()
    print("Data aggregation completed!") 