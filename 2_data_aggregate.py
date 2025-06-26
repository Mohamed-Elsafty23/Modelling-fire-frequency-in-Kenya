#!/usr/bin/env python3
"""
Data aggregation script - Python version of 2data_aggregate.R
Merges climate and fire data into final analysis dataset
Enhanced to handle climate data from separate CSV folders
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from output_utils import get_output_path, ensure_output_dirs, get_climate_data_path

def aggregate_climate_data():
    """Aggregate climate data from separate CSV folders"""
    print("Aggregating climate data from separate folders...")
    
    climate_folders = ['climate_tmax_csv', 'climate_tmin_csv', 'climate_rainfall_csv']
    climate_summary = {}
    
    for folder in climate_folders:
        if os.path.exists(folder):
            csv_files = glob.glob(os.path.join(folder, "*.csv"))
            print(f"Found {len(csv_files)} files in {folder}")
            
            # Aggregate data for this climate variable
            all_data = []
            for file in csv_files:
                try:
                    df = pd.read_csv(file)
                    if len(df) > 0:
                        all_data.append(df)
                except Exception as e:
                    print(f"Error reading {file}: {e}")
                    continue
            
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                
                # Extract variable name
                var_name = folder.replace('climate_', '').replace('_csv', '')
                
                # Calculate summary statistics by date
                summary = combined_data.groupby('date').agg({
                    var_name: ['mean', 'std', 'min', 'max', 'count']
                }).reset_index()
                
                # Flatten column names
                summary.columns = ['date', f'{var_name}_mean', f'{var_name}_std', 
                                 f'{var_name}_min', f'{var_name}_max', f'{var_name}_count']
                
                climate_summary[var_name] = summary
                
                # Save individual climate summary
                summary_file = os.path.join(folder, f"{var_name}_summary.csv")
                summary.to_csv(summary_file, index=False)
                print(f"Saved {var_name} summary to {summary_file}")
    
    return climate_summary

def aggregate_data():
    """Aggregate all climate-fire data into single dataset"""
    
    print("Reading climate-fire datasets...")
    
    # Ensure output directory exists
    ensure_output_dirs()
    
    # Aggregate climate data from separate folders (optional)
    if any(os.path.exists(folder) for folder in ['climate_tmax_csv', 'climate_tmin_csv', 'climate_rainfall_csv']):
        climate_summary = aggregate_climate_data()
        print("Climate data aggregation completed.")
    
    # Read all climate files (use the consistent fire-climate naming pattern)
    climate_files = glob.glob(get_climate_data_path("fire-climate_*.csv"))
    
    if not climate_files:
        print("No fire-climate files found! Run 1_import_merge.py first.")
        return
    
    print(f"Found {len(climate_files)} fire-climate files to process")
    
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
    
    # Check data quality before filtering
    print(f"Records with max_temp: {(~fire_all['max_temp'].isna()).sum():,}")
    print(f"Records with min_temp: {(~fire_all['min_temp'].isna()).sum():,}")
    print(f"Records with rainfall: {(~fire_all['rainfall'].isna()).sum():,}")
    
    # Check which months have all 3 climate variables available
    print("Identifying months with complete climate data for ALL 3 variables...")
    
    # Get available dates from each climate variable folder
    tmax_files = glob.glob('./climate_tmax_csv/tmax_*.csv')
    tmin_files = glob.glob('./climate_tmin_csv/tmin_*.csv') 
    rainfall_files = glob.glob('./climate_rainfall_csv/rainfall_*.csv')
    
    # Extract dates from filenames (excluding summary files)
    def extract_dates_from_files(file_list, prefix):
        dates = []
        for file in file_list:
            basename = os.path.basename(file)
            if basename.startswith(prefix) and 'summary' not in basename:
                date_part = basename.replace(f'{prefix}_', '').replace('.csv', '')
                dates.append(date_part)
        return set(dates)
    
    tmax_dates = extract_dates_from_files(tmax_files, 'tmax')
    tmin_dates = extract_dates_from_files(tmin_files, 'tmin')
    rainfall_dates = extract_dates_from_files(rainfall_files, 'rainfall')
    
    print(f"Available dates - tmax: {len(tmax_dates)}, tmin: {len(tmin_dates)}, rainfall: {len(rainfall_dates)}")
    
    # Find intersection - months with ALL 3 climate variables
    complete_climate_dates = tmax_dates & tmin_dates & rainfall_dates
    print(f"Months with ALL 3 climate variables: {len(complete_climate_dates)}")
    
    # Convert dates to year-month format for filtering
    complete_months = []
    for date_str in complete_climate_dates:
        year, month = date_str.split('-')
        complete_months.append({'year': int(year), 'month': int(month)})
    
    complete_months_df = pd.DataFrame(complete_months)
    print(f"Converting to {len(complete_months_df)} year-month combinations")
    
    # Filter fire data to only include months with complete climate data
    print("Filtering fire data to only include months with complete climate variables...")
    initial_count = len(fire_all)
    fire_all = fire_all.merge(complete_months_df, on=['month', 'year'])
    filtered_count = len(fire_all)
    print(f"Filtered from {initial_count:,} to {filtered_count:,} records")
    
    # Now filter individual records to remove any remaining NaN values
    fire_all = fire_all.dropna(subset=['max_temp', 'min_temp', 'rainfall'])
    final_count = len(fire_all)
    print(f"After removing NaN records: {final_count:,} records")
    
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
    print("Note: Dataset only includes months where ALL 3 climate variables (max_temp, min_temp, rainfall) are available")
    
    return fire_data

if __name__ == "__main__":
    fire_data = aggregate_data()
    print("Data aggregation completed!") 