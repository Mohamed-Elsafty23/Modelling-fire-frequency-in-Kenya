from output_utils import get_output_path, get_model_results_path, get_simulated_data_path, ensure_output_dirs
#!/usr/bin/env python3
"""
Master script to run all model evaluations - Python version of all 7models_*.R
Runs standard and Bayesian NB models on all simulated datasets (5, 10, 20, 30 years)
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import time

# Import model functions
from model_functions import negbinner, stanbinner

def process_single_file(file_path, model_func, theta, n_months):
    """Process a single simulated dataset file"""
    try:
        # Read data
        data = pd.read_csv(file_path)
        
        # Run model
        result = model_func(data, theta=theta, n=n_months)
        
        return result
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {
            'rmse_train': np.nan, 'rmse_test': np.nan,
            'mase_test': np.nan, 'bias_test': np.nan,
            'theta': theta, 'n': n_months
        }

def run_models_on_theta(theta_value, model_func, model_name, time_period, n_months, max_workers=4):
    """Run models on all datasets for a specific theta value and time period"""
    
    print(f"\n{'='*60}")
    print(f"Running {model_name} models: {time_period} years, theta = {theta_value}")
    print('='*60)
    
    # Get file paths
    data_path = get_simulated_data_path(f"d{time_period}year/theta_{theta_value}")
    file_pattern = f"{data_path}/*.csv"
    file_paths = glob.glob(file_pattern)
    
    if not file_paths:
        print(f"No data files found in {data_path}")
        return pd.DataFrame()
    
    print(f"Found {len(file_paths)} datasets to process")
    
    # Process files
    results = []
    start_time = time.time()
    
    if max_workers > 1 and model_name.lower() == 'nb':
        # Parallel processing for standard models
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(process_single_file, file_path, model_func, theta_value, n_months): file_path
                for file_path in file_paths
            }
            
            for i, future in enumerate(as_completed(future_to_file)):
                result = future.result()
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    eta = (len(file_paths) - i - 1) / rate
                    print(f"Processed {i + 1}/{len(file_paths)} files | Rate: {rate:.1f}/sec | ETA: {eta/60:.1f}min")
    else:
        # Sequential processing for Bayesian models (or if max_workers=1)
        for i, file_path in enumerate(file_paths):
            result = process_single_file(file_path, model_func, theta_value, n_months)
            results.append(result)
            
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(file_paths) - i - 1) / rate
                print(f"Processed {i + 1}/{len(file_paths)} files | Rate: {rate:.1f}/sec | ETA: {eta/60:.1f}min")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    os.makedirs(get_model_results_path(""), exist_ok=True)
    
    time_suffix = {5: "five", 10: "ten", 20: "twenty", 30: "thirty"}[time_period]
    
    if model_name.lower() == 'nb':
        output_file = get_model_results_path(f"{time_suffix}_year_{theta_value}_metrics.csv")
    else:  # Bayesian
        output_file = get_model_results_path(f"{time_suffix}_year_{theta_value}b_metrics.csv")
    
    results_df.to_csv(output_file, index=False)
    
    elapsed = time.time() - start_time
    print(f"Results saved to: {output_file}")
    print(f"Processing time: {elapsed/60:.1f} minutes")
    print(f"Summary statistics:")
    if len(results_df) > 0:
        numeric_cols = ['rmse_train', 'rmse_test', 'mase_test', 'bias_test']
        print(results_df[numeric_cols].describe().round(4))
    
    return results_df

def run_time_period_models(time_period, n_months):
    """Run all models for a specific time period"""
    
    print(f"\n{'='*80}")
    print(f"PROCESSING {time_period}-YEAR MODELS ({n_months} months)")
    print("="*80)
    
    # Check if data exists
    data_dir = get_simulated_data_path(f"d{time_period}year")
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} not found!")
        print("Please run 5_simulation_temp.py first to generate simulated datasets.")
        return
    
    # Parameters
    theta_values = [1.5, 5, 10, 100]
    max_workers = min(mp.cpu_count(), 4)
    
    # Run Standard Negative Binomial models
    print(f"\n--- Standard NB Models ({time_period} years) ---")
    for theta in theta_values:
        try:
            run_models_on_theta(
                theta_value=theta,
                model_func=negbinner,
                model_name='NB',
                time_period=time_period,
                n_months=n_months,
                max_workers=max_workers
            )
        except Exception as e:
            print(f"Error processing NB models for {time_period}yr, theta {theta}: {e}")
    
    # Run Bayesian models (with reduced parallelism)
    print(f"\n--- Bayesian NB Models ({time_period} years) ---")
    for theta in theta_values:
        try:
            run_models_on_theta(
                theta_value=theta,
                model_func=stanbinner,
                model_name='Bayesian',
                time_period=time_period,
                n_months=n_months,
                max_workers=1  # Sequential for Bayesian models
            )
        except Exception as e:
            print(f"Error processing Bayesian models for {time_period}yr, theta {theta}: {e}")

def main():
    """Main function to run all model evaluations"""
    
    print("="*80)
    print("COMPREHENSIVE MODEL EVALUATION PIPELINE")
    print("Python version of 7models_*.R scripts")
    print("="*80)
    
    # Set seed EXACTLY like R scripts
    # R code: set.seed(76568)
    np.random.seed(76568)
    
    # Time periods and corresponding months
    time_periods = {
        5: 60,    # 5 years = 60 months
        10: 120,  # 10 years = 120 months
        20: 240,  # 20 years = 240 months
        30: 360   # 30 years = 360 months
    }
    
    # Check if simulated data exists
    if not os.path.exists(get_simulated_data_path()):
        print("Error: Simulated data directory not found!")
        print("Please run 5_simulation_temp.py first to generate simulated datasets.")
        return
    
    print(f"Available CPU cores: {mp.cpu_count()}")
    print(f"Time periods to process: {list(time_periods.keys())} years")
    print(f"Theta values: [1.5, 5, 10, 100]")
    
    total_start_time = time.time()
    
    # Process each time period
    for time_period, n_months in time_periods.items():
        try:
            run_time_period_models(time_period, n_months)
        except Exception as e:
            print(f"Error processing {time_period}-year models: {e}")
            continue
    
    # Final summary
    total_elapsed = time.time() - total_start_time
    print(f"\nTotal processing time: {total_elapsed/60:.1f} minutes")
    print(f"Results saved in: {get_model_results_path('')}")

if __name__ == "__main__":
    main() 