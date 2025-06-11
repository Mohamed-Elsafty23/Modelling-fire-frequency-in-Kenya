from output_utils import get_output_path, get_model_results_path, get_simulated_data_path, ensure_output_dirs
#!/usr/bin/env python3
"""
5-year model evaluation - Python version of 7models_5years.R
Runs standard and Bayesian NB models on 5-year simulated datasets
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

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

def run_models_on_theta(theta_value, model_func, model_name, n_months=60, max_workers=4):
    """Run models on all datasets for a specific theta value"""
    
    print(f"\n{'='*60}")
    print(f"Running {model_name} models for theta = {theta_value}")
    print('='*60)
    
    # Get file paths
    data_path = get_simulated_data_path(f"d5year/theta_{theta_value}")
    file_pattern = f"{data_path}/*.csv"
    file_paths = glob.glob(file_pattern)
    
    if not file_paths:
        print(f"No data files found in {data_path}")
        return pd.DataFrame()
    
    print(f"Found {len(file_paths)} datasets to process")
    
    # Process files in parallel
    results = []
    
    if max_workers > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_single_file, file_path, model_func, theta_value, n_months): file_path
                for file_path in file_paths
            }
            
            # Collect results
            for i, future in enumerate(as_completed(future_to_file)):
                result = future.result()
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(file_paths)} files")
    else:
        # Sequential processing
        for i, file_path in enumerate(file_paths):
            result = process_single_file(file_path, model_func, theta_value, n_months)
            results.append(result)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(file_paths)} files")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    os.makedirs(get_model_results_path(""), exist_ok=True)
    
    if model_name.lower() == 'nb':
        output_file = get_model_results_path(f"five_year_{theta_value}_metrics.csv")
    else:  # Bayesian
        output_file = get_model_results_path(f"five_year_{theta_value}b_metrics.csv")
    
    results_df.to_csv(output_file, index=False)
    
    print(f"Results saved to: {output_file}")
    print(f"Summary statistics:")
    print(results_df[['rmse_train', 'rmse_test', 'mase_test', 'bias_test']].describe())
    
    return results_df

def main():
    """Main function to run all 5-year model evaluations"""
    
    print("Starting 5-Year Model Evaluation")
    print("="*50)
    
    # Check if simulated data exists
    if not os.path.exists(get_simulated_data_path("d5year")):
        print("Error: Simulated data not found!")
        print("Please run 5_simulation_temp.py first to generate simulated datasets.")
        return
    
    # Parameters
    theta_values = [1.5, 5, 10, 100]
    n_months = 60  # 5 years * 12 months
    
    # Determine number of workers
    max_workers = min(mp.cpu_count(), 4)  # Use max 4 cores
    print(f"Using {max_workers} parallel workers")
    
    # Run Standard Negative Binomial models
    print("\n" + "="*70)
    print("RUNNING STANDARD NEGATIVE BINOMIAL MODELS")
    print("="*70)
    
    nb_results = {}
    for theta in theta_values:
        try:
            results_df = run_models_on_theta(
                theta_value=theta,
                model_func=negbinner,
                model_name='NB',
                n_months=n_months,
                max_workers=max_workers
            )
            nb_results[theta] = results_df
            
        except Exception as e:
            print(f"Error processing NB models for theta {theta}: {e}")
    
    # Run Bayesian Negative Binomial models
    print("\n" + "="*70)
    print("RUNNING BAYESIAN NEGATIVE BINOMIAL MODELS")
    print("="*70)
    
    bayesian_results = {}
    for theta in theta_values:
        try:
            results_df = run_models_on_theta(
                theta_value=theta,
                model_func=stanbinner,
                model_name='Bayesian',
                n_months=n_months,
                max_workers=1  # Bayesian models are computationally heavy, use less parallelism
            )
            bayesian_results[theta] = results_df
            
        except Exception as e:
            print(f"Error processing Bayesian models for theta {theta}: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("5-YEAR MODEL EVALUATION COMPLETED")
    print("="*70)
    
    print("\nGenerated results files:")
    result_files = glob.glob(get_model_results_path("five_year_*_metrics.csv"))
    for file in sorted(result_files):
        print(f"  - {file}")
    
    print(f"\nTotal result files: {len(result_files)}")
    print("Ready for analysis and visualization!")

if __name__ == "__main__":
    main() 