"""
Script to apply negbinner and stanbinner models to simulated datasets
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from model_functions import negbinner, stanbinner


# --------------------------
# CONFIGURATION
# --------------------------

# Time periods and corresponding months
time_periods = {
    5: 60,    # 5 years = 60 months
    10: 120,  # 10 years = 120 months
    20: 240,  # 20 years = 240 months
    30: 360   # 30 years = 360 months
}

# TODO: Set model parameters
theta = 1.5 # theta_values = [1.5, 5, 10, 100]
years = 5
n = time_periods[years]

# Set folder path accordingly
folder_path = f"our_output/simulated_data/d{years}year/theta_{theta}"


# --------------------------
# SCRIPT START
# --------------------------

# Set seed for reproducibility
np.random.seed(76568)

# Prepare lists to collect results
nb_rows = []
bayes_rows = []

def extract_number(filename):
    match = re.search(r"_(\d+)\.csv$", filename)
    return int(match.group(1)) if match else float('inf')

csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
csv_files.sort(key=extract_number)
csv_files = csv_files[:3]

print(f"Found {len(csv_files)} datasets in: {folder_path}")

for file in tqdm(csv_files, desc="Processing files"):
    file_path = os.path.join(folder_path, file)
    dataset_id = os.path.splitext(file)[0]
    
    print(f"\nProcessing: {dataset_id}")
    data = pd.read_csv(file_path)
    
    nb_result = negbinner(data, theta=theta, n=n)
    bayes_result = stanbinner(data, theta=theta, n=n)
    
    # Store results as dict with dataset_id as row label
    nb_rows.append(pd.Series(nb_result, name=dataset_id))
    bayes_rows.append(pd.Series(bayes_result, name=dataset_id))

# Combine into separate DataFrames
nb_results_df = pd.DataFrame(nb_rows)
bayes_results_df = pd.DataFrame(bayes_rows)

# Show results
print("\nStandard NB Model Results:")
print(nb_results_df)

print("\nBayesian NB Model Results:")
print(bayes_results_df)

# output folder
output_dir = "our_output/model_results"

# Format theta nicely (e.g., 1.5 → "1.5", 10 → "10")
theta_str = str(theta)

# Build filenames
nb_filename = f"{output_dir}/d{years}year_theta_{theta_str}_nb_metrics.csv"
bnb_filename = f"{output_dir}/d{years}year_theta_{theta_str}_bnb_metrics.csv"

# Save to CSV
nb_results_df.to_csv(nb_filename)
bayes_results_df.to_csv(bnb_filename)

print(f"\nSaved Standard NB results to: {nb_filename}")
print(f"Saved Bayesian NB results to: {bnb_filename}")

    