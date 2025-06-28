#!/usr/bin/env python3
"""
Script to apply negbinner and stanbinner models to a dataset
"""

import pandas as pd
from model_functions import negbinner, stanbinner

# Define path to the input dataset
data_path = "our_output/simulated_data/d5year/theta_1.5/dyear5_1.5_1.csv"

# Load dataset
data = pd.read_csv(data_path)
print(f"Loaded data from {data_path}. Shape: {data.shape}")

# Run standard negative binomial model
print("\nRunning Standard Negative Binomial Model...")
nb_result = negbinner(data, theta=1.5, n=60)
print("Standard Model Results:")
for k, v in nb_result.items():
    print(f"{k}: {v:.4f}")

# Run Bayesian negative binomial model
print("\nRunning Bayesian Negative Binomial Model...")
bayes_result = stanbinner(data, theta=1.5, n=60)
print("Bayesian Model Results:")
for k, v in bayes_result.items():
    print(f"{k}: {v:.4f}")
