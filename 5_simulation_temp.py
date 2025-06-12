from output_utils import get_output_path, get_model_results_path, get_simulated_data_path, ensure_output_dirs
#!/usr/bin/env python3
"""
Data simulation script - Python version of 5simulation_temp.R
Generates synthetic climate-fire datasets for model evaluation
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import gamma, norm, truncnorm
from sklearn.linear_model import LinearRegression
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FireClimateSimulator:
    """Class to simulate fire-climate data based on real data patterns"""
    
    def __init__(self, data_file="fire_data_2000-18.csv"):
        """Initialize with real data to fit distributions"""
        try:
            self.real_data = pd.read_csv(data_file)
            self.fit_distributions()
            self.fit_relationships()
        except FileNotFoundError:
            print(f"Error: {data_file} not found. Run 2_data_aggregate.py first.")
            raise
    
    def fit_distributions(self):
        """Fit statistical distributions to real data"""
        data = self.real_data.dropna()
        
        # Fit gamma distribution to rainfall
        rainfall_clean = data['mean_rainfall'][data['mean_rainfall'] > 0]
        if len(rainfall_clean) > 0:
            self.rainfall_gamma = stats.gamma.fit(rainfall_clean)
        else:
            self.rainfall_gamma = (2.75, 0, 30.5)  # Default parameters
        
        # Fit normal distribution to max temperature
        temp_clean = data['mean_max_temp'].dropna()
        if len(temp_clean) > 0:
            self.temp_params = {
                'mu': temp_clean.mean(),
                'sigma': temp_clean.std(),
                'min': temp_clean.min(),
                'max': temp_clean.max()
            }
        else:
            self.temp_params = {'mu': 29.18, 'sigma': 2.29, 'min': 23.44, 'max': 34.82}
        
        print(f"Fitted rainfall gamma parameters: shape={self.rainfall_gamma[0]:.3f}, "
              f"scale={self.rainfall_gamma[2]:.3f}")
        print(f"Temperature parameters: mu={self.temp_params['mu']:.3f}, "
              f"sigma={self.temp_params['sigma']:.3f}")
    
    def fit_relationships(self):
        """Fit relationships between variables"""
        data = self.real_data.dropna()
        
        # Fit linear relationship between max_temp and rainfall
        X = data[['mean_max_temp']].values
        y = data['mean_rainfall'].values
        
        self.rainfall_model = LinearRegression()
        self.rainfall_model.fit(X, y)
        
        # Calculate residuals for error simulation
        y_pred = self.rainfall_model.predict(X)
        self.rainfall_residuals = y - y_pred
        
        print(f"Rainfall-temperature relationship: "
              f"slope={self.rainfall_model.coef_[0]:.3f}, "
              f"intercept={self.rainfall_model.intercept_:.3f}")
    
    def simulate_dataset(self, n_months=60, theta=1.5, seed=None):
        """
        Simulate a single dataset
        
        Parameters:
        - n_months: Number of months to simulate
        - theta: Dispersion parameter for negative binomial
        - seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Simulate max temperature using truncated normal
        a = (self.temp_params['min'] - self.temp_params['mu']) / self.temp_params['sigma']
        b = (self.temp_params['max'] - self.temp_params['mu']) / self.temp_params['sigma']
        
        max_temp = truncnorm.rvs(a, b, 
                                loc=self.temp_params['mu'], 
                                scale=self.temp_params['sigma'], 
                                size=n_months)
        
        # Simulate rainfall based on temperature relationship + random error
        rainfall_pred = self.rainfall_model.predict(max_temp.reshape(-1, 1))
        
        # Add random errors
        if len(self.rainfall_residuals) > 0:
            error_min, error_max = self.rainfall_residuals.min(), self.rainfall_residuals.max()
            errors = np.random.uniform(error_min, error_max, n_months)
        else:
            errors = np.random.normal(0, 10, n_months)  # Default error
        
        rainfall = rainfall_pred + errors
        rainfall = np.maximum(rainfall, 0)  # Ensure non-negative
        
        # Create time index
        time_idx = np.arange(1, n_months + 1)
        
        # Add seasonal components
        cos_term = np.cos(2 * np.pi * time_idx / 12 + np.random.normal(0, 0.1))
        sin_term = np.sin(2 * np.pi * time_idx / 12 + np.random.normal(0, 0.1))
        
        # Simulate fire count using negative binomial with seasonal effects
        # Log-linear model: log(mu) = intercept + b1*temp + b2*rain + b3*cos + b4*sin
        log_mu = (5.6 +  # Base fire rate (log scale)
                 0.05 * max_temp +  # Temperature effect
                 -0.01 * rainfall +  # Rainfall effect (negative)
                 0.3 * cos_term +   # Seasonal cosine
                 0.2 * sin_term)    # Seasonal sine
        
        mu = np.exp(log_mu)
        
        # Add noise to predicted values
        if theta > 0:
            # Convert theta to n, p parameterization for negative binomial
            p = theta / (theta + mu)
            n = theta
            
            # Simulate counts
            count = np.random.negative_binomial(n, p)
            
            # Add additional residual errors
            residual_errors = np.random.uniform(-2, 2, n_months)
            count = np.maximum(np.round(mu + residual_errors), 0).astype(int)
        else:
            count = np.random.poisson(mu)
        
        # Create dataset
        dataset = pd.DataFrame({
            'max_temp': max_temp,
            'rainfall': rainfall,
            'count': count,
            'tyme': time_idx
        })
        
        return dataset
    
    def generate_multiple_datasets(self, n_datasets=1000, n_months=60, theta=1.5, 
                                 output_dir=get_simulated_data_path(), subdir_name=None):
        """Generate multiple datasets and save them"""
        
        if subdir_name is None:
            years = n_months // 12
            subdir_name = f"d{years}year/theta_{theta}"
        
        output_path = Path(output_dir) / subdir_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating {n_datasets} datasets with {n_months} months each...")
        print(f"Theta parameter: {theta}")
        print(f"Output directory: {output_path}")
        
        for i in range(n_datasets):
            dataset = self.simulate_dataset(n_months=n_months, theta=theta, seed=76568+i)
            
            filename = f"dyear{n_months//12}_{theta}_{i+1}.csv"
            filepath = output_path / filename
            dataset.to_csv(filepath, index=False)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i+1}/{n_datasets} datasets")
        
        print(f"Completed generating datasets in {output_path}")

def run_all_simulations():
    """Run all simulation scenarios from the original R script"""
    
    # Initialize simulator
    simulator = FireClimateSimulator()
    
    # Simulation parameters
    theta_values = [1.5, 5, 10, 100]
    time_periods = {
        5: 60,    # 5 years = 60 months
        10: 120,  # 10 years = 120 months
        20: 240,  # 20 years = 240 months
        30: 360   # 30 years = 360 months
    }
    
    print("Starting comprehensive simulation...")
    print(f"Theta values: {theta_values}")
    print(f"Time periods: {list(time_periods.keys())} years")
    
    for theta in theta_values:
        print(f"\n{'='*50}")
        print(f"Simulating with theta = {theta}")
        print('='*50)
        
        for years, months in time_periods.items():
            print(f"\nGenerating {years}-year datasets ({months} months)...")
            
            simulator.generate_multiple_datasets(
                n_datasets=1000,
                n_months=months,
                theta=theta,
                subdir_name=f"d{years}year/theta_{theta}"
            )
    
    print("\n" + "="*50)
    print("ALL SIMULATIONS COMPLETED!")
    print("="*50)

if __name__ == "__main__":
    # Create base directories
    os.makedirs(get_simulated_data_path(), exist_ok=True)
    
    # Run all simulations
    run_all_simulations()
    
    print("\nSimulation script completed!") 