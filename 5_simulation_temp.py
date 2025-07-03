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
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
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
            # Note: fit_relationships is called later with specific theta parameter
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
    
    def fit_relationships(self, theta=1.5):
        """Fit relationships between variables - matches R code exactly"""
        # Select and rename columns to match R code
        # CHANGED: added month
        data = self.real_data[['mean_max_temp', 'mean_rainfall', 'count', 'month']].copy()
        data.columns = ['max_temp', 'rainfall', 'count', 'month']
        data = data.dropna()
        
        # Add tyme column (time index) - matches R code
        # CHANGED: data['tyme'] = range(1, len(data) + 1)

        
        # Fit linear relationship between max_temp and rainfall (for rainfall simulation)
        X = data[['max_temp']].values
        y = data['rainfall'].values
        
        self.rainfall_model = LinearRegression()
        self.rainfall_model.fit(X, y)
        
        # Calculate residuals for error simulation
        y_pred = self.rainfall_model.predict(X)
        self.rainfall_residuals = np.round(y - y_pred, 5) # 5 signifigant figures
        
        print(f"Rainfall-temperature relationship: "
              f"slope={self.rainfall_model.coef_[0]:.3f}, "
              f"intercept={self.rainfall_model.intercept_:.3f}")
        
        # Fit negative binomial GLM for fire count - EXACTLY like R code
        print("Fitting negative binomial GLM for fire count...")
        
        # Create seasonal terms exactly like R code: sin((2*12*pi/tyme) + rnorm(1,sd=0.1))
        # Note: This matches the R formula exactly, including the unusual (2*12*pi/tyme) form
        np.random.seed(42)  # For reproducible seasonal term noise
        sin_noise = np.random.normal(0, 0.1)
        cos_noise = np.random.normal(0, 0.1)
        
        # CHANGED tyme to month & corrected seasonality formula
        data['sin_term'] = np.sin((2 * np.pi * data['month'] / 12) + sin_noise)
        data['cos_term'] = np.cos((2 * np.pi * data['month'] / 12) + cos_noise)
        
        # Prepare design matrix for GLM
        X_glm = data[['max_temp', 'rainfall', 'sin_term', 'cos_term']].copy()
        X_glm = sm.add_constant(X_glm)  # Add intercept
        y_glm = data['count']
        
        # Fit negative binomial GLM with log link (matching R's glm.nb)
        # CHANGED: self.fire_model = sm.GLM(y_glm, X_glm, family=sm.families.NegativeBinomial()).fit()
        self.fire_model = sm.GLM(y_glm, X_glm, family=sm.families.NegativeBinomial(alpha=1/theta)).fit()
        
        # Store coefficients
        self.fire_coefficients = {
            'intercept': self.fire_model.params['const'],
            'max_temp': self.fire_model.params['max_temp'],
            'rainfall': self.fire_model.params['rainfall'],
            'sin_term': self.fire_model.params['sin_term'],
            'cos_term': self.fire_model.params['cos_term']
        }
        
        # Store noise values used for seasonal terms
        self.sin_noise = sin_noise
        self.cos_noise = cos_noise
        
        print("Fire model coefficients:")
        for name, coef in self.fire_coefficients.items():
            print(f"  {name}: {coef:.6f}")
    
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
        
        # Fit the fire model with this specific theta (like R code does)
        if not hasattr(self, 'current_theta') or self.current_theta != theta:
            print(f"Fitting fire model with theta = {theta}")
            self.fit_relationships(theta=theta)
            self.current_theta = theta
        
        # Simulate max temperature using truncated normal
        a = (self.temp_params['min'] - self.temp_params['mu']) / self.temp_params['sigma']
        b = (self.temp_params['max'] - self.temp_params['mu']) / self.temp_params['sigma']
        
        max_temp = truncnorm.rvs(a, b, 
                                loc=self.temp_params['mu'], 
                                scale=self.temp_params['sigma'], 
                                size=n_months)
        
        # Simulate rainfall based on temperature relationship + random error
        rainfall_pred = self.rainfall_model.predict(max_temp.reshape(-1, 1))
        
        # CHANGED this section
        # Adding residual errors
        # Set bounds for binning residuals
        error_min, error_max = self.rainfall_residuals.min(), self.rainfall_residuals.max()
        bounds = np.arange(error_min, error_max + 0.2, 0.2)

        # Compute relative frequency of residuals across bins
        resid_binned = pd.cut(self.rainfall_residuals, bins=bounds)
        error_distribution = resid_binned.value_counts().sort_index()

        # Generate uniform random errors over the residual range
        errors = np.random.uniform(low=error_min, high=error_max, size=len(rainfall_pred))
        rainfall = rainfall_pred + errors
        rainfall = np.maximum(rainfall, 0)  # Ensure non-negative
        
        # Create time index (tyme column like R code)
        time_idx = np.arange(1, n_months + 1)
        month = ((time_idx - 1) % 12) + 1  # CHANGED: Generate months 1–12 cyclically
        
        # Add corrected seasonal components (vs. R code: sin((2*12*pi/tyme) + rnorm(1,sd=0.1))
        # Use the same noise values that were used during model fitting
        # CHANGED time_idx to month
        sin_term = np.sin((2 * np.pi * month / 12) + self.sin_noise)
        cos_term = np.cos((2  * np.pi * month / 12) + self.cos_noise)
        
        # Simulate fire count using fitted coefficients from negative binomial GLM
        # Log-linear model: log(mu) = intercept + b1*temp + b2*rain + b3*sin + b4*cos
        log_mu = (self.fire_coefficients['intercept'] +
                 self.fire_coefficients['max_temp'] * max_temp +
                 self.fire_coefficients['rainfall'] * rainfall +
                 self.fire_coefficients['sin_term'] * sin_term +
                 self.fire_coefficients['cos_term'] * cos_term)
        
    
        
        # Get predicted values (like R's predict(fm, dataclean, type = "response"))
        mu = np.exp(log_mu)
        
        # Add residual errors exactly like R code
        # R code: simulated_errors2 <- runif(length(y_count2), min = min(resid2), max = max(resid2))
        # Use residuals from fitted model
        resid_min = self.fire_model.resid_response.min()
        resid_max = self.fire_model.resid_response.max()
            
        # Generate uniform errors within residual range
        residual_errors = np.random.uniform(resid_min, resid_max, n_months)
        
        # Add errors to predicted values and round (matching R code exactly)
        simulated_values = mu + residual_errors
        count = np.maximum(np.round(simulated_values), 0).astype(int)
        
        # Create dataset
        dataset = pd.DataFrame({
            'max_temp': max_temp,
            'rainfall': rainfall,
            'count': count,
            'tyme': time_idx,
            'month': month # CHANGED: added month
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
                n_datasets=100,
                # n_datasets=1000,
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