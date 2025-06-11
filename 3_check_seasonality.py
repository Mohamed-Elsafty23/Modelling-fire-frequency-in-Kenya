#!/usr/bin/env python3
"""
Seasonality analysis script - Python version of 3check_seasonality.R
Checks for seasonal patterns in climate data
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from output_utils import get_output_path, ensure_output_dirs

def check_seasonality():
    """Check for seasonality in climate time series data"""
    
    print("Reading fire-climate data...")
    
    # Ensure output directory exists
    ensure_output_dirs()
    
    try:
        series_data = pd.read_csv(get_output_path("fire_data_2000-18.csv"))
    except FileNotFoundError:
        print("Error: fire_data_2000-18.csv not found. Run 2_data_aggregate.py first.")
        return
    
    print(f"Data shape: {series_data.shape}")
    print(series_data.head())
    
    # Create time index
    series_data['date'] = pd.to_datetime(series_data[['year', 'month']].assign(day=1))
    series_data = series_data.set_index('date').sort_index()
    
    # Variables to check for seasonality
    variables = {
        'mean_max_temp': 'Maximum Temperature',
        'mean_min_temp': 'Minimum Temperature', 
        'mean_rainfall': 'Rainfall'
    }
    
    seasonality_results = {}
    
    print("\n" + "="*50)
    print("SEASONALITY ANALYSIS RESULTS")
    print("="*50)
    
    for var, label in variables.items():
        print(f"\nAnalyzing {label} ({var})...")
        
        # Extract time series
        ts = series_data[var].dropna()
        
        if len(ts) < 24:  # Need at least 2 years for seasonal analysis
            print(f"Insufficient data for {var}")
            continue
        
        # Method 1: Seasonal decomposition
        try:
            decomposition = seasonal_decompose(ts, model='additive', period=12)
            seasonal_component = decomposition.seasonal
            
            # Test if seasonal component is significantly different from zero
            seasonal_var = np.var(seasonal_component.dropna())
            residual_var = np.var(decomposition.resid.dropna())
            
            # F-test for seasonality
            if residual_var > 0:
                f_stat = seasonal_var / residual_var
                seasonal_significant = f_stat > 1.5  # Simple threshold
            else:
                seasonal_significant = False
                
        except Exception as e:
            print(f"Error in seasonal decomposition for {var}: {e}")
            seasonal_significant = False
        
        # Method 2: Monthly variance analysis
        monthly_means = ts.groupby(ts.index.month).mean()
        monthly_var = np.var(monthly_means)
        overall_var = np.var(ts)
        
        # High monthly variance relative to overall suggests seasonality
        variance_ratio = monthly_var / overall_var if overall_var > 0 else 0
        
        # Method 3: Autocorrelation at lag 12
        try:
            from statsmodels.tsa.stattools import acf
            autocorr = acf(ts, nlags=12, fft=True)
            lag12_autocorr = abs(autocorr[12]) if len(autocorr) > 12 else 0
        except:
            lag12_autocorr = 0
        
        # Combine evidence for seasonality
        seasonality_evidence = {
            'seasonal_decomp': seasonal_significant,
            'variance_ratio': variance_ratio > 0.1,  # Arbitrary threshold
            'lag12_autocorr': lag12_autocorr > 0.3   # Moderate autocorrelation
        }
        
        # Overall seasonality decision
        seasonality_score = sum(seasonality_evidence.values())
        has_seasonality = seasonality_score >= 2  # Majority evidence
        
        seasonality_results[var] = {
            'has_seasonality': has_seasonality,
            'evidence': seasonality_evidence,
            'variance_ratio': variance_ratio,
            'lag12_autocorr': lag12_autocorr
        }
        
        # Print results
        print(f"  Seasonality detected: {'YES' if has_seasonality else 'NO'}")
        print(f"  Evidence score: {seasonality_score}/3")
        print(f"  - Seasonal decomposition: {'✓' if seasonal_significant else '✗'}")
        print(f"  - Monthly variance ratio: {variance_ratio:.3f} {'✓' if variance_ratio > 0.1 else '✗'}")
        print(f"  - 12-month autocorr: {lag12_autocorr:.3f} {'✓' if lag12_autocorr > 0.3 else '✗'}")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    seasonal_vars = [var for var, result in seasonality_results.items() if result['has_seasonality']]
    
    if seasonal_vars:
        print(f"Variables with seasonality: {', '.join(seasonal_vars)}")
        print("Recommendation: Include seasonal terms in models")
    else:
        print("No clear seasonal patterns detected")
        print("Recommendation: Seasonal terms may not be necessary")
    
    # Create visualization
    create_seasonality_plots(series_data, variables, seasonality_results)
    
    return seasonality_results

def create_seasonality_plots(data, variables, results):
    """Create plots to visualize seasonality"""
    
    # Adjust subplot layout based on number of variables
    n_vars = len(variables)
    if n_vars <= 2:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    fig.suptitle('Seasonality Analysis', fontsize=16)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Time series plots
    for i, (var, label) in enumerate(variables.items()):
        ax = axes_flat[i]
        
        ts = data[var].dropna()
        ax.plot(ts.index, ts.values, linewidth=1, alpha=0.7)
        ax.set_title(f'{label}\nSeasonality: {"YES" if results[var]["has_seasonality"] else "NO"}')
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    
    # Monthly box plots - use the next available subplot
    ax = axes_flat[n_vars]
    
    # Prepare data for box plot
    monthly_data = []
    for var in variables.keys():
        ts = data[var].dropna()
        monthly = ts.groupby(ts.index.month).apply(list).to_dict()
        for month, values in monthly.items():
            for value in values:
                monthly_data.append({'Month': month, 'Variable': var, 'Value': value})
    
    if monthly_data:
        monthly_df = pd.DataFrame(monthly_data)
        
        # Normalize values for comparison
        for var in variables.keys():
            var_data = monthly_df[monthly_df['Variable'] == var]['Value']
            if len(var_data) > 0:
                monthly_df.loc[monthly_df['Variable'] == var, 'Value'] = (
                    (var_data - var_data.mean()) / var_data.std()
                )
        
        sns.boxplot(data=monthly_df, x='Month', y='Value', hue='Variable', ax=ax)
        ax.set_title('Monthly Patterns (Normalized)')
        ax.set_ylabel('Normalized Value')
    
    # Hide any unused subplots
    for i in range(n_vars + 1, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    output_file = get_output_path('seasonality_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSeasonality plots saved as: {output_file}")
    plt.show()

if __name__ == "__main__":
    results = check_seasonality()
    print("\nSeasonality analysis completed!") 