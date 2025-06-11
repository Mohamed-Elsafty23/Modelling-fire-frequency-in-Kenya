from output_utils import get_output_path, get_model_results_path, get_simulated_data_path, ensure_output_dirs
#!/usr/bin/env python3
"""
Descriptive statistics and visualization - Python version of 8descriptive_stats.R
Analyzes real fire-climate data and creates publication-quality visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load and prepare the fire-climate data"""
    try:
        fire_clim = pd.read_csv(get_output_path("fire_data_2000-18.csv"))
    except FileNotFoundError:
        print("Error: fire_data_2000-18.csv not found!")
        print("Please run 2_data_aggregate.py first.")
        return None
    
    # Create time index
    fire_clim['time'] = pd.to_datetime(fire_clim[['year', 'month']].assign(day=1))
    fire_clim = fire_clim.sort_values('time').reset_index(drop=True)
    
    # Create quarter variable
    fire_clim['quarter'] = fire_clim['month'].apply(lambda x: 
        'Q1' if x <= 3 else 'Q2' if x <= 6 else 'Q3' if x <= 9 else 'Q4')
    
    print(f"Data loaded: {len(fire_clim)} records from {fire_clim['year'].min()} to {fire_clim['year'].max()}")
    
    return fire_clim

def generate_summary_statistics(data):
    """Generate summary statistics"""
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    # Select key variables
    key_vars = ['count', 'mean_max_temp', 'mean_min_temp', 'mean_rainfall']
    summary_stats = data[key_vars].describe()
    
    print(summary_stats.round(3))
    
    # Additional statistics
    print(f"\nAdditional Statistics:")
    print(f"Total fire events: {data['count'].sum():,}")
    print(f"Mean fires per month: {data['count'].mean():.1f}")
    print(f"Peak fire month: {data.loc[data['count'].idxmax(), 'month']} " +
          f"({data.loc[data['count'].idxmax(), 'year']})")
    print(f"Temperature range: {data['mean_min_temp'].min():.1f}°C to {data['mean_max_temp'].max():.1f}°C")
    print(f"Rainfall range: {data['mean_rainfall'].min():.1f}mm to {data['mean_rainfall'].max():.1f}mm")
    
    return summary_stats

def create_time_series_plots(data):
    """Create time series plots for all variables"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Monthly Time Series Trends in Kenya (2000-2018)', fontsize=16, fontweight='bold')
    
    # Fire count
    axes[0,0].plot(data['time'], data['count'], color='red', linewidth=1.2, alpha=0.8)
    axes[0,0].set_title('a. Monthly Fire Frequency Trend\nfrom 2000 to 2018 in Kenya', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Number of Fires', fontsize=12, fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Max temperature
    axes[0,1].plot(data['time'], data['mean_max_temp'], color='brown', linewidth=1.2, alpha=0.8)
    axes[0,1].set_title('b. Monthly Maximum Temperature Trend\nfrom 2000 to 2018 in Kenya', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('Max Temperature (°C)', fontsize=12, fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Min temperature
    axes[1,0].plot(data['time'], data['mean_min_temp'], color='blue', linewidth=1.2, alpha=0.8)
    axes[1,0].set_title('c. Monthly Minimum Temperature Trend\nfrom 2000 to 2018 in Kenya', fontsize=14, fontweight='bold')
    axes[1,0].set_ylabel('Min Temperature (°C)', fontsize=12, fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Rainfall
    axes[1,1].plot(data['time'], data['mean_rainfall'], color='orange', linewidth=1.2, alpha=0.8)
    axes[1,1].set_title('d. Monthly Rainfall Trend\nfrom 2000 to 2018 in Kenya', fontsize=14, fontweight='bold')
    axes[1,1].set_ylabel('Rainfall (mm)', fontsize=12, fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(get_output_path('time_series_plots.png'), dpi=300, bbox_inches='tight')
    print("Time series plots saved as: time_series_plots.png")
    
    return fig

def create_monthly_patterns(data):
    """Create monthly pattern visualizations"""
    
    # Monthly averages
    monthly_avg = data.groupby('month').agg({
        'count': 'mean',
        'mean_max_temp': 'mean',
        'mean_min_temp': 'mean',
        'mean_rainfall': 'mean'
    }).reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Monthly Patterns (2000-2018 Averages)', fontsize=16, fontweight='bold')
    
    # Fire count bars
    axes[0,0].bar(monthly_avg['month'], monthly_avg['count'], color='red', alpha=0.7)
    axes[0,0].set_title('Average Monthly Fire Frequency', fontsize=12, fontweight='bold')
    axes[0,0].set_ylabel('Number of Fires', fontsize=11)
    axes[0,0].set_xlabel('Month', fontsize=11)
    axes[0,0].set_xticks(range(1, 13))
    
    # Temperature patterns
    axes[0,1].plot(monthly_avg['month'], monthly_avg['mean_max_temp'], 'o-', color='red', label='Max Temp', linewidth=2)
    axes[0,1].plot(monthly_avg['month'], monthly_avg['mean_min_temp'], 'o-', color='blue', label='Min Temp', linewidth=2)
    axes[0,1].set_title('Average Monthly Temperatures', fontsize=12, fontweight='bold')
    axes[0,1].set_ylabel('Temperature (°C)', fontsize=11)
    axes[0,1].set_xlabel('Month', fontsize=11)
    axes[0,1].legend()
    axes[0,1].set_xticks(range(1, 13))
    axes[0,1].grid(True, alpha=0.3)
    
    # Rainfall pattern
    axes[1,0].bar(monthly_avg['month'], monthly_avg['mean_rainfall'], color='blue', alpha=0.7)
    axes[1,0].set_title('Average Monthly Rainfall', fontsize=12, fontweight='bold')
    axes[1,0].set_ylabel('Rainfall (mm)', fontsize=11)
    axes[1,0].set_xlabel('Month', fontsize=11)
    axes[1,0].set_xticks(range(1, 13))
    
    # Box plots for seasonal variation
    months_long = []
    for _, row in data.iterrows():
        months_long.append({'Month': row['month'], 'Fires': row['count']})
    months_df = pd.DataFrame(months_long)
    
    sns.boxplot(data=months_df, x='Month', y='Fires', ax=axes[1,1])
    axes[1,1].set_title('Monthly Fire Frequency Distribution', fontsize=12, fontweight='bold')
    axes[1,1].set_ylabel('Number of Fires', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(get_output_path('monthly_patterns.png'), dpi=300, bbox_inches='tight')
    print("Monthly patterns saved as: monthly_patterns.png")
    
    return monthly_avg

def create_correlation_plots(data):
    """Create correlation plots between variables"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Fire Frequency vs Climate Variables', fontsize=16, fontweight='bold')
    
    # Fire vs Max Temperature by quarter
    for i, quarter in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        quarter_data = data[data['quarter'] == quarter]
        axes[0,0].scatter(quarter_data['mean_max_temp'], np.log(quarter_data['count'] + 1), 
                         alpha=0.6, label=quarter, s=30)
    
    # Add regression line for all data
    x = data['mean_max_temp'].values
    y = np.log(data['count'] + 1).values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    axes[0,0].plot(x, p(x), 'k--', alpha=0.8, linewidth=2)
    
    axes[0,0].set_xlabel('Maximum Temperature (°C)', fontsize=11)
    axes[0,0].set_ylabel('Log(Fire Frequency)', fontsize=11)
    axes[0,0].set_title('Fire vs Maximum Temperature', fontsize=12, fontweight='bold')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Fire vs Min Temperature by quarter
    for i, quarter in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        quarter_data = data[data['quarter'] == quarter]
        axes[0,1].scatter(quarter_data['mean_min_temp'], np.log(quarter_data['count'] + 1), 
                         alpha=0.6, label=quarter, s=30)
    
    x = data['mean_min_temp'].values
    y = np.log(data['count'] + 1).values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    axes[0,1].plot(x, p(x), 'k--', alpha=0.8, linewidth=2)
    
    axes[0,1].set_xlabel('Minimum Temperature (°C)', fontsize=11)
    axes[0,1].set_ylabel('Log(Fire Frequency)', fontsize=11)
    axes[0,1].set_title('Fire vs Minimum Temperature', fontsize=12, fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Fire vs Rainfall by quarter
    for i, quarter in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        quarter_data = data[data['quarter'] == quarter]
        axes[1,0].scatter(quarter_data['mean_rainfall'], np.log(quarter_data['count'] + 1), 
                         alpha=0.6, label=quarter, s=30)
    
    x = data['mean_rainfall'].values
    y = np.log(data['count'] + 1).values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    axes[1,0].plot(x, p(x), 'k--', alpha=0.8, linewidth=2)
    
    axes[1,0].set_xlabel('Rainfall (mm)', fontsize=11)
    axes[1,0].set_ylabel('Log(Fire Frequency)', fontsize=11)
    axes[1,0].set_title('Fire vs Rainfall', fontsize=12, fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Correlation heatmap
    corr_vars = ['count', 'mean_max_temp', 'mean_min_temp', 'mean_rainfall']
    corr_matrix = data[corr_vars].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=axes[1,1], fmt='.3f')
    axes[1,1].set_title('Correlation Matrix', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(get_output_path('correlation_plots.png'), dpi=300, bbox_inches='tight')
    print("Correlation plots saved as: correlation_plots.png")
    
    # Print correlation statistics
    print(f"\nCorrelation Statistics:")
    print(f"Fire vs Max Temp: r = {stats.pearsonr(data['count'], data['mean_max_temp'])[0]:.3f}")
    print(f"Fire vs Min Temp: r = {stats.pearsonr(data['count'], data['mean_min_temp'])[0]:.3f}")
    print(f"Fire vs Rainfall: r = {stats.pearsonr(data['count'], data['mean_rainfall'])[0]:.3f}")

def analyze_model_results():
    """Analyze simulation model results if available"""
    
    print(f"\n{'='*50}")
    print("MODEL RESULTS ANALYSIS")
    print("="*50)
    
    # Look for model results
    result_files = []
    if os.path.exists(get_model_results_path("")):
        result_files = [f for f in os.listdir(get_model_results_path("")) if f.endswith("_metrics.csv")]
    
    if not result_files:
        print("No model result files found in model_results/")
        print("Run run_all_models.py first to generate model evaluation results.")
        return
    
    print(f"Found {len(result_files)} model result files")
    
    # Read and combine results
    all_results = []
    for file in result_files:
        try:
            df = pd.read_csv(get_model_results_path(file))
            
            # Extract metadata from filename
            parts = file.replace("_metrics.csv", "").split("_")
            time_period = parts[0] + "_" + parts[1]  # e.g., "five_year"
            theta = parts[2]
            model_type = "Bayesian" if file.endswith("b_metrics.csv") else "Standard"
            
            df['time_period'] = time_period
            df['theta_param'] = theta
            df['model_type'] = model_type
            
            all_results.append(df)
            
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Summary statistics
        print(f"\nModel Results Summary:")
        print(f"Total model runs: {len(combined_results):,}")
        print(f"Time periods: {combined_results['time_period'].unique()}")
        print(f"Theta values: {combined_results['theta_param'].unique()}")
        print(f"Model types: {combined_results['model_type'].unique()}")
        
        # Performance comparison
        print(f"\nPerformance Comparison (Mean ± Std):")
        performance_summary = combined_results.groupby(['model_type']).agg({
            'rmse_test': ['mean', 'std'],
            'mase_test': ['mean', 'std'],
            'bias_test': ['mean', 'std']
        }).round(4)
        
        print(performance_summary)
        
        # Save combined results
        combined_results.to_csv(get_output_path("combined_model_results.csv"), index=False)
        print(f"\nCombined results saved as: combined_model_results.csv")
        
        return combined_results
    
    return None

def main():
    """Main function to run all descriptive analyses"""
    
    print("="*60)
    print("DESCRIPTIVE STATISTICS AND VISUALIZATION")
    print("Python version of 8descriptive_stats.R")
    print("="*60)
    
    # Load data
    data = load_and_prepare_data()
    if data is None:
        return
    
    # Generate summary statistics
    summary_stats = generate_summary_statistics(data)
    
    # Create visualizations
    print(f"\nGenerating visualizations...")
    
    # Time series plots
    time_series_fig = create_time_series_plots(data)
    
    # Monthly patterns
    monthly_avg = create_monthly_patterns(data)
    
    # Correlation analysis
    create_correlation_plots(data)
    
    # Analyze model results if available
    model_results = analyze_model_results()
    
    # Final summary
    print(f"\n{'='*60}")
    print("DESCRIPTIVE ANALYSIS COMPLETED")
    print("="*60)
    
    print(f"\nGenerated files:")
    output_files = [
        'time_series_plots.png',
        'monthly_patterns.png', 
        'correlation_plots.png'
    ]
    
    for file in output_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (not created)")
    
    if model_results is not None:
        print(f"  ✓ combined_model_results.csv")
    
    print(f"\nAnalysis complete! Check the generated plots and files.")

if __name__ == "__main__":
    main() 