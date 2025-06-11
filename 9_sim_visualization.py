from output_utils import get_output_path, get_model_results_path, get_simulated_data_path, ensure_output_dirs
#!/usr/bin/env python3
"""
Simulation results visualization - Python version of 9sim_visualization.R
Creates comprehensive visualizations of model performance across different scenarios
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("Set1")

def load_all_results():
    """Load and combine all model result files"""
    
    print("Loading model results...")
    
    if not os.path.exists(get_model_results_path("")):
        print("Error: model_results directory not found!")
        print("Please run run_all_models.py first to generate model results.")
        return None
    
    # Get all result files
    result_files = glob.glob(get_model_results_path("*_metrics.csv"))
    
    if not result_files:
        print("No result files found in model_results/")
        return None
    
    print(f"Found {len(result_files)} result files")
    
    all_results = []
    
    for file in result_files:
        try:
            df = pd.read_csv(file)
            
            # Extract metadata from filename
            filename = os.path.basename(file)
            parts = filename.replace("_metrics.csv", "").split("_")
            
            # Parse filename: e.g., "five_year_1.5_metrics.csv" or "five_year_1.5b_metrics.csv"
            time_period = parts[0] + "_" + parts[1]  # e.g., "five_year"
            theta_str = parts[2]
            
            # Determine model type
            if filename.endswith("b_metrics.csv"):
                model_type = "BNB"  # Bayesian Negative Binomial
                theta = theta_str.replace("b", "")
            else:
                model_type = "NB"   # Standard Negative Binomial
                theta = theta_str
            
            # Convert time period to numeric (years)
            time_mapping = {
                "five_year": 5, "ten_year": 10, 
                "twenty_year": 20, "thirty_year": 30
            }
            years = time_mapping.get(time_period, 0)
            
            # Add metadata to dataframe
            df['time_period'] = time_period
            df['years'] = years
            df['theta_param'] = float(theta)
            df['model_type'] = model_type
            df['n_months'] = years * 12
            
            all_results.append(df)
            
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
    
    if not all_results:
        print("No valid result files found!")
        return None
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    print(f"Loaded {len(combined_results):,} model runs")
    print(f"Time periods: {sorted(combined_results['years'].unique())} years")
    print(f"Theta values: {sorted(combined_results['theta_param'].unique())}")
    print(f"Model types: {combined_results['model_type'].unique()}")
    
    return combined_results

def create_performance_comparison_plots(data):
    """Create comprehensive performance comparison plots"""
    
    print("Creating performance comparison plots...")
    
    # Set up the figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison Across Scenarios', fontsize=16, fontweight='bold')
    
    # Plot 1: RMSE on Test Data
    sns.violinplot(data=data, x='theta_param', y='rmse_test', hue='model_type', 
                   ax=axes[0,0], inner='box')
    axes[0,0].set_title('a. RMSE on Test Data', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Theta (θ)', fontsize=12, fontweight='bold')
    axes[0,0].set_ylabel('RMSE on Test Data', fontsize=12, fontweight='bold')
    axes[0,0].legend(title='Model', fontsize=10)
    
    # Plot 2: MASE on Test Data
    sns.violinplot(data=data, x='theta_param', y='mase_test', hue='model_type', 
                   ax=axes[0,1], inner='box')
    axes[0,1].set_title('b. MASE on Test Data', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Theta (θ)', fontsize=12, fontweight='bold')
    axes[0,1].set_ylabel('MASE on Test Data', fontsize=12, fontweight='bold')
    axes[0,1].legend(title='Model', fontsize=10)
    
    # Plot 3: Bias on Test Data
    sns.violinplot(data=data, x='theta_param', y='bias_test', hue='model_type', 
                   ax=axes[1,0], inner='box')
    axes[1,0].set_title('c. Bias on Test Data', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Theta (θ)', fontsize=12, fontweight='bold')
    axes[1,0].set_ylabel('Bias on Test Data', fontsize=12, fontweight='bold')
    axes[1,0].legend(title='Model', fontsize=10)
    axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Plot 4: RMSE on Training Data
    sns.violinplot(data=data, x='theta_param', y='rmse_train', hue='model_type', 
                   ax=axes[1,1], inner='box')
    axes[1,1].set_title('d. RMSE on Training Data', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Theta (θ)', fontsize=12, fontweight='bold')
    axes[1,1].set_ylabel('RMSE on Training Data', fontsize=12, fontweight='bold')
    axes[1,1].legend(title='Model', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(get_output_path('model_performance_comparison.png'), dpi=300, bbox_inches='tight')
    print("Performance comparison plot saved as: model_performance_comparison.png")
    
    return fig

def create_time_period_analysis(data):
    """Create analysis by time period (sample size)"""
    
    print("Creating time period analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance by Time Period (Sample Size)', fontsize=16, fontweight='bold')
    
    # Plot 1: RMSE Test by Years and Theta
    pivot_rmse = data.groupby(['years', 'theta_param', 'model_type'])['rmse_test'].mean().reset_index()
    
    for model in ['NB', 'BNB']:
        model_data = pivot_rmse[pivot_rmse['model_type'] == model]
        for theta in sorted(data['theta_param'].unique()):
            theta_data = model_data[model_data['theta_param'] == theta]
            axes[0,0].plot(theta_data['years'], theta_data['rmse_test'], 
                          marker='o', label=f'{model} θ={theta}', linewidth=2, markersize=6)
    
    axes[0,0].set_title('RMSE Test vs Time Period', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Time Period (Years)', fontsize=12)
    axes[0,0].set_ylabel('Mean RMSE Test', fontsize=12)
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: MASE by sample size
    sns.boxplot(data=data, x='years', y='mase_test', hue='model_type', ax=axes[0,1])
    axes[0,1].set_title('MASE by Time Period', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Time Period (Years)', fontsize=12)
    axes[0,1].set_ylabel('MASE Test', fontsize=12)
    axes[0,1].legend(title='Model', fontsize=10)
    
    # Plot 3: Bias variability by time period
    bias_stats = data.groupby(['years', 'model_type'])['bias_test'].agg(['mean', 'std']).reset_index()
    
    for model in ['NB', 'BNB']:
        model_stats = bias_stats[bias_stats['model_type'] == model]
        axes[1,0].errorbar(model_stats['years'], model_stats['mean'], 
                          yerr=model_stats['std'], marker='o', capsize=5, 
                          label=model, linewidth=2, markersize=8)
    
    axes[1,0].set_title('Bias Mean ± Std by Time Period', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Time Period (Years)', fontsize=12)
    axes[1,0].set_ylabel('Bias Test (Mean ± Std)', fontsize=12)
    axes[1,0].legend(title='Model', fontsize=10)
    axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Performance improvement (BNB vs NB)
    nb_rmse = data[data['model_type'] == 'NB'].groupby(['years', 'theta_param'])['rmse_test'].mean()
    bnb_rmse = data[data['model_type'] == 'BNB'].groupby(['years', 'theta_param'])['rmse_test'].mean()
    
    improvement_data = []
    for (years, theta), nb_val in nb_rmse.items():
        if (years, theta) in bnb_rmse.index:
            bnb_val = bnb_rmse[(years, theta)]
            improvement = ((nb_val - bnb_val) / nb_val) * 100  # Percentage improvement
            improvement_data.append({
                'years': years, 'theta': theta, 'improvement_pct': improvement
            })
    
    if improvement_data:
        improvement_df = pd.DataFrame(improvement_data)
        
        for theta in sorted(improvement_df['theta'].unique()):
            theta_data = improvement_df[improvement_df['theta'] == theta]
            axes[1,1].plot(theta_data['years'], theta_data['improvement_pct'], 
                          marker='o', label=f'θ={theta}', linewidth=2, markersize=6)
        
        axes[1,1].set_title('BNB Improvement over NB (%)', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Time Period (Years)', fontsize=12)
        axes[1,1].set_ylabel('RMSE Improvement (%)', fontsize=12)
        axes[1,1].legend(title='Theta', fontsize=10)
        axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(get_output_path('time_period_analysis.png'), dpi=300, bbox_inches='tight')
    print("Time period analysis saved as: time_period_analysis.png")
    
    return fig

def create_summary_statistics_table(data):
    """Create detailed summary statistics table"""
    
    print("Creating summary statistics...")
    
    # Overall summary by model type
    summary_by_model = data.groupby('model_type').agg({
        'rmse_train': ['mean', 'std', 'min', 'max'],
        'rmse_test': ['mean', 'std', 'min', 'max'],
        'mase_test': ['mean', 'std', 'min', 'max'],
        'bias_test': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    print("\n" + "="*60)
    print("OVERALL MODEL PERFORMANCE SUMMARY")
    print("="*60)
    print(summary_by_model)
    
    # Summary by theta parameter
    summary_by_theta = data.groupby(['model_type', 'theta_param']).agg({
        'rmse_test': ['mean', 'std'],
        'mase_test': ['mean', 'std'],
        'bias_test': ['mean', 'std']
    }).round(4)
    
    print(f"\n{'='*60}")
    print("PERFORMANCE BY THETA PARAMETER")
    print("="*60)
    print(summary_by_theta)
    
    # Summary by time period
    summary_by_time = data.groupby(['model_type', 'years']).agg({
        'rmse_test': ['mean', 'std'],
        'mase_test': ['mean', 'std'],
        'bias_test': ['mean', 'std']
    }).round(4)
    
    print(f"\n{'='*60}")
    print("PERFORMANCE BY TIME PERIOD")
    print("="*60)
    print(summary_by_time)
    
    # Save detailed results
    detailed_summary = data.groupby(['model_type', 'years', 'theta_param']).agg({
        'rmse_train': ['count', 'mean', 'std', 'min', 'max'],
        'rmse_test': ['mean', 'std', 'min', 'max'],
        'mase_test': ['mean', 'std', 'min', 'max'],
        'bias_test': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    detailed_summary.to_csv(get_output_path('detailed_model_summary.csv'))
    print(f"\nDetailed summary saved as: detailed_model_summary.csv")
    
    return summary_by_model, summary_by_theta, summary_by_time

def create_theta_comparison_heatmap(data):
    """Create heatmap showing performance across theta values"""
    
    print("Creating theta comparison heatmap...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Performance Heatmaps by Theta Parameter', fontsize=16, fontweight='bold')
    
    metrics = ['rmse_test', 'mase_test', 'bias_test', 'rmse_train']
    titles = ['RMSE Test', 'MASE Test', 'Bias Test', 'RMSE Train']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i//2, i%2]
        
        # Create pivot table for heatmap
        pivot_data = data.groupby(['model_type', 'theta_param'])[metric].mean().reset_index()
        pivot_table = pivot_data.pivot(index='model_type', columns='theta_param', values=metric)
        
        # Create heatmap
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis', 
                   ax=ax, cbar_kws={'label': title})
        ax.set_title(f'{title} by Model and Theta', fontsize=12, fontweight='bold')
        ax.set_xlabel('Theta Parameter', fontsize=11)
        ax.set_ylabel('Model Type', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(get_output_path('theta_performance_heatmap.png'), dpi=300, bbox_inches='tight')
    print("Theta performance heatmap saved as: theta_performance_heatmap.png")
    
    return fig

def generate_final_report(data):
    """Generate a final performance report"""
    
    print(f"\n{'='*80}")
    print("FINAL MODEL EVALUATION REPORT")
    print("="*80)
    
    # Key findings
    print("\nKEY FINDINGS:")
    print("-" * 40)
    
    # Best performing model overall
    avg_performance = data.groupby('model_type').agg({
        'rmse_test': 'mean',
        'mase_test': 'mean',
        'bias_test': lambda x: abs(x).mean()  # Absolute bias
    })
    
    best_rmse = avg_performance['rmse_test'].idxmin()
    best_mase = avg_performance['mase_test'].idxmin()
    best_bias = avg_performance['bias_test'].idxmin()
    
    print(f"• Best RMSE performance: {best_rmse} (avg: {avg_performance.loc[best_rmse, 'rmse_test']:.3f})")
    print(f"• Best MASE performance: {best_mase} (avg: {avg_performance.loc[best_mase, 'mase_test']:.3f})")
    print(f"• Best bias performance: {best_bias} (avg: {avg_performance.loc[best_bias, 'bias_test']:.3f})")
    
    # Performance by theta
    print(f"\nPERFORMANCE BY THETA:")
    print("-" * 40)
    theta_performance = data.groupby(['model_type', 'theta_param'])['rmse_test'].mean()
    
    for model in ['NB', 'BNB']:
        model_performance = theta_performance[model]
        best_theta = model_performance.idxmin()
        worst_theta = model_performance.idxmax()
        
        print(f"• {model}: Best θ={best_theta} (RMSE: {model_performance[best_theta]:.3f})")
        print(f"  {model}: Worst θ={worst_theta} (RMSE: {model_performance[worst_theta]:.3f})")
    
    # Performance by time period
    print(f"\nPERFORMANCE BY TIME PERIOD:")
    print("-" * 40)
    time_performance = data.groupby(['model_type', 'years'])['rmse_test'].mean()
    
    for model in ['NB', 'BNB']:
        model_performance = time_performance[model]
        print(f"• {model} performance by years:")
        for years in sorted(model_performance.index):
            print(f"    {years} years: RMSE = {model_performance[years]:.3f}")
    
    # Statistical significance test
    print(f"\nSTATISTICAL COMPARISON:")
    print("-" * 40)
    from scipy import stats as scipy_stats
    
    nb_rmse = data[data['model_type'] == 'NB']['rmse_test']
    bnb_rmse = data[data['model_type'] == 'BNB']['rmse_test']
    
    if len(nb_rmse) > 0 and len(bnb_rmse) > 0:
        t_stat, p_value = scipy_stats.ttest_ind(nb_rmse, bnb_rmse)
        
        print(f"• T-test comparing NB vs BNB RMSE:")
        print(f"    t-statistic: {t_stat:.3f}")
        print(f"    p-value: {p_value:.6f}")
        print(f"    Significant difference: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")
        
        print(f"• Effect size (Cohen's d): {(nb_rmse.mean() - bnb_rmse.mean()) / np.sqrt((nb_rmse.var() + bnb_rmse.var()) / 2):.3f}")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    print("-" * 40)
    
    overall_best = avg_performance['rmse_test'].idxmin()
    print(f"• Overall best model: {overall_best}")
    
    # Best theta for each model
    for model in ['NB', 'BNB']:
        model_data = data[data['model_type'] == model]
        best_theta_rmse = model_data.groupby('theta_param')['rmse_test'].mean().idxmin()
        print(f"• Best θ parameter for {model}: {best_theta_rmse}")
    
    # Sample size recommendations
    rmse_by_years = data.groupby('years')['rmse_test'].mean()
    min_years = rmse_by_years.idxmin()
    print(f"• Optimal time period: {min_years} years")
    
    print(f"\n{'='*80}")

def main():
    """Main function to create all visualizations and analysis"""
    
    print("="*80)
    print("SIMULATION RESULTS VISUALIZATION PIPELINE")
    print("Python version of 9sim_visualization.R")
    print("="*80)
    
    # Load all results
    data = load_all_results()
    if data is None:
        return
    
    # Create output directory for plots
    os.makedirs("plots", exist_ok=True)
    
    # Create all visualizations
    print(f"\nGenerating comprehensive visualizations...")
    
    # 1. Performance comparison plots
    perf_fig = create_performance_comparison_plots(data)
    
    # 2. Time period analysis
    time_fig = create_time_period_analysis(data)
    
    # 3. Theta comparison heatmap
    heatmap_fig = create_theta_comparison_heatmap(data)
    
    # 4. Summary statistics
    summary_stats = create_summary_statistics_table(data)
    
    # 5. Final report
    generate_final_report(data)
    
    # Save combined dataset for further analysis
    data.to_csv(get_output_path('all_simulation_results.csv'), index=False)
    
    # Final summary
    print(f"\n{'='*80}")
    print("VISUALIZATION PIPELINE COMPLETED")
    print("="*80)
    
    output_files = [
        'model_performance_comparison.png',
        'time_period_analysis.png',
        'theta_performance_heatmap.png',
        get_output_path('detailed_model_summary.csv'),
        get_output_path('all_simulation_results.csv')
    ]
    
    print(f"\nGenerated files:")
    for file in output_files:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024*1024)
            print(f"  ✓ {file} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {file} (not created)")
    
    print(f"\nAll visualizations and analysis completed!")
    print(f"Check the generated plots and CSV files for detailed results.")

if __name__ == "__main__":
    main() 