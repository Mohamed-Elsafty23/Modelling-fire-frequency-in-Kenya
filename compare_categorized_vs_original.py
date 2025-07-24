#!/usr/bin/env python3
"""
Comparison script between categorized and original fire frequency models
Compares performance metrics and visualizations between the two approaches
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings('ignore')

# Create output directory for plots
OUTPUT_DIR = "comparison_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Created output directory: {OUTPUT_DIR}")

# Import the model functions from both scripts
import sys
import importlib.util

def load_module_from_file(file_path, module_name):
    """Load a module from a file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load both modules
categorized_module = load_module_from_file("6_final_models_real_categorized.py", "categorized")
original_module = load_module_from_file("6_final_models_real.py", "original")

def run_comparison():
    """Run both approaches and compare results"""
    
    print("="*80)
    print("FIRE FREQUENCY MODEL COMPARISON: CATEGORIZED vs ORIGINAL")
    print("="*80)
    
    # Load both datasets
    print("\n1. Loading datasets...")
    categorized_data = pd.read_csv("fire_data_2000-18_categorized.csv")
    categorized_data = categorized_data.rename(columns={"mean_max_temp":"max_temp", "mean_rainfall":"rainfall"})
    
    original_data = pd.read_csv("fire_data_2000-18.csv")
    original_data = original_data.rename(columns={"mean_max_temp":"max_temp", "mean_rainfall":"rainfall"})
    
    print(f"Categorized data shape: {categorized_data.shape}")
    print(f"Original data shape: {original_data.shape}")
    
    # Check if datasets have same number of observations
    if len(categorized_data) != len(original_data):
        print("WARNING: Datasets have different lengths!")
    
    # Show quadrant distribution in categorized data
    print("\nQuadrant distribution in categorized data:")
    quadrant_cols = ['quadrant_NE', 'quadrant_NW', 'quadrant_SE', 'quadrant_SW']
    if all(col in categorized_data.columns for col in quadrant_cols):
        print(categorized_data[quadrant_cols].sum())
    else:
        print("Quadrant columns not found in categorized data")
    
    # Run models on categorized data
    print("\n2. Running models on CATEGORIZED data...")
    print("   Running Standard Negative Binomial Model...")
    cat_nb_result = categorized_module.negbinner(categorized_data, theta=1.5, n=60)
    
    print("   Running Bayesian Negative Binomial Model...")
    cat_bayes_result = categorized_module.stanbinner(categorized_data, theta=1.5, n=60)
    
    # Run models on original data
    print("\n3. Running models on ORIGINAL data...")
    print("   Running Standard Negative Binomial Model...")
    orig_nb_result = original_module.negbinner(original_data, theta=1.5, n=60)
    
    print("   Running Bayesian Negative Binomial Model...")
    orig_bayes_result = original_module.stanbinner(original_data, theta=1.5, n=60)
    
    # Create comparison dataframes
    print("\n4. Comparing performance metrics...")
    
    # Metrics comparison
    metrics_comparison = pd.DataFrame({
        'Categorized_NB': [cat_nb_result['rmse_train'], cat_nb_result['rmse_test'], 
                          cat_nb_result['mase_test'], cat_nb_result['bias_test']],
        'Categorized_Bayes': [cat_bayes_result['rmse_train'], cat_bayes_result['rmse_test'],
                             cat_bayes_result['mase_test'], cat_bayes_result['bias_test']],
        'Original_NB': [orig_nb_result['rmse_train'], orig_nb_result['rmse_test'],
                       orig_nb_result['mase_test'], orig_nb_result['bias_test']],
        'Original_Bayes': [orig_bayes_result['rmse_train'], orig_bayes_result['rmse_test'],
                          orig_bayes_result['mase_test'], orig_bayes_result['bias_test']]
    }, index=['RMSE_Train', 'RMSE_Test', 'MASE_Test', 'Bias_Test'])
    
    print("\nPerformance Metrics Comparison:")
    print(metrics_comparison.round(4))
    
    # Calculate improvements
    print("\n5. Performance improvements (Categorized vs Original):")
    improvements = pd.DataFrame({
        'NB_Improvement': ((metrics_comparison['Original_NB'] - metrics_comparison['Categorized_NB']) / 
                          metrics_comparison['Original_NB'] * 100),
        'Bayes_Improvement': ((metrics_comparison['Original_Bayes'] - metrics_comparison['Categorized_Bayes']) / 
                             metrics_comparison['Original_Bayes'] * 100)
    })
    print(improvements.round(2))
    print("(Positive values indicate improvement with categorized data)")
    
    # Create visualization dataframes
    cat_nb_df = categorized_module.build_df(cat_nb_result, 'Categorized NB', categorized_data)
    cat_bayes_df = categorized_module.build_df(cat_bayes_result, 'Categorized Bayes', categorized_data)
    orig_nb_df = original_module.build_df(orig_nb_result, 'Original NB', original_data)
    orig_bayes_df = original_module.build_df(orig_bayes_result, 'Original Bayes', original_data)
    
    # Combine all results
    all_results = pd.concat([cat_nb_df, cat_bayes_df, orig_nb_df, orig_bayes_df])
    
    # Analyze quadrant distributions for climate variables
    print("\n6. Analyzing quadrant distributions for climate variables...")
    analyze_quadrant_distributions(categorized_data)
    
    # Create comprehensive comparison plots
    print("\n7. Generating comparison visualizations...")
    create_comparison_plots(metrics_comparison, all_results, categorized_data)
    
    # Quadrant-specific analysis for categorized data
    if 'quadrant' in categorized_data.columns:
        print("\n8. Quadrant-specific analysis...")
        quadrant_analysis(cat_nb_df, cat_bayes_df, categorized_data)
    
    return {
        'metrics_comparison': metrics_comparison,
        'improvements': improvements,
        'categorized_results': {'nb': cat_nb_result, 'bayes': cat_bayes_result},
        'original_results': {'nb': orig_nb_result, 'bayes': orig_bayes_result},
        'plot_data': all_results
    }

def analyze_quadrant_distributions(categorized_data):
    """Analyze and visualize the distribution of climate variables across quadrants"""
    
    print("\nQuadrant distribution summary:")
    quadrant_counts = categorized_data['quadrant'].value_counts()
    print(quadrant_counts)
    
    # Create comprehensive distribution analysis
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # 1. Quadrant counts
    ax1 = axes[0, 0]
    quadrant_counts.plot(kind='bar', ax=ax1, color=['red', 'blue', 'green', 'orange'])
    ax1.set_title('Number of Observations by Quadrant')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(quadrant_counts.values):
        ax1.text(i, v + 1, str(v), ha='center', va='bottom')
    
    # 2. Temperature distribution by quadrant
    ax2 = axes[0, 1]
    sns.boxplot(data=categorized_data, x='quadrant', y='max_temp', ax=ax2)
    ax2.set_title('Maximum Temperature Distribution by Quadrant')
    ax2.set_xlabel('Quadrant')
    ax2.set_ylabel('Max Temperature (°C)')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Rainfall distribution by quadrant
    ax3 = axes[1, 0]
    sns.boxplot(data=categorized_data, x='quadrant', y='rainfall', ax=ax3)
    ax3.set_title('Rainfall Distribution by Quadrant')
    ax3.set_xlabel('Quadrant')
    ax3.set_ylabel('Rainfall (mm)')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Fire count distribution by quadrant
    ax4 = axes[1, 1]
    sns.boxplot(data=categorized_data, x='quadrant', y='count', ax=ax4)
    ax4.set_title('Fire Count Distribution by Quadrant')
    ax4.set_xlabel('Quadrant')
    ax4.set_ylabel('Fire Count')
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Temperature vs Rainfall scatter plot colored by quadrant
    ax5 = axes[2, 0]
    quadrant_colors = {'Northeast': 'red', 'Northwest': 'blue', 'Southeast': 'green', 'Southwest': 'orange'}
    
    for quadrant, color in quadrant_colors.items():
        quad_data = categorized_data[categorized_data['quadrant'] == quadrant]
        ax5.scatter(quad_data['max_temp'], quad_data['rainfall'], 
                   c=color, label=quadrant, alpha=0.6, s=30)
    
    ax5.set_xlabel('Max Temperature (°C)')
    ax5.set_ylabel('Rainfall (mm)')
    ax5.set_title('Temperature vs Rainfall by Quadrant')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary statistics table as text
    ax6 = axes[2, 1]
    ax6.axis('off')
    
    # Calculate summary statistics by quadrant
    summary_stats = categorized_data.groupby('quadrant')[['max_temp', 'rainfall', 'count']].agg([
        'mean', 'std', 'min', 'max'
    ]).round(2)
    
    # Create a formatted text summary
    summary_text = "Summary Statistics by Quadrant\n\n"
    for quadrant in ['Northeast', 'Northwest', 'Southeast', 'Southwest']:
        if quadrant in summary_stats.index:
            stats = summary_stats.loc[quadrant]
            summary_text += f"{quadrant}:\n"
            summary_text += f"  Temperature: {stats[('max_temp', 'mean')]:.1f}±{stats[('max_temp', 'std')]:.1f}°C\n"
            summary_text += f"  Rainfall: {stats[('rainfall', 'mean')]:.1f}±{stats[('rainfall', 'std')]:.1f}mm\n"
            summary_text += f"  Fire Count: {stats[('count', 'mean')]:.1f}±{stats[('count', 'std')]:.1f}\n"
            summary_text += f"  Observations: {len(categorized_data[categorized_data['quadrant'] == quadrant])}\n\n"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/quadrant_distributions.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Seasonal analysis by quadrant
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, quadrant in enumerate(['Northeast', 'Northwest', 'Southeast', 'Southwest']):
        ax = axes[i]
        quad_data = categorized_data[categorized_data['quadrant'] == quadrant]
        
        if len(quad_data) > 0:
            # Monthly averages
            monthly_stats = quad_data.groupby('month')[['max_temp', 'rainfall', 'count']].mean()
            
            # Create dual y-axis plot
            ax2 = ax.twinx()
            
            # Plot temperature and fire count on left axis
            line1 = ax.plot(monthly_stats.index, monthly_stats['max_temp'], 
                           'r-o', label='Max Temp (°C)', linewidth=2)
            line2 = ax.plot(monthly_stats.index, monthly_stats['count'], 
                           'g-s', label='Fire Count', linewidth=2)
            
            # Plot rainfall on right axis
            line3 = ax2.plot(monthly_stats.index, monthly_stats['rainfall'], 
                            'b-^', label='Rainfall (mm)', linewidth=2)
            
            ax.set_xlabel('Month')
            ax.set_ylabel('Temperature (°C) / Fire Count')
            ax2.set_ylabel('Rainfall (mm)', color='blue')
            ax.set_title(f'{quadrant} - Seasonal Patterns')
            ax.grid(True, alpha=0.3)
            
            # Combine legends
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/seasonal_patterns_by_quadrant.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return summary_stats

def create_comparison_plots(metrics_df, plot_data, categorized_data):
    """Create comprehensive comparison plots"""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    
    # Create a 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Metrics comparison bar plot
    ax1 = axes[0, 0]
    metrics_df.T.plot(kind='bar', ax=ax1, rot=45)
    ax1.set_title('Performance Metrics Comparison')
    ax1.set_ylabel('Metric Value')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. RMSE Test comparison
    ax2 = axes[0, 1]
    rmse_data = metrics_df.loc['RMSE_Test']
    colors = ['lightcoral', 'lightblue', 'orange', 'lightgreen']
    bars = ax2.bar(rmse_data.index, rmse_data.values, color=colors)
    ax2.set_title('Test RMSE Comparison')
    ax2.set_ylabel('RMSE')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 3. Time series comparison - Test set only
    ax3 = axes[0, 2]
    test_data = plot_data[plot_data['set'] == 'test']
    
    # Plot actual values once
    actual_data = test_data[test_data['model'] == 'Categorized NB']
    ax3.plot(actual_data['date'], actual_data['actual'], 
             label='Actual', color='black', linewidth=2, alpha=0.8)
    
    # Plot predictions for each model
    colors = {'Categorized NB': 'red', 'Categorized Bayes': 'blue', 
              'Original NB': 'orange', 'Original Bayes': 'green'}
    
    for model, color in colors.items():
        model_data = test_data[test_data['model'] == model]
        ax3.plot(model_data['date'], model_data['predicted'], 
                label=f'{model} Pred', color=color, linestyle='--', alpha=0.7)
    
    ax3.set_title('Test Set Predictions Comparison')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Fire Count')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Residuals comparison
    ax4 = axes[1, 0]
    for model, color in colors.items():
        model_data = test_data[test_data['model'] == model]
        residuals = model_data['actual'] - model_data['predicted']
        ax4.scatter(model_data['predicted'], residuals, 
                   label=model, alpha=0.6, color=color, s=30)
    
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_title('Residuals vs Fitted (Test Set)')
    ax4.set_xlabel('Fitted Values')
    ax4.set_ylabel('Residuals')
    ax4.legend()
    
    # 5. Prediction accuracy scatter
    ax5 = axes[1, 1]
    for model, color in colors.items():
        model_data = test_data[test_data['model'] == model]
        ax5.scatter(model_data['actual'], model_data['predicted'], 
                   label=model, alpha=0.6, color=color, s=30)
    
    # Add perfect prediction line
    min_val = test_data['actual'].min()
    max_val = test_data['actual'].max()
    ax5.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    
    ax5.set_title('Predicted vs Actual (Test Set)')
    ax5.set_xlabel('Actual')
    ax5.set_ylabel('Predicted')
    ax5.legend()
    
    # 6. Model complexity comparison
    ax6 = axes[1, 2]
    
    # Create a simple complexity comparison
    model_features = {
        'Original NB': 2,  # max_temp, rainfall
        'Original Bayes': 4,  # max_temp, rainfall, cos, sin
        'Categorized NB': 6,  # max_temp, rainfall + 4 quadrants
        'Categorized Bayes': 8  # max_temp, rainfall, cos, sin + 4 quadrants
    }
    
    complexity_df = pd.DataFrame(list(model_features.items()), 
                                columns=['Model', 'Features'])
    bars = ax6.bar(complexity_df['Model'], complexity_df['Features'], 
                   color=['orange', 'green', 'red', 'blue'])
    ax6.set_title('Model Complexity (Number of Features)')
    ax6.set_ylabel('Number of Features')
    ax6.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def quadrant_analysis(cat_nb_df, cat_bayes_df, categorized_data):
    """Perform quadrant-specific analysis"""
    
    print("\nQuadrant-specific performance analysis:")
    
    # Combine categorized results
    cat_combined = pd.concat([cat_nb_df, cat_bayes_df])
    test_data = cat_combined[cat_combined['set'] == 'test']
    
    # Analysis by quadrant
    quadrant_performance = {}
    
    for quadrant in ['Northeast', 'Northwest', 'Southeast', 'Southwest']:
        quadrant_data = test_data[test_data['quadrant'] == quadrant]
        
        if len(quadrant_data) > 0:
            quadrant_perf = {}
            
            for model in ['Categorized NB', 'Categorized Bayes']:
                model_data = quadrant_data[quadrant_data['model'] == model]
                
                if len(model_data) > 0:
                    rmse = np.sqrt(mean_squared_error(model_data['actual'], model_data['predicted']))
                    mae = mean_absolute_error(model_data['actual'], model_data['predicted'])
                    bias = np.mean((model_data['actual'] - model_data['predicted'])/np.abs(model_data['actual']))
                    
                    quadrant_perf[model] = {
                        'RMSE': rmse,
                        'MAE': mae,
                        'Bias': bias,
                        'n_obs': len(model_data)
                    }
            
            quadrant_performance[quadrant] = quadrant_perf
            
            print(f"\n{quadrant} Quadrant:")
            for model, metrics in quadrant_perf.items():
                print(f"  {model}: RMSE={metrics['RMSE']:.3f}, MAE={metrics['MAE']:.3f}, "
                      f"Bias={metrics['Bias']:.3f}, n={metrics['n_obs']}")
    
    # Create quadrant comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    quadrant_colors = {'Northeast': 'red', 'Northwest': 'blue', 
                      'Southeast': 'green', 'Southwest': 'orange'}
    
    for i, (quadrant, color) in enumerate(quadrant_colors.items()):
        ax = axes[i]
        quad_data = test_data[test_data['quadrant'] == quadrant]
        
        if len(quad_data) > 0:
            for model in ['Categorized NB', 'Categorized Bayes']:
                model_data = quad_data[quad_data['model'] == model]
                
                if len(model_data) > 0:
                    linestyle = '-' if 'NB' in model else '--'
                    alpha = 0.8 if 'NB' in model else 0.6
                    
                    ax.plot(model_data['date'], model_data['actual'], 
                           color='black', linewidth=2, alpha=0.9, label='Actual' if model == 'Categorized NB' else "")
                    ax.plot(model_data['date'], model_data['predicted'], 
                           color=color, linestyle=linestyle, alpha=alpha,
                           label=f'{model} Pred')
        
        ax.set_title(f'{quadrant} Quadrant')
        ax.set_xlabel('Date')
        ax.set_ylabel('Fire Count')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/quadrant_specific_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return quadrant_performance

def summary_report(comparison_results):
    """Generate a summary report"""
    
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    metrics = comparison_results['metrics_comparison']
    improvements = comparison_results['improvements']
    
    print("\n1. KEY FINDINGS:")
    
    # Best performing models
    print("\nBest Test RMSE:")
    test_rmse = metrics.loc['RMSE_Test']
    best_rmse_model = test_rmse.idxmin()
    print(f"   {best_rmse_model}: {test_rmse[best_rmse_model]:.4f}")
    
    print("\nBest Test MASE:")
    test_mase = metrics.loc['MASE_Test']
    best_mase_model = test_mase.idxmin()
    print(f"   {best_mase_model}: {test_mase[best_mase_model]:.4f}")
    
    # Average improvements
    print("\n2. AVERAGE IMPROVEMENTS (Categorized vs Original):")
    avg_improvements = improvements.mean()
    print(f"   Negative Binomial: {avg_improvements['NB_Improvement']:.2f}%")
    print(f"   Bayesian: {avg_improvements['Bayes_Improvement']:.2f}%")
    
    # Recommendations
    print("\n3. RECOMMENDATIONS:")
    if avg_improvements['NB_Improvement'] > 0 and avg_improvements['Bayes_Improvement'] > 0:
        print("   ✓ The categorized approach shows improvements in both models")
        print("   ✓ Including spatial quadrant features enhances prediction accuracy")
    elif avg_improvements['NB_Improvement'] > avg_improvements['Bayes_Improvement']:
        print("   ✓ Categorization particularly benefits the Standard NB model")
    else:
        print("   ✓ Categorization particularly benefits the Bayesian model")
    
    print(f"\n   Best overall model: {best_rmse_model}")
    print("   Consider using categorized data for improved spatial modeling")

if __name__ == "__main__":
    # Run the comparison
    results = run_comparison()
    
    # Save detailed results to CSV
    print(f"\n9. Saving detailed results...")
    
    # Save metrics comparison
    results['metrics_comparison'].to_csv(f"{OUTPUT_DIR}/metrics_comparison.csv")
    
    # Save improvements
    results['improvements'].to_csv(f"{OUTPUT_DIR}/improvements_analysis.csv")
    
    # Save plot data for further analysis
    results['plot_data'].to_csv(f"{OUTPUT_DIR}/prediction_results.csv", index=False)
    
    # Generate summary report
    summary_report(results)
    
    print(f"\nComparison complete! Check the results in '{OUTPUT_DIR}/':")
    print(f"  Visualizations:")
    print(f"    - {OUTPUT_DIR}/quadrant_distributions.png")
    print(f"    - {OUTPUT_DIR}/seasonal_patterns_by_quadrant.png")
    print(f"    - {OUTPUT_DIR}/model_comparison_analysis.png")
    print(f"    - {OUTPUT_DIR}/quadrant_specific_analysis.png")
    print(f"  Data files:")
    print(f"    - {OUTPUT_DIR}/metrics_comparison.csv")
    print(f"    - {OUTPUT_DIR}/improvements_analysis.csv")
    print(f"    - {OUTPUT_DIR}/prediction_results.csv") 