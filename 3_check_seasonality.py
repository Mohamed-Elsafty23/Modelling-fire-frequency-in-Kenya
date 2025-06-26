#!/usr/bin/env python3
"""
Seasonality analysis script - Python version of 3check_seasonality.R
Uses TBATS model to check for seasonal patterns in climate data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from output_utils import get_output_path, ensure_output_dirs
import warnings

# Suppress sklearn deprecation warnings from tbats package
warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*ensure_all_finite.*", category=FutureWarning)

def check_seasonality():
    """Check for seasonality in climate time series data using TBATS model"""
    
    print("Reading fire-climate data...")
    
    # Ensure output directory exists
    ensure_output_dirs()
    
    try:
        series_data = pd.read_csv("fire_data_2000-18.csv")
    except FileNotFoundError:
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
    print("TBATS SEASONALITY ANALYSIS RESULTS")
    print("="*50)
    
    # Try to import TBATS
    try:
        from tbats import TBATS
        tbats_available = True
        tbats_package = 'tbats'
        print("Using tbats package for seasonality detection")
    except ImportError:
        try:
            from sktime.forecasting.tbats import TBATS
            tbats_available = True
            tbats_package = 'sktime'
            print("Using sktime.forecasting.tbats for seasonality detection")
        except ImportError:
            print("ERROR: TBATS not available. Installing tbats package...")
            import subprocess
            import sys
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "tbats"])
                from tbats import TBATS
                tbats_available = True
                tbats_package = 'tbats'
                print("Successfully installed and imported tbats package")
            except Exception as e:
                print(f"Failed to install tbats: {e}")
                print("Falling back to alternative seasonality detection methods...")
                tbats_available = False
                tbats_package = None
    
    if tbats_available:
        # Use TBATS for seasonality detection (matching R approach)
    for var, label in variables.items():
            print(f"\nAnalyzing {label} ({var}) with TBATS...")
        
        # Extract time series
        ts = series_data[var].dropna()
        
        if len(ts) < 24:  # Need at least 2 years for seasonal analysis
            print(f"Insufficient data for {var}")
                seasonality_results[var] = {
                    'has_seasonality': False,
                    'method': 'insufficient_data',
                    'seasonal_periods': None
                }
                continue
            
            try:
                if tbats_package == 'sktime':
                    print("  Skipping sktime TBATS due to performance issues")
                    # Skip sktime TBATS for now due to complexity
                    seasonality_results[var] = fallback_seasonality_analysis(ts, var)
                    
                else:  # tbats package
                    print("  Using simplified TBATS approach...")
                    # Create TBATS estimator with simplified parameters to avoid hanging
                    # Use smaller grid for efficiency
                    estimator = TBATS(
                        seasonal_periods=[12],
                        use_arma_errors=False,  # Disable ARMA for speed
                        n_jobs=1,  # Single thread to avoid multiprocessing issues
                        show_warnings=False
                    )
                    
                    # Add timeout protection
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError("TBATS fitting timed out")
                    
                    try:
                        # Set timeout for Windows (use threading instead of signal)
                        import threading
                        import time
                        
                        result_container = {}
                        
                        def fit_tbats():
                            try:
                                fitted = estimator.fit(ts.values)
                                result_container['fitted_model'] = fitted
                                result_container['success'] = True
                            except Exception as e:
                                result_container['error'] = str(e)
                                result_container['success'] = False
                        
                        # Start fitting in separate thread with timeout
                        fit_thread = threading.Thread(target=fit_tbats)
                        fit_thread.daemon = True
                        fit_thread.start()
                        fit_thread.join(timeout=30)  # 30 second timeout
                        
                        if fit_thread.is_alive():
                            print("  TBATS fitting timed out, falling back to alternative method")
                            seasonality_results[var] = fallback_seasonality_analysis(ts, var)
                            continue
                        
                        if not result_container.get('success', False):
                            error = result_container.get('error', 'Unknown error')
                            print(f"  TBATS fitting failed: {error}")
                            seasonality_results[var] = fallback_seasonality_analysis(ts, var)
            continue
                        
                        fitted_model = result_container['fitted_model']
                        
                        # Check if seasonal components were fitted
                        # This matches the R approach: !is.null(fit_tmax$seasonal)
                        has_seasonality = (
                            hasattr(fitted_model.params.components, 'seasonal_harmonics') and 
                            fitted_model.params.components.seasonal_harmonics is not None and
                            len(fitted_model.params.components.seasonal_harmonics) > 0 and
                            any(fitted_model.params.components.seasonal_harmonics > 0)
                        )
                        
                        seasonal_periods = fitted_model.params.components.seasonal_periods if has_seasonality else None
                        seasonal_harmonics = fitted_model.params.components.seasonal_harmonics if has_seasonality else None
                        
                        seasonality_results[var] = {
                            'has_seasonality': has_seasonality,
                            'method': 'tbats_direct',
                            'seasonal_periods': seasonal_periods,
                            'seasonal_harmonics': seasonal_harmonics,
                            'aic': fitted_model.aic,
                            'use_box_cox': fitted_model.params.components.use_box_cox,
                            'use_trend': fitted_model.params.components.use_trend,
                            'tbats_package': tbats_package
                        }
                        
                    except Exception as e:
                        print(f"  Error with timeout mechanism: {e}")
                        print("  Falling back to alternative method...")
                        seasonality_results[var] = fallback_seasonality_analysis(ts, var)
                
                # Print results based on the method used
                result = seasonality_results[var]
                print(f"  Seasonality detected: {'YES' if result['has_seasonality'] else 'NO'}")
                print(f"  Method used: {result['method']}")
                
                if result['has_seasonality']:
                    if 'seasonal_periods' in result and result['seasonal_periods']:
                        print(f"  Seasonal periods: {result['seasonal_periods']}")
                    if 'seasonal_harmonics' in result and result['seasonal_harmonics'] is not None:
                        print(f"  Seasonal harmonics: {result['seasonal_harmonics']}")
                
                if result['method'] == 'tbats_direct':
                    if 'aic' in result:
                        print(f"  AIC: {result['aic']:.2f}")
                    if 'use_box_cox' in result:
                        print(f"  Box-Cox transformation: {'YES' if result['use_box_cox'] else 'NO'}")
                    if 'use_trend' in result:
                        print(f"  Trend component: {'YES' if result['use_trend'] else 'NO'}")
                elif result['method'] == 'fallback':
                    if 'evidence_score' in result:
                        print(f"  Evidence score: {result['evidence_score']}/3")
                    if 'variance_ratio' in result:
                        print(f"  Monthly variance ratio: {result['variance_ratio']:.3f}")
                    if 'lag12_autocorr' in result:
                        print(f"  12-month autocorrelation: {result['lag12_autocorr']:.3f}")
                
            except Exception as e:
                print(f"Error fitting TBATS model for {var}: {e}")
                print("Falling back to alternative method...")
                
                # Fallback to simple seasonal analysis
                seasonality_results[var] = fallback_seasonality_analysis(ts, var)
    
    else:
        # Fallback approach if TBATS is not available
        print("Using fallback seasonality detection methods...")
        for var, label in variables.items():
            print(f"\nAnalyzing {label} ({var})...")
            ts = series_data[var].dropna()
            seasonality_results[var] = fallback_seasonality_analysis(ts, var)
    
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

def fallback_seasonality_analysis(ts, var_name):
    """Fallback seasonality analysis when TBATS is not available"""
    
    if len(ts) < 24:
        return {
            'has_seasonality': False,
            'method': 'insufficient_data',
            'seasonal_periods': None
        }
    
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        from statsmodels.tsa.stattools import acf
        
        # Method 1: Seasonal decomposition
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
        
        # Method 2: Monthly variance analysis
        monthly_means = ts.groupby(ts.index.month).mean()
        monthly_var = np.var(monthly_means)
        overall_var = np.var(ts)
        
        # High monthly variance relative to overall suggests seasonality
        variance_ratio = monthly_var / overall_var if overall_var > 0 else 0
        
        # Method 3: Autocorrelation at lag 12
            autocorr = acf(ts, nlags=12, fft=True)
            lag12_autocorr = abs(autocorr[12]) if len(autocorr) > 12 else 0
        
        # Combine evidence for seasonality
        seasonality_evidence = {
            'seasonal_decomp': seasonal_significant,
            'variance_ratio': variance_ratio > 0.1,
            'lag12_autocorr': lag12_autocorr > 0.3
        }
        
        # Overall seasonality decision
        seasonality_score = sum(seasonality_evidence.values())
        has_seasonality = seasonality_score >= 2  # Majority evidence
        
        print(f"  Seasonality detected: {'YES' if has_seasonality else 'NO'}")
        print(f"  Evidence score: {seasonality_score}/3")
        print(f"  - Seasonal decomposition: {'✓' if seasonal_significant else '✗'}")
        print(f"  - Monthly variance ratio: {variance_ratio:.3f} {'✓' if variance_ratio > 0.1 else '✗'}")
        print(f"  - 12-month autocorr: {lag12_autocorr:.3f} {'✓' if lag12_autocorr > 0.3 else '✗'}")
    
        return {
            'has_seasonality': has_seasonality,
            'method': 'fallback',
            'seasonal_periods': [12] if has_seasonality else None,
            'evidence_score': seasonality_score,
            'variance_ratio': variance_ratio,
            'lag12_autocorr': lag12_autocorr
        }
        
    except Exception as e:
        print(f"Error in fallback analysis for {var_name}: {e}")
        return {
            'has_seasonality': False,
            'method': 'error',
            'seasonal_periods': None
        }

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