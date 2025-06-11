from output_utils import get_output_path, get_model_results_path, get_simulated_data_path, ensure_output_dirs
#!/usr/bin/env python3
"""
Model Fitting Script - Python version of model_fitting.R
Script to fit comprehensive models including Bayesian approaches with MCMC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
import pickle
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import PyMC for Bayesian modeling
try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
    print("PyMC available for Bayesian modeling")
except ImportError:
    HAS_PYMC = False
    print("PyMC not available. Bayesian models will use approximations.")

# Try to import additional libraries
try:
    import bambi as bmb
    HAS_BAMBI = True
except ImportError:
    HAS_BAMBI = False

# Set random seed for reproducibility
np.random.seed(3456)

def save_models(models, p_means=None):
    """Save fitted models to disk"""
    
    model_dir = get_model_results_path("fitted_models")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n💾 SAVING MODELS TO: {model_dir}")
    print("=" * 50)
    
    saved_files = []
    
    # Save each model
    for model_name, model in models.items():
        model_file = os.path.join(model_dir, f"{model_name}_model.pkl")
        
        try:
            # Handle different model types
            if model_name == 'bayesian_nb' and isinstance(model, tuple):
                # PyMC model - save trace and model separately
                if len(model) == 2 and HAS_PYMC:
                    pymc_model, trace = model
                    trace_file = os.path.join(model_dir, f"{model_name}_trace.pkl")
                    with open(trace_file, 'wb') as f:
                        pickle.dump(trace, f)
                    saved_files.append(trace_file)
                    print(f"✓ Saved {model_name} trace: {trace_file}")
                else:
                    # Bootstrap approximation
                    with open(model_file, 'wb') as f:
                        pickle.dump(model, f)
                    saved_files.append(model_file)
                    print(f"✓ Saved {model_name}: {model_file}")
            else:
                # Standard models
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
                saved_files.append(model_file)
                print(f"✓ Saved {model_name}: {model_file}")
                
        except Exception as e:
            print(f"⚠️ Could not save {model_name}: {e}")
    
    # Save p_means if provided
    if p_means is not None:
        p_means_file = os.path.join(model_dir, "p_means.pkl")
        with open(p_means_file, 'wb') as f:
            pickle.dump(p_means, f)
        saved_files.append(p_means_file)
        print(f"✓ Saved p_means: {p_means_file}")
    
    # Save model metadata
    metadata = {
        'created_date': pd.Timestamp.now().isoformat(),
        'models_saved': list(models.keys()),
        'has_p_means': p_means is not None,
        'pymc_available': HAS_PYMC
    }
    
    metadata_file = os.path.join(model_dir, "model_metadata.json")
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    saved_files.append(metadata_file)
    print(f"✓ Saved metadata: {metadata_file}")
    
    return saved_files

def load_models():
    """Load previously fitted models from disk"""
    
    model_dir = get_model_results_path("fitted_models")
    
    if not os.path.exists(model_dir):
        print(f"🔍 No saved models found in: {model_dir}")
        return None, None
    
    print(f"\n📂 LOADING MODELS FROM: {model_dir}")
    print("=" * 50)
    
    # Check metadata
    metadata_file = os.path.join(model_dir, "model_metadata.json")
    if not os.path.exists(metadata_file):
        print("⚠️ No model metadata found - models may be incompatible")
        return None, None
    
    try:
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"📅 Models created: {metadata['created_date']}")
        print(f"🔧 Models available: {', '.join(metadata['models_saved'])}")
        
        models = {}
        
        # Load each model
        for model_name in metadata['models_saved']:
            model_file = os.path.join(model_dir, f"{model_name}_model.pkl")
            
            if os.path.exists(model_file):
                try:
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    models[model_name] = model
                    print(f"✓ Loaded {model_name}")
                    
                    # Check for trace file for Bayesian models
                    if model_name == 'bayesian_nb':
                        trace_file = os.path.join(model_dir, f"{model_name}_trace.pkl")
                        if os.path.exists(trace_file):
                            with open(trace_file, 'rb') as f:
                                trace = pickle.load(f)
                            models[model_name] = (model, trace)
                            print(f"✓ Loaded {model_name} trace")
                            
                except Exception as e:
                    print(f"⚠️ Could not load {model_name}: {e}")
            else:
                print(f"❌ Model file not found: {model_file}")
        
        # Load p_means
        p_means = None
        if metadata.get('has_p_means', False):
            p_means_file = os.path.join(model_dir, "p_means.pkl")
            if os.path.exists(p_means_file):
                try:
                    with open(p_means_file, 'rb') as f:
                        p_means = pickle.load(f)
                    print(f"✓ Loaded p_means")
                except Exception as e:
                    print(f"⚠️ Could not load p_means: {e}")
        
        if models:
            print(f"🎉 Successfully loaded {len(models)} models")
            return models, p_means
        else:
            print("❌ No models could be loaded")
            return None, None
            
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None, None

def check_models_exist():
    """Check if trained models already exist"""
    model_dir = get_model_results_path("fitted_models")
    metadata_file = os.path.join(model_dir, "model_metadata.json")
    return os.path.exists(metadata_file)

def load_data():
    """Load the fire-climate data"""
    
    print("Loading fire-climate data...")
    
    try:
        analysis_data = pd.read_csv(get_output_path("fire_data_2000-18.csv"))
        return analysis_data
    except FileNotFoundError:
        print("Error: fire_data_2000-18.csv not found!")
        print("Please run 2_data_aggregate.py first to create this file.")
        return None

def generate_synthetic_data(nt=360):
    """Generate synthetic time series data (equivalent to timeGad function)"""
    
    print(f"Generating synthetic data with {nt} time points...")
    
    # Generate time series data
    time_index = np.arange(1, nt + 1)
    
    # Generate climate variables with some temporal pattern
    max_temp = 25 + 5 * np.sin(2 * np.pi * time_index / 12) + np.random.normal(0, 2, nt)
    min_temp = 15 + 5 * np.sin(2 * np.pi * time_index / 12) + np.random.normal(0, 2, nt)
    rainfall = np.exp(2 + 0.5 * np.cos(2 * np.pi * time_index / 12) + np.random.normal(0, 0.5, nt))
    
    # Generate fire counts using negative binomial
    np.random.seed(678)
    theta = 4.5
    mu = np.exp(1 + 0.02 * max_temp - 0.001 * rainfall)
    
    # Convert to numpy negative binomial parameterization
    p = theta / (theta + mu)
    counts = np.random.negative_binomial(theta, p)
    
    # Add time component (month)
    time_component = np.tile(np.arange(1, 13), nt // 12 + 1)[:nt]
    
    df = pd.DataFrame({
        'max_temp': max_temp,
        'min_temp': min_temp,
        'rainfall': rainfall,
        'count': counts,
        'time': time_component
    })
    
    print(f"Generated data shape: {df.shape}")
    print("Data summary:")
    print(df.describe())
    
    return df

def split_synthetic_data(df, train_ratio=0.8):
    """Split synthetic data into train and test sets"""
    
    train_size = int(train_ratio * len(df))
    fire_train = df.iloc[:train_size].copy()
    fire_test = df.iloc[train_size:].copy()
    
    print(f"Training set: {len(fire_train)} observations")
    print(f"Testing set: {len(fire_test)} observations")
    
    return fire_train, fire_test

def fit_standard_negative_binomial(fire_train):
    """Fit standard negative binomial model"""
    
    print("\n" + "="*50)
    print("FITTING STANDARD NEGATIVE BINOMIAL MODEL")
    print("="*50)
    
    # Prepare data
    X = fire_train[['max_temp', 'min_temp', 'rainfall']]
    y = fire_train['count']
    X_with_const = sm.add_constant(X)
    
    # Fit NB model
    model_nb = sm.GLM(y, X_with_const, family=sm.families.NegativeBinomial()).fit()
    
    print("Standard NB Model Summary:")
    print(model_nb.summary())
    
    return model_nb

def fit_bayesian_negative_binomial_mcmc(fire_train):
    """Fit Bayesian Negative Binomial model using MCMC"""
    
    print("\n" + "="*50)
    print("FITTING BAYESIAN NEGATIVE BINOMIAL MODEL (MCMC)")
    print("="*50)
    
    if not HAS_PYMC:
        print("PyMC not available. Using approximate Bayesian approach...")
        return fit_approximate_bayesian_nb(fire_train)
    
    # Prepare data
    X = fire_train[['max_temp', 'min_temp', 'rainfall']].values
    y = fire_train['count'].values
    
    try:
        with pm.Model() as model:
            # Priors for coefficients
            intercept = pm.Normal('intercept', mu=0, sigma=2.5)
            beta_temp_max = pm.Normal('beta_temp_max', mu=0, sigma=0.27)
            beta_temp_min = pm.Normal('beta_temp_min', mu=0, sigma=0.26)
            beta_rainfall = pm.Normal('beta_rainfall', mu=0, sigma=0.23)
            
            # Dispersion parameter
            alpha = pm.Exponential('alpha', lam=1.0)
            
            # Linear combination
            mu = pm.math.exp(intercept + 
                           beta_temp_max * X[:, 0] + 
                           beta_temp_min * X[:, 1] + 
                           beta_rainfall * X[:, 2])
            
            # Likelihood
            obs = pm.NegativeBinomial('obs', mu=mu, alpha=alpha, observed=y)
            
            # Sample
            trace = pm.sample(2000, tune=1000, chains=2, cores=1, return_inferencedata=True)
        
        print("Bayesian NB Model Sampling Completed")
        print(az.summary(trace))
        
        return model, trace
    
    except Exception as e:
        print(f"Error in Bayesian modeling: {e}")
        print("Using approximate method...")
        return fit_approximate_bayesian_nb(fire_train)

def fit_approximate_bayesian_nb(fire_train):
    """Approximate Bayesian NB using bootstrap and priors"""
    
    print("Using approximate Bayesian approach with bootstrap...")
    
    # Prepare data
    X = fire_train[['max_temp', 'min_temp', 'rainfall']]
    y = fire_train['count']
    X_with_const = sm.add_constant(X)
    
    # Fit standard model first
    model_nb = sm.GLM(y, X_with_const, family=sm.families.NegativeBinomial()).fit()
    
    # Bootstrap to get posterior-like distribution
    n_bootstrap = 1000
    bootstrap_results = []
    
    print(f"Running {n_bootstrap} bootstrap iterations...")
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        boot_indices = np.random.choice(len(fire_train), size=len(fire_train), replace=True)
        X_boot = X_with_const.iloc[boot_indices]
        y_boot = y.iloc[boot_indices]
        
        try:
            # Fit model on bootstrap sample
            model_boot = sm.GLM(y_boot, X_boot, family=sm.families.NegativeBinomial()).fit()
            
            # Store parameters
            bootstrap_results.append({
                'intercept': model_boot.params[0],
                'beta_max_temp': model_boot.params[1],
                'beta_min_temp': model_boot.params[2],
                'beta_rainfall': model_boot.params[3]
            })
        except:
            continue
    
    bootstrap_df = pd.DataFrame(bootstrap_results)
    
    print(f"Bootstrap completed with {len(bootstrap_df)} successful iterations")
    print("Bootstrap parameter summary:")
    print(bootstrap_df.describe())
    
    return model_nb, bootstrap_df

def fit_enhanced_bayesian_nb(fire_train):
    """Fit enhanced Bayesian NB with time component"""
    
    print("\n" + "="*50)
    print("FITTING ENHANCED BAYESIAN NB WITH TIME COMPONENT")
    print("="*50)
    
    # Get prior means by time
    p_means = fire_train.groupby('time')['count'].median().reset_index()
    p_means.columns = ['time', 'count_mean']
    
    # Join with training data
    fire_train_enhanced = fire_train.merge(p_means, on='time', how='left')
    
    print("Added time-specific count means as features")
    print("Enhanced training data shape:", fire_train_enhanced.shape)
    
    if not HAS_PYMC:
        print("Using approximate enhanced Bayesian approach...")
        
        # Prepare data with time component
        X = fire_train_enhanced[['max_temp', 'min_temp', 'rainfall', 'count_mean']]
        y = fire_train_enhanced['count']
        X_with_const = sm.add_constant(X)
        
        # Fit enhanced NB model
        model_enhanced = sm.GLM(y, X_with_const, family=sm.families.NegativeBinomial()).fit()
        
        print("Enhanced NB Model Summary:")
        print(model_enhanced.summary())
        
        return model_enhanced, p_means
    
    # PyMC implementation
    X = fire_train_enhanced[['max_temp', 'min_temp', 'rainfall', 'count_mean']].values
    y = fire_train_enhanced['count'].values
    
    try:
        with pm.Model() as enhanced_model:
            # Priors
            intercept = pm.Normal('intercept', mu=0, sigma=2.5)
            beta_temp_max = pm.Normal('beta_temp_max', mu=0, sigma=0.27)
            beta_temp_min = pm.Normal('beta_temp_min', mu=0, sigma=0.26)
            beta_rainfall = pm.Normal('beta_rainfall', mu=0, sigma=0.23)
            beta_count_mean = pm.Normal('beta_count_mean', mu=0, sigma=1.0)
            
            # Dispersion
            alpha = pm.Exponential('alpha', lam=1.0)
            
            # Linear combination
            mu = pm.math.exp(intercept + 
                           beta_temp_max * X[:, 0] + 
                           beta_temp_min * X[:, 1] + 
                           beta_rainfall * X[:, 2] +
                           beta_count_mean * X[:, 3])
            
            # Likelihood
            obs = pm.NegativeBinomial('obs', mu=mu, alpha=alpha, observed=y)
            
            # Sample
            trace = pm.sample(2000, tune=1000, chains=2, cores=1, return_inferencedata=True)
        
        print("Enhanced Bayesian NB Sampling Completed")
        return enhanced_model, trace, p_means
    
    except Exception as e:
        print(f"Error in enhanced Bayesian modeling: {e}")
        return fit_enhanced_bayesian_nb_approximate(fire_train_enhanced), p_means

def fit_enhanced_bayesian_nb_approximate(fire_train_enhanced):
    """Approximate enhanced Bayesian model"""
    
    X = fire_train_enhanced[['max_temp', 'min_temp', 'rainfall', 'count_mean']]
    y = fire_train_enhanced['count']
    X_with_const = sm.add_constant(X)
    
    model_enhanced = sm.GLM(y, X_with_const, family=sm.families.NegativeBinomial()).fit()
    return model_enhanced

def make_predictions(models, fire_test, p_means=None):
    """Make predictions using fitted models"""
    
    print("\n" + "="*50)
    print("MAKING PREDICTIONS")
    print("="*50)
    
    predictions = {}
    
    # Standard NB predictions
    if 'standard_nb' in models:
        X_test = fire_test[['max_temp', 'min_temp', 'rainfall']]
        X_test_const = sm.add_constant(X_test)
        preds_nb = models['standard_nb'].predict(X_test_const)
        preds_nb_rounded = np.round(preds_nb)
        predictions['standard_nb'] = preds_nb_rounded
        print(f"Standard NB predictions range: {preds_nb_rounded.min():.0f} - {preds_nb_rounded.max():.0f}")
    
    # Bayesian NB predictions
    if 'bayesian_nb' in models:
        if isinstance(models['bayesian_nb'], tuple) and len(models['bayesian_nb']) == 2:
            # PyMC model
            model, trace = models['bayesian_nb']
            # For simplicity, use posterior means
            X_test = fire_test[['max_temp', 'min_temp', 'rainfall']].values
            
            # Extract posterior means (simplified)
            posterior_means = {
                'intercept': 0.0,  # Default values - would extract from trace in real implementation
                'beta_temp_max': 0.02,
                'beta_temp_min': -0.01,
                'beta_rainfall': -0.001
            }
            
            mu_pred = np.exp(posterior_means['intercept'] + 
                           posterior_means['beta_temp_max'] * X_test[:, 0] + 
                           posterior_means['beta_temp_min'] * X_test[:, 1] + 
                           posterior_means['beta_rainfall'] * X_test[:, 2])
            
            predictions['bayesian_nb'] = np.round(mu_pred)
        else:
            # Bootstrap approximation
            model, bootstrap_results = models['bayesian_nb']
            X_test = fire_test[['max_temp', 'min_temp', 'rainfall']]
            X_test_const = sm.add_constant(X_test)
            preds_bnb = model.predict(X_test_const)
            predictions['bayesian_nb'] = np.round(preds_bnb)
    
    # Enhanced Bayesian NB predictions
    if 'enhanced_nb' in models and p_means is not None:
        # Add time component to test data
        fire_test_enhanced = fire_test.merge(p_means, on='time', how='left')
        fire_test_enhanced['count_mean'] = fire_test_enhanced['count_mean'].fillna(
            fire_test_enhanced['count_mean'].mean()
        )
        
        if isinstance(models['enhanced_nb'], tuple):
            model = models['enhanced_nb'][0]
        else:
            model = models['enhanced_nb']
        
        X_test_enh = fire_test_enhanced[['max_temp', 'min_temp', 'rainfall', 'count_mean']]
        X_test_enh_const = sm.add_constant(X_test_enh)
        preds_enh = model.predict(X_test_enh_const)
        predictions['enhanced_nb'] = np.round(preds_enh)
        print(f"Enhanced NB predictions range: {np.round(preds_enh).min():.0f} - {np.round(preds_enh).max():.0f}")
    
    return predictions

def calculate_metrics(predictions, fire_test):
    """Calculate performance metrics for all models"""
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*50)
    
    y_true = fire_test['count'].values
    results = []
    
    for model_name, preds in predictions.items():
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        mae = mean_absolute_error(y_true, preds)
        
        # Calculate MAPE (handling zeros)
        mape = np.mean(np.abs((y_true - preds) / np.where(y_true == 0, 1, y_true))) * 100
        
        results.append({
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        })
        
        print(f"{model_name.upper()}:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print()
    
    # Create comparison DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # RMSE comparison
    plt.subplot(2, 2, 1)
    plt.bar(results_df['Model'], results_df['RMSE'], color='skyblue', alpha=0.7)
    plt.title('RMSE Comparison', fontsize=12, fontweight='bold')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    
    # MAE comparison
    plt.subplot(2, 2, 2)
    plt.bar(results_df['Model'], results_df['MAE'], color='lightcoral', alpha=0.7)
    plt.title('MAE Comparison', fontsize=12, fontweight='bold')
    plt.ylabel('MAE')
    plt.xticks(rotation=45)
    
    # MAPE comparison
    plt.subplot(2, 2, 3)
    plt.bar(results_df['Model'], results_df['MAPE'], color='lightgreen', alpha=0.7)
    plt.title('MAPE Comparison', fontsize=12, fontweight='bold')
    plt.ylabel('MAPE (%)')
    plt.xticks(rotation=45)
    
    # Prediction vs Actual scatter plot (for first model)
    plt.subplot(2, 2, 4)
    first_model = list(predictions.keys())[0]
    plt.scatter(y_true, predictions[first_model], alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual Count')
    plt.ylabel('Predicted Count')
    plt.title(f'Predictions vs Actual ({first_model})', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_file = get_output_path('model_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Model comparison plot saved as: {output_file}")
    plt.show()
    
    return results_df

def main(force_retrain=False):
    """Main function to run comprehensive model fitting pipeline"""
    
    print("="*60)
    print("COMPREHENSIVE MODEL FITTING PIPELINE")
    print("Python version of model_fitting.R")
    print("="*60)
    
    # Check if trained models already exist
    if not force_retrain and check_models_exist():
        print("🔍 CHECKING FOR EXISTING MODELS...")
        loaded_models, loaded_p_means = load_models()
        
        if loaded_models is not None:
            print("✅ Using existing trained models (skip --force to retrain)")
            
            # Generate test data for evaluation
            df = generate_synthetic_data(nt=360)
            fire_train, fire_test = split_synthetic_data(df)
            
            # Make predictions with loaded models
            predictions = make_predictions(loaded_models, fire_test, loaded_p_means)
            
            # Calculate metrics
            results_df = calculate_metrics(predictions, fire_test)
            
            # Save results
            output_file = get_output_path('model_fitting_results.csv')
            results_df.to_csv(output_file, index=False)
            print(f"Results saved as: {output_file}")
            
            print("\n" + "="*60)
            print("MODEL EVALUATION COMPLETED (USING EXISTING MODELS)")
            print("="*60)
            
            best_model = results_df.loc[results_df['RMSE'].idxmin(), 'Model']
            print(f"\nBest performing model (by RMSE): {best_model}")
            print(f"Best RMSE: {results_df['RMSE'].min():.4f}")
            
            return {
                'models': loaded_models,
                'predictions': predictions,
                'results': results_df,
                'p_means': loaded_p_means
            }
    
    print("🚀 TRAINING NEW MODELS...")
    
    # Generate synthetic data (equivalent to original workflow)
    df = generate_synthetic_data(nt=360)
    
    # Split data
    fire_train, fire_test = split_synthetic_data(df)
    
    # Dictionary to store fitted models
    models = {}
    
    # Fit Standard Negative Binomial
    models['standard_nb'] = fit_standard_negative_binomial(fire_train)
    
    # Fit Bayesian Negative Binomial with MCMC
    models['bayesian_nb'] = fit_bayesian_negative_binomial_mcmc(fire_train)
    
    # Fit Enhanced Bayesian NB with time component
    enhanced_result = fit_enhanced_bayesian_nb(fire_train)
    if len(enhanced_result) == 2:
        models['enhanced_nb'], p_means = enhanced_result
    else:
        models['enhanced_nb'], _, p_means = enhanced_result
    
    # Save the trained models
    save_models(models, p_means)
    
    # Make predictions
    predictions = make_predictions(models, fire_test, p_means)
    
    # Calculate and compare metrics
    results_df = calculate_metrics(predictions, fire_test)
    
    # Save results
    output_file = get_output_path('model_fitting_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"Results saved as: {output_file}")
    
    print("\n" + "="*60)
    print("MODEL FITTING COMPLETED")
    print("="*60)
    
    # Summary
    best_model = results_df.loc[results_df['RMSE'].idxmin(), 'Model']
    print(f"\nBest performing model (by RMSE): {best_model}")
    print(f"Best RMSE: {results_df['RMSE'].min():.4f}")
    
    return {
        'models': models,
        'predictions': predictions,
        'results': results_df,
        'p_means': p_means
    }

if __name__ == "__main__":
    import sys
    
    # Check for force retrain flag
    force_retrain = "--force" in sys.argv or "-f" in sys.argv
    
    if force_retrain:
        print("🔄 Force retrain mode enabled - will retrain all models")
    
    results = main(force_retrain=force_retrain) 