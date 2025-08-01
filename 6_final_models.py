"""
Model definitions - Python version of 6final_models.R
Defines standard and Bayesian negative binomial models for fire frequency prediction
"""

from output_utils import get_output_path, get_model_results_path, get_simulated_data_path, ensure_output_dirs
#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
import pymc as pm
import warnings
import sys
import io
import contextlib
warnings.filterwarnings('ignore')

def negbinner(x, theta=1.5, n=60):
    """
    Standard negative binomial model function
    """
    try:
        # Set seed
        np.random.seed(456)
        
        # Create training and test sets like R
        trainIndex = int(np.round(0.8 * len(x)))
        fireTrain = x.iloc[:trainIndex].copy()
        fireTest = x.iloc[trainIndex:].copy()
        
        # Fit model on training set like R
        # R code: glmNB <- MASS::glm.nb(count ~ max_temp + rainfall, data = fireTrain, link = "log")
        X_train = fireTrain[['max_temp', 'rainfall']]
        y_train = fireTrain['count']
        X_train_const = sm.add_constant(X_train)
        
        # fit negative binomial model that estimates the dispersion
        # suppress optimization messages
        with contextlib.redirect_stdout(io.StringIO()):
            glmNB = NegativeBinomial(y_train, X_train_const).fit(disp=0)
        
        # Predict on training set
        predictions_train = glmNB.predict(X_train_const)
        
        # Predict on testing set 
        X_test = fireTest[['max_temp', 'rainfall']]
        y_test = fireTest['count']
        X_test_const = sm.add_constant(X_test)
        predictions_test = glmNB.predict(X_test_const)
        
        # Get the RMSE
        train_rmse = np.sqrt(mean_squared_error(y_train, np.round(predictions_train)))
        test_rmse = np.sqrt(mean_squared_error(y_test, np.round(predictions_test)))
        
        # Get the MASE
        mae_naive = np.mean(np.abs(np.diff(y_train)))
        mae_model = mean_absolute_error(y_test, np.round(predictions_test))
        test_mase = mae_model / mae_naive if mae_naive > 0 else np.inf
        
        # Get the bias
        test_bias = np.mean((y_test - np.round(predictions_test))/np.abs(y_test))

        # Return metrics
        return {
            'rmse_train': train_rmse,
            'rmse_test': test_rmse, 
            'mase_test': test_mase,
            'bias_test': test_bias,
            'theta': theta,
            'n': n
            }
            
    except Exception as e:
        print(f"Error in negbinner: {e}")
        return {
            'rmse_train': np.nan, 'rmse_test': np.nan,
            'mase_test': np.nan, 'bias_test': np.nan,
            'theta': theta, 'n': n
        }

def stanbinner(x, theta=1.5, n=60):
    """
    Bayesian negative binomial model function
    """
    try:
        # Set seed
        np.random.seed(456)
        
        # Use the month column of the dataset as time variable (instead of a continuous time index like in R)
        x = x.copy()
        x['time'] = x['month']
        
        # Create training and test sets 
        trainIndex = int(np.round(0.8 * len(x)))
        fireTrain = x.iloc[:trainIndex].copy()
        fireTest = x.iloc[trainIndex:].copy()
        
        # Get prior means
        def get_prior_means(x):
            
            monthly_stats = x.groupby('time')['count'].mean().reset_index()
            monthly_stats.columns = ['time', 'count_mean']
            
            # Add CORRECTED seasonal features
            monthly_stats['costhet'] = np.cos((2 * np.pi) * monthly_stats['time'] / 12)
            monthly_stats['sinthet'] = np.sin((2 * np.pi) * monthly_stats['time'] / 12)
            
            return monthly_stats
        
        # R code: p_means = get_prior_means(fireTrain)
        p_means = get_prior_means(fireTrain)
        
        # Join means to train data
        fireTrain2 = fireTrain.merge(p_means, on='time', how='inner')
        
        # Join means to test data
        fireTest2 = fireTest.merge(p_means, on='time', how='inner')

        # Standardize predictors
        for col in ['max_temp', 'rainfall', 'costhet', 'sinthet']:
            mean = fireTrain2[col].mean()
            std = fireTrain2[col].std()
            fireTrain2[col] = (fireTrain2[col] - mean) / std
            fireTest2[col] = (fireTest2[col] - mean) / std
        
        # Define and fit Bayesian Negative Binomial model using PyMC
        with pm.Model() as model:

            # Priors (rstanarm defaults)
            alpha = pm.Normal('alpha', mu=0, sigma=2.5)
            beta_temp = pm.Normal('beta_temp', mu=0, sigma=2.5)
            beta_rain = pm.Normal('beta_rain', mu=0, sigma=2.5)
            beta_cos = pm.Normal('beta_cos', mu=0, sigma=2.5)
            beta_sin = pm.Normal('beta_sin', mu=0, sigma=2.5)
            reciprocal_dispersion = pm.Exponential('reciprocal_dispersion', lam=1)
            alpha_nb = pm.Deterministic('alpha_nb', 1.0 / reciprocal_dispersion)

            # Linear predictor
            mu = pm.math.exp(
                alpha +
                beta_temp * fireTrain2['max_temp'].values +
                beta_rain * fireTrain2['rainfall'].values +
                beta_cos * fireTrain2['costhet'].values +
                beta_sin * fireTrain2['sinthet'].values
            )

            # Likelihood
            obs = pm.NegativeBinomial('obs', mu=mu, alpha=alpha_nb, observed=fireTrain2['count'].values)

            # Sample
            trace = pm.sample(500, tune=250, random_seed=456, progressbar=False, return_inferencedata=True,
                              target_accept=0.9, chains=4, cores=1)

        # Posterior predictions
        def predict_posterior(data, trace, n_samples=100):
            posterior = trace.posterior
            samples = []
            total_samples = posterior.sizes['chain'] * posterior.sizes['draw']
            idx = np.random.choice(total_samples, min(n_samples, total_samples), replace=False)

            # Flatten posterior dimensions
            alpha_vals = posterior['alpha'].stack(sample=("chain", "draw")).values
            beta_temp_vals = posterior['beta_temp'].stack(sample=("chain", "draw")).values
            beta_rain_vals = posterior['beta_rain'].stack(sample=("chain", "draw")).values
            beta_cos_vals = posterior['beta_cos'].stack(sample=("chain", "draw")).values
            beta_sin_vals = posterior['beta_sin'].stack(sample=("chain", "draw")).values

            for i in idx:
                mu_i = np.exp(
                    alpha_vals[i] +
                    beta_temp_vals[i] * data['max_temp'].values +
                    beta_rain_vals[i] * data['rainfall'].values +
                    beta_cos_vals[i] * data['costhet'].values +
                    beta_sin_vals[i] * data['sinthet'].values
                )
                samples.append(mu_i)

            return np.mean(samples, axis=0)

        pred_train = predict_posterior(fireTrain2, trace)
        pred_test = predict_posterior(fireTest2, trace)

        # Evaluation metrics
        # RMSE
        train_rmse = np.sqrt(mean_squared_error(fireTrain2['count'], np.round(pred_train)))
        test_rmse = np.sqrt(mean_squared_error(fireTest2['count'], np.round(pred_test)))

        # MASE
        mae_naive = np.mean(np.abs(np.diff(fireTrain2['count'])))
        mae_model = mean_absolute_error(fireTest2['count'], np.round(pred_test))
        test_mase = mae_model / mae_naive if mae_naive > 0 else np.inf

        # Bias
        test_bias = np.mean((fireTest2['count'] - np.round(pred_test))/np.abs(fireTest2['count']))

        # Return metrics 
        return {
            'rmse_train': train_rmse,
            'rmse_test': test_rmse,
            'mase_test': test_mase,
            'bias_test': test_bias,
            'theta': theta,
            'n': n
        }
            
    except Exception as e:
        print(f"Error in stanbinner: {e}")
        return {
            'rmse_train': np.nan, 'rmse_test': np.nan,
            'mase_test': np.nan, 'bias_test': np.nan,
            'theta': theta, 'n': n
        }

if __name__ == "__main__":
    # Test the models with sample data
    print("Testing model implementations...")
    
    # Create sample data
    np.random.seed(456)
    n_obs = 60
    sample_data = pd.DataFrame({
        'max_temp': np.random.normal(29, 2, n_obs),
        'rainfall': np.random.gamma(2, 30, n_obs),
        'count': np.random.negative_binomial(5, 0.3, n_obs),
        'month': np.tile(np.arange(1, 13), n_obs // 12 + 1)[:n_obs]  # cyclic month 1-12
    })
    
    # Test standard model
    print("\nTesting Standard Negative Binomial Model...")
    nb_results = negbinner(sample_data, theta=1.5, n=60)
    print(f"Results: {nb_results}")
    
    # Test Bayesian model
    print("\nTesting Bayesian Negative Binomial Model...")
    bayesian_results = stanbinner(sample_data, theta=1.5, n=60)
    print(f"Results: {bayesian_results}")
    
    print("\nModel testing completed!") 