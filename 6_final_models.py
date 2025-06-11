from output_utils import get_output_path, get_model_results_path, get_simulated_data_path, ensure_output_dirs
#!/usr/bin/env python3
"""
Model definitions - Python version of 6final_models.R
Defines standard and Bayesian negative binomial models for fire frequency prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy.optimize import minimize
import pymc as pm
import arviz as az
import warnings
warnings.filterwarnings('ignore')

class NegativeBinomialModel:
    """Standard Negative Binomial GLM model"""
    
    def __init__(self, random_state=456):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y, train_ratio=0.8):
        """
        Fit negative binomial model
        
        Parameters:
        - X: Features (max_temp, rainfall)
        - y: Target (fire count)
        - train_ratio: Proportion for training set
        """
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, train_size=train_ratio, random_state=self.random_state, shuffle=False
        )
        
        # Fit model using statsmodels
        try:
            # Add constant for intercept
            X_train_const = sm.add_constant(self.X_train)
            
            # Fit negative binomial GLM
            self.model = sm.GLM(
                self.y_train, 
                X_train_const, 
                family=sm.families.NegativeBinomial()
            ).fit()
            
            return True
            
        except Exception as e:
            print(f"Error fitting NB model: {e}")
            # Fallback to Poisson if NB fails
            self.model = sm.GLM(
                self.y_train, 
                X_train_const, 
                family=sm.families.Poisson()
            ).fit()
            return False
    
    def predict(self, X=None, return_train=False):
        """Make predictions on test set or provided data"""
        if X is None:
            X = self.X_test if not return_train else self.X_train
        
        X_const = sm.add_constant(X)
        
        # Handle missing constant column
        if X_const.shape[1] != self.model.params.shape[0]:
            if 'const' not in X_const.columns:
                X_const.insert(0, 'const', 1.0)
        
        predictions = self.model.predict(X_const)
        return predictions
    
    def evaluate(self, theta=1.5, n=60):
        """Evaluate model performance"""
        # Training predictions
        pred_train = self.predict(return_train=True)
        rmse_train = np.sqrt(mean_squared_error(self.y_train, pred_train))
        
        # Test predictions  
        pred_test = self.predict()
        rmse_test = np.sqrt(mean_squared_error(self.y_test, pred_test))
        
        # MASE (Mean Absolute Scaled Error)
        naive_forecast = np.mean(self.y_train)
        mae_naive = mean_absolute_error(self.y_test, [naive_forecast] * len(self.y_test))
        mae_model = mean_absolute_error(self.y_test, pred_test)
        mase_test = mae_model / mae_naive if mae_naive > 0 else np.inf
        
        # Bias
        bias_test = np.mean(pred_test - self.y_test)
        
        return {
            'rmse_train': rmse_train,
            'rmse_test': rmse_test,
            'mase_test': mase_test,
            'bias_test': bias_test,
            'theta': theta,
            'n': n
        }

class BayesianNegativeBinomialModel:
    """Bayesian Negative Binomial model with seasonal components"""
    
    def __init__(self, random_state=456):
        self.random_state = random_state
        self.model = None
        self.trace = None
        
    def prepare_seasonal_features(self, data, n_months):
        """Add seasonal features to data"""
        data = data.copy()
        
        # Add time index (monthly)
        data['time'] = np.tile(np.arange(1, 13), n_months // 12 + 1)[:n_months]
        
        # Calculate prior means by month
        monthly_means = data.groupby('time')['count'].mean()
        
        # Add seasonal trigonometric features
        data['costhet'] = np.cos(2 * np.pi * data['time'] / 12)
        data['sinthet'] = np.sin(2 * np.pi * data['time'] / 12)
        
        # Merge monthly means
        data = data.merge(
            monthly_means.rename('count_mean').reset_index(), 
            on='time', how='left'
        )
        
        return data
    
    def fit(self, X, y, train_ratio=0.8, n_months=60):
        """
        Fit Bayesian negative binomial model
        
        Parameters:
        - X: Features (max_temp, rainfall)  
        - y: Target (fire count)
        - train_ratio: Proportion for training set
        - n_months: Total number of months in dataset
        """
        # Prepare data with seasonal features
        data = pd.DataFrame(X, columns=['max_temp', 'rainfall'])
        data['count'] = y
        data = self.prepare_seasonal_features(data, n_months)
        
        # Split data
        train_size = int(len(data) * train_ratio)
        self.train_data = data.iloc[:train_size].copy()
        self.test_data = data.iloc[train_size:].copy()
        
        # Fit Bayesian model
        try:
            with pm.Model() as self.model:
                # Priors for regression coefficients
                alpha = pm.Normal('alpha', mu=0, sigma=2)  # Intercept
                beta_temp = pm.Normal('beta_temp', mu=0, sigma=1)  # Temperature effect
                beta_rain = pm.Normal('beta_rain', mu=0, sigma=1)  # Rainfall effect
                beta_cos = pm.Normal('beta_cos', mu=0, sigma=1)   # Cosine seasonal
                beta_sin = pm.Normal('beta_sin', mu=0, sigma=1)   # Sine seasonal
                
                # Dispersion parameter
                theta = pm.Exponential('theta', lam=1)
                
                # Linear combination
                mu = pm.math.exp(
                    alpha + 
                    beta_temp * self.train_data['max_temp'] + 
                    beta_rain * self.train_data['rainfall'] +
                    beta_cos * self.train_data['costhet'] +
                    beta_sin * self.train_data['sinthet']
                )
                
                # Likelihood
                obs = pm.NegativeBinomial('obs', mu=mu, alpha=theta, observed=self.train_data['count'])
                
                # Sample
                self.trace = pm.sample(
                    1000, 
                    tune=500, 
                    random_seed=self.random_state,
                    progressbar=False,
                    return_inferencedata=True
                )
                
            return True
            
        except Exception as e:
            print(f"Error fitting Bayesian model: {e}")
            return False
    
    def predict(self, return_train=False):
        """Make predictions using posterior samples"""
        if self.trace is None:
            raise ValueError("Model not fitted yet")
        
        data = self.train_data if return_train else self.test_data
        
        # Get posterior samples
        posterior = self.trace.posterior
        
        # Calculate predictions for each posterior sample
        predictions = []
        n_samples = len(posterior.chain) * len(posterior.draw)
        
        # Take subset of samples for efficiency
        sample_indices = np.random.choice(n_samples, min(100, n_samples), replace=False)
        
        for i in sample_indices:
            chain_idx = i // len(posterior.draw)
            draw_idx = i % len(posterior.draw)
            
            alpha = float(posterior['alpha'][chain_idx, draw_idx])
            beta_temp = float(posterior['beta_temp'][chain_idx, draw_idx])
            beta_rain = float(posterior['beta_rain'][chain_idx, draw_idx])
            beta_cos = float(posterior['beta_cos'][chain_idx, draw_idx])
            beta_sin = float(posterior['beta_sin'][chain_idx, draw_idx])
            
            mu = np.exp(
                alpha + 
                beta_temp * data['max_temp'] + 
                beta_rain * data['rainfall'] +
                beta_cos * data['costhet'] +
                beta_sin * data['sinthet']
            )
            
            predictions.append(mu.values)
        
        # Return mean prediction across samples
        return np.mean(predictions, axis=0)
    
    def evaluate(self, theta=1.5, n=60):
        """Evaluate model performance"""
        # Training predictions
        pred_train = self.predict(return_train=True)
        rmse_train = np.sqrt(mean_squared_error(self.train_data['count'], pred_train))
        
        # Test predictions
        pred_test = self.predict()
        rmse_test = np.sqrt(mean_squared_error(self.test_data['count'], pred_test))
        
        # MASE
        naive_forecast = np.mean(self.train_data['count'])
        mae_naive = mean_absolute_error(self.test_data['count'], [naive_forecast] * len(self.test_data))
        mae_model = mean_absolute_error(self.test_data['count'], pred_test)
        mase_test = mae_model / mae_naive if mae_naive > 0 else np.inf
        
        # Bias
        bias_test = np.mean(pred_test - self.test_data['count'])
        
        return {
            'rmse_train': rmse_train,
            'rmse_test': rmse_test,
            'mase_test': mase_test,
            'bias_test': bias_test,
            'theta': theta,
            'n': n
        }

def negbinner(data, theta=1.5, n=60):
    """
    Standard negative binomial model function
    Python equivalent of negbinner() in R
    """
    try:
        # Prepare features
        X = data[['max_temp', 'rainfall']].values
        y = data['count'].values
        
        # Fit model
        model = NegativeBinomialModel()
        success = model.fit(X, y)
        
        if success:
            results = model.evaluate(theta, n)
            return results
        else:
            return {
                'rmse_train': np.nan, 'rmse_test': np.nan,
                'mase_test': np.nan, 'bias_test': np.nan,
                'theta': theta, 'n': n
            }
            
    except Exception as e:
        print(f"Error in negbinner: {e}")
        return {
            'rmse_train': np.nan, 'rmse_test': np.nan,
            'mase_test': np.nan, 'bias_test': np.nan,
            'theta': theta, 'n': n
        }

def stanbinner(data, theta=1.5, n=60):
    """
    Bayesian negative binomial model function
    Python equivalent of stanbinner() in R
    """
    try:
        # Prepare features
        X = data[['max_temp', 'rainfall']].values
        y = data['count'].values
        
        # Fit model
        model = BayesianNegativeBinomialModel()
        success = model.fit(X, y, n_months=n)
        
        if success:
            results = model.evaluate(theta, n)
            return results
        else:
            return {
                'rmse_train': np.nan, 'rmse_test': np.nan,
                'mase_test': np.nan, 'bias_test': np.nan,
                'theta': theta, 'n': n
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
        'count': np.random.negative_binomial(5, 0.3, n_obs)
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