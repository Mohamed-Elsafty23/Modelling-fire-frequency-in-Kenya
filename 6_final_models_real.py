"""
Defines standard and Bayesian negative binomial models for fire frequency prediction
and applies the models to real dataset
"""

from output_utils import get_output_path, get_model_results_path, get_simulated_data_path, ensure_output_dirs
#!/usr/bin/env python3
import openpyxl
#import pytensor
#pytensor.config.cxx = "/usr/bin/clang++" # remove hashtag to fix pytensor issue on Mac
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
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
warnings.filterwarnings('ignore')

def negbinner(x, theta=1.5, n=60):
    """
    Standard negative binomial model function
    """
    try:
        # Set seed
        np.random.seed(456)
        
        # Create training and test sets
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

        # Return metrics and predicted values
        return {
            'rmse_train': train_rmse,
            'rmse_test': test_rmse, 
            'mase_test': test_mase,
            'bias_test': test_bias,
            'theta': theta,
            'n': n,
            'y_true_train': y_train.values,
            'y_pred_train': predictions_train,
            'y_true_test': y_test.values,
            'y_pred_test': predictions_test,
            'index_train': fireTrain.index,
            'index_test': fireTest.index
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
        
        p_means = get_prior_means(fireTrain)
        
        # Join means to train data
        fireTrain2_index = fireTrain.index
        fireTrain2 = fireTrain.merge(p_means, on='time', how='inner')
        fireTrain2.index = fireTrain2_index
        
        # Join means to test data
        fireTest2_index = fireTest.index
        fireTest2 = fireTest.merge(p_means, on='time', how='inner')
        fireTest2.index = fireTest2_index

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

            # Compute mean and 95% prediction interval
            mean = np.mean(samples, axis=0)
            lower = np.percentile(samples, 2.5, axis=0)
            upper = np.percentile(samples, 97.5, axis=0)

            return mean, lower, upper
        
        pred_train, pred_train_lower, pred_train_upper = predict_posterior(fireTrain2, trace)
        pred_test, pred_test_lower, pred_test_upper = predict_posterior(fireTest2, trace)

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

        # Get estimates and diagnostics
        summary_df = az.summary(trace, hdi_prob=0.95)
        print(summary_df)

        # Save to Excel
        # summary_df.to_excel("our_output/estimates_real.xlsx", index=True)

        # Traceplot
        az.plot_trace(trace, figsize=(8, 10))
        plt.suptitle("Traceplot", fontsize=12)
        plt.tight_layout()
        #plt.savefig("our_output/diagnostics/traceplot.png", dpi=300)
        plt.close()

        # Posterior distributions
        az.plot_posterior(trace, figsize=(10, 8))
        plt.suptitle("Posterior Distributions", fontsize=12)
        plt.tight_layout()
        #plt.savefig("our_output/diagnostics/posterior_distributions.png", dpi=300)
        plt.close()

        # Forest plot with R-hat
        az.plot_forest(trace, figsize=(8, 4), combined=True, hdi_prob=0.95, r_hat=True)
        plt.suptitle("Forest Plot with R-hat", fontsize=12)
        plt.tight_layout()
        #plt.savefig("our_output/diagnostics/forest_plot.png", dpi=300)
        plt.close()

        # Energy plot
        az.plot_energy(trace, figsize=(6, 4))
        plt.title("Energy Plot", fontsize=12)
        plt.tight_layout()
        #plt.savefig("our_output/diagnostics/energy_plot.png", dpi=300)
        plt.close()

        # Autocorrelation plot
        az.plot_autocorr(
            trace,
            var_names=["alpha", "beta_temp", "beta_rain", "beta_cos", "beta_sin"],
            combined=True,
            figsize=(10, 6)
        )
        plt.suptitle("Autocorrelation", fontsize=12)
        plt.tight_layout()
        #plt.savefig("our_output/diagnostics/autocorrelation_plot.png", dpi=300)
        plt.close()

        # Return metrics and predicted values
        return {
            'rmse_train': train_rmse,
            'rmse_test': test_rmse,
            'mase_test': test_mase,
            'bias_test': test_bias,
            'theta': theta,
            'n': n,
            'y_true_train': fireTrain2['count'].values,
            'y_pred_train': pred_train,
            'y_lower_train': pred_train_lower,
            'y_upper_train': pred_train_upper,
            'y_true_test': fireTest2['count'].values,
            'y_pred_test': pred_test,
            'y_lower_test': pred_test_lower,
            'y_upper_test': pred_test_upper,
            'index_train': fireTrain2.index,
            'index_test': fireTest2.index
        }
            
    except Exception as e:
        print(f"Error in stanbinner: {e}")
        return {
            'rmse_train': np.nan, 'rmse_test': np.nan,
            'mase_test': np.nan, 'bias_test': np.nan,
            'theta': theta, 'n': n
        }

# Combine predictions and actuals
def build_df(result, label, data):
    # Match index to actual timestamps
    full_data = data.copy().reset_index(drop=True)

    df_train = pd.DataFrame({
        'index': result['index_train'],
        'actual': result['y_true_train'],
        'predicted': result['y_pred_train'],
        'lower': result.get('y_lower_train', np.nan),
        'upper': result.get('y_upper_train', np.nan),
        'set': 'train',
        'model': label,
        'year': full_data.loc[result['index_train'], 'year'].values,
        'month': full_data.loc[result['index_train'], 'month'].values
    })
    df_test = pd.DataFrame({
        'index': result['index_test'],
        'actual': result['y_true_test'],
        'predicted': result['y_pred_test'],
        'lower': result.get('y_lower_test', np.nan),
        'upper': result.get('y_upper_test', np.nan),
        'set': 'test',
        'model': label,
        'year': full_data.loc[result['index_test'], 'year'].values,
        'month': full_data.loc[result['index_test'], 'month'].values
    })
    df = pd.concat([df_train, df_test])
    df['date'] = pd.to_datetime(dict(year=df['year'], month=df['month'], day=1))
    return df.sort_values('date')

if __name__ == "__main__":
    # Define path to the input dataset (real data)
    data_path = "fire_data_2000-18.csv"

    # Load dataset
    data = pd.read_csv(data_path)
    data = data.rename(columns={"mean_max_temp":"max_temp", "mean_rainfall":"rainfall"})
    print(f"Loaded data from {data_path}. Shape: {data.shape}")
    print(data.head())

    # Run standard negative binomial model
    print("\nRunning Standard Negative Binomial Model...")
    nb_result = negbinner(data, theta=1.5, n=60)

    # Run Bayesian negative binomial model
    print("\nRunning Bayesian Negative Binomial Model...")
    bayes_result = stanbinner(data, theta=1.5, n=60)

    # Combine both results into a DataFrame
    results_df = pd.DataFrame([nb_result, bayes_result], index=["Standard NB", "Bayesian NB"])
    results_df = results_df.loc[:, ["rmse_train", "rmse_test", "mase_test", "bias_test"]].T
    print("\nModel Results:")
    print(results_df)

    # Save to Excel
    # results_df.to_excel("our_output/metrics_real.xlsx", index=True)

    # Build df for plotting
    df_nb = build_df(nb_result, 'Standard NB', data)
    df_bayes = build_df(bayes_result, 'Bayesian NB', data)
    plot_df = pd.concat([df_nb, df_bayes])

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Create a lineplot of predicted vs observed for BNB and NB model
    for ax, model_name in zip(axes, ['Bayesian NB', 'Standard NB']):
        sub_df = plot_df[(plot_df['model'] == model_name) & (plot_df['set'] == 'test')] # only consider test data

        # Plot actual and predicted
        ax.plot(sub_df['date'], sub_df['actual'], label='Actual', color='black', linewidth=1.5)
        ax.plot(sub_df['date'], sub_df['predicted'], label='Predicted', color='steelblue', linestyle='--')

        # Plot prediction interval if present
        if 'lower' in sub_df and 'upper' in sub_df:
            ax.fill_between(
                sub_df['date'],
                sub_df['lower'],
                sub_df['upper'],
                color='steelblue',
                alpha=0.2,
                label='95% Prediction Interval'
            )

        # Train/test split marker
        #split_date = sub_df[sub_df['set'] == 'train']['date'].max()
        #ax.axvline(split_date, color='grey', linestyle=':', label='Train/Test Split')

        # Axis settings
        ax.set_title(f'{model_name} Predictions', fontsize=20)
        ax.set_xlabel('Date', fontsize=18)
        ax.set_ylabel('Count', fontsize=18)

        # Format x-axis to show every 2nd year and rotate labels
        ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=45, labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        
        # Add legend
        ax.legend(fontsize=16, title_fontsize=16)

    plt.tight_layout()
    #plt.savefig("our_output/diagnostics/predictions_year_test.png", dpi=300)
    plt.show()
