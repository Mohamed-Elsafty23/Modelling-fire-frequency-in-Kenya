# Modelling-fire-frequency-in-Kenya

This repository contains the data files and code used in the publication titled "A Bayesian Model for Predicting Fire Frequency in Kenya." The objective of the study was to create and evaluate a Bayesian model to predict fire frequency in Kenya. The repository includes code used to preprocess the data and combine it into the final form. It also includes simulation scripts used to generate the datasets used in the study.

## Repository Structure

The repository is organized into the following folders:

- **modis_data**: This folder contains CSV files of daily fire hotspot data downloaded from MODIS. The data spans the period from 2000 to 2020.

- **fire**: This folder contains CSV files of fire frequency by month from 2000 to 2018.

- **rainfall**: This folder contains GeoTIFF files of precipitation data in Kenya from 2000 to 2018.

- **tmin**: This folder contains GeoTIFF files of minimum temperature data in Kenya from 2000 to 2018.

- **tmax**: This folder contains GeoTIFF files of maximum temperature data in Kenya from 2000 to 2018.

- **climate**: This folder contains merged CSV files of rainfall, fire, minimum temperature, and maximum temperature data matched by coordinates. The merged data is used for analysis.

- **simulated_data**: This folder contains CSV files of data simulated during the study. These simulated datasets are used for model evaluation.

- **model_results**: This folder contains model results on the simulated datasets. The results provide insights into the performance of the Bayesian model.

## Code Scripts

The repository also contains several R scripts that perform various tasks:

- **1import_merge**: This script contains code to import and merge climate and fire data. It processes the raw data files and combines them into a format suitable for further analysis.

- **2data_aggregate**: This script merges data in the climate folder into a single file to be used in the analysis. It aggregates the data from different sources and prepares it for modeling.

- **3check_seasonality**: This script checks for seasonality in the data. It performs seasonal analysis to understand patterns and trends in fire frequency.

- **5simulation_temp**: This script runs simulations based on specified parameters. It generates simulated datasets to evaluate the performance of the Bayesian model.

- **6final_models**: This script contains the formulation of the models used in the study. It defines the Bayesian model and sets up the necessary priors and likelihoods.

- **7models_years**: This folder contains four scripts that implement the models on four different time periods. Each script applies the Bayesian model to the respective dataset and generates predictions.

- **8descriptive_stats**: This script conducts data analysis and descriptive statistics on the real aggregated data. It provides insights into the characteristics of the data and helps in understanding the variables.

- **9sim_visualization**: This script contains code to create graphs from the simulation model results. It visualizes the simulation outputs, making it easier to interpret the results.

## Publication Objective

The objective of the publication was to create and evaluate a Bayesian model for predicting fire frequency in Kenya. The study used data from MODIS, which provides daily fire hotspot information, and climate data including rainfall, minimum temperature, and maximum temperature. By merging and analyzing these datasets, the study aimed to develop a model that could predict fire frequency in Kenya based on climate variables.

This repository serves as a comprehensive resource for replicating the study's results and exploring the code and data used in the analysis. You can use the provided scripts and datasets to further investigate fire frequency patterns in Kenya or extend the research to other regions. Feel free to explore the code and adapt it to your specific needs.

Please refer to the upcoming publication for detailed information on the methodology, results, and conclusions of the study.
