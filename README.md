# Fire Frequency Modeling in Kenya

A comprehensive Bayesian modeling framework for predicting fire frequency in Kenya using climate variables. This project implements both standard and Bayesian negative binomial models to analyze the relationship between climate factors (temperature, rainfall) and fire occurrence patterns.

## 🎯 Project Overview

This repository contains a complete data science pipeline that:
- Processes MODIS fire hotspot data and climate datasets
- Applies spatial categorization to improve model performance
- Implements both frequentist and Bayesian statistical models
- Generates synthetic data for model validation
- Provides comprehensive model comparison and visualization tools

## 📋 Prerequisites

### Required Software
- Python 3.8 or higher
- Git (for cloning the repository)

### Python Dependencies
Install all required packages using:
```bash
pip install -r requirements.txt
```

**Key dependencies include:**
- `pandas`, `numpy`, `scipy` - Data manipulation and analysis
- `geopandas`, `rasterio` - Geospatial data processing
- `pymc`, `arviz` - Bayesian modeling
- `scikit-learn`, `statsmodels` - Statistical modeling
- `matplotlib`, `seaborn` - Data visualization

## 📁 Data Structure

Ensure your data is organized in the following folders:
- `modis_data/` - MODIS fire hotspot CSV files (2000-2020)
- `fire/` - Monthly fire frequency CSV files (2000-2018)
- `rain/` - Precipitation TIFF files (2000-2018)
- `tmin/` - Minimum temperature TIFF files (2000-2018)
- `tmax/` - Maximum temperature TIFF files (2000-2018)

## 🚀 How to Run the Project

Follow these steps in order to reproduce the complete analysis:

### Step 1: Data Import and Merging
```bash
python 1_import_merge.py
```
**Purpose:** Imports MODIS fire data and merges with climate data (rainfall, tmax, tmin). Creates climate data CSV files in separate folders.

**Output:** 
- `climate_tmax_csv/`, `climate_tmin_csv/`, `climate_rainfall_csv/` folders
- Climate data stored as CSV files for efficient processing

### Step 2: Data Aggregation
```bash
python 2_data_aggregate.py
```
**Purpose:** Merges all climate and fire data into a single analysis-ready dataset.

**Output:** 
- `fire_data_2000-18.csv` - Main dataset for analysis

### Step 3: Seasonality Analysis
```bash
python 3_check_seasonality.py
```
**Purpose:** Uses TBATS model to detect seasonal patterns in climate and fire data.

**Output:** 
- Seasonality plots in `our_output/plots/`
- Statistical summaries of seasonal components

### Step 4: Location Categorization
```bash
python categorize_locations.py
```
**Purpose:** Categorizes geographic locations into quadrants based on Kenya's geographic center for improved spatial modeling.

**Output:** 
- `fire_data_2000-18_categorized.csv` - Dataset with spatial categories
- Visualization maps showing quadrant divisions

### Step 5: Generate Simulated Data
```bash
python 5_simulation_temp.py
```
**Purpose:** Creates synthetic datasets with different dispersion parameters (theta) and time periods for model validation.

**Output:** 
- `our_output/simulated_data/` - Simulated datasets organized by time period and theta values
- Various combinations: 5, 10, 20, 30 years with theta values 1.5, 5, 10, 100

### Step 6A: Real Data Modeling (Original)
```bash
python 6_final_models_real.py
```
**Purpose:** Applies standard and Bayesian negative binomial models to the original real dataset.

**Output:** 
- Model results, diagnostics, and estimates in `our_output/tables/`
- Statistical model summaries and performance metrics

### Step 6B: Real Data Modeling (Categorized)
```bash
python 6_final_models_real_categorized.py
```
**Purpose:** Applies models to the spatially categorized dataset for comparison.

**Output:** 
- Categorized model results and comparisons
- Improved model performance metrics

### Step 7: Comprehensive Model Evaluation
```bash
python run_all_models.py
```
**Purpose:** Runs all models on simulated datasets. This is computationally intensive and may take several hours.

**Features:**
- GPU acceleration support (if available)
- Parallel processing optimization
- Progress tracking and file-based checkpointing
- Configurable time periods and theta values

**Output:** 
- `our_output/model_results_500/` - Comprehensive model evaluation results
- Performance metrics for all model-dataset combinations

### Step 8: Model Comparison
```bash
python compare_categorized_vs_original.py
```
**Purpose:** Compares performance between categorized and original modeling approaches.

**Output:** 
- `comparison_results/` - Side-by-side performance comparisons
- Improvement analysis and visualization

### Step 9: Results Visualization
```bash
python visualize_simulations.py
```
**Purpose:** Creates comprehensive visualizations of simulation results.

**Output:** 
- Summary tables and performance plots
- Model comparison visualizations

### Step 10: Temporal Analysis Plots
```bash
python scatterplots_(monthly_yearly).py
```
**Purpose:** Generates detailed scatterplots showing temporal relationships between climate variables and fire frequency.

**Output:** 
- Temporal trend visualizations
- Decade-wise comparison plots

## 📊 Key Outputs

After running the complete pipeline, you'll find:

- **`our_output/tables/`** - Excel files with model estimates, diagnostics, and metrics
- **`our_output/plots/`** - Visualization plots and charts
- **`our_output/model_results_500/`** - Comprehensive model evaluation results
- **`comparison_results/`** - Performance comparison between approaches

## ⚙️ Configuration Options

### Computational Performance
- **GPU Support:** Automatically detected and enabled if available
- **Parallel Processing:** Optimized for multi-core systems
- **Memory Management:** Efficient handling of large datasets

### Model Parameters
Edit `run_all_models.py` to customize:
- Time periods: 5, 10, 20, 30 years (line 43)
- Theta values: 1.5, 5, 10, 100 (line 48)
- Number of simulation iterations

## 🔧 Troubleshooting

### Common Issues:
1. **PyMC Installation:** If Bayesian models fail, ensure PyMC is properly installed
2. **Memory Errors:** Reduce the number of parallel workers in `run_all_models.py`
3. **Missing Data:** Ensure all input data folders are present before starting
4. **Mac Users:** Uncomment PyTensor configuration lines in model files if needed

### Performance Tips:
- Run on a machine with at least 8GB RAM for large simulations
- Use GPU acceleration for faster Bayesian model computation
- Monitor disk space - simulated data can be large

## 📈 Project Structure

```
├── Data Import & Processing
│   ├── 1_import_merge.py          # Import and merge climate/fire data
│   ├── 2_data_aggregate.py        # Create final analysis dataset
│   └── categorize_locations.py    # Spatial categorization
├── Analysis & Modeling
│   ├── 3_check_seasonality.py     # Seasonality detection
│   ├── 5_simulation_temp.py       # Generate synthetic data
│   ├── 6_final_models_real.py     # Real data modeling
│   └── 6_final_models_real_categorized.py  # Categorized modeling
├── Evaluation & Comparison
│   ├── run_all_models.py          # Comprehensive evaluation
│   ├── compare_categorized_vs_original.py  # Performance comparison
│   ├── visualize_simulations.py   # Results visualization
│   └── scatterplots_(monthly_yearly).py    # Temporal analysis
└── Utilities
    ├── model_functions.py         # Shared model functions
    ├── output_utils.py            # Output management
    └── checkpoint_manager.py      # Progress tracking
```

## 🤝 Contributors

This project was developed through collaborative effort by:

- [@Mohamed-Elsafty23](https://github.com/Mohamed-Elsafty23)
- [@Wanja24](https://github.com/Wanja24)
- [@Ghaith-Dailami](https://github.com/Ghaith-Dailami) 
- [@joyntv](https://github.com/joyntv)

## 📝 Citation

If you use this code or methodology in your research, please cite the associated publication: "A Bayesian Model for Predicting Fire Frequency in Kenya."

## 📄 License

This project is open source. Please refer to the LICENSE file for details.

---

*For questions or issues, please open a GitHub issue or contact the contributors.*
