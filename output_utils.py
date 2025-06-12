#!/usr/bin/env python3
"""
Output utilities for the Fire Frequency Modeling Project
Provides consistent output directory management across all scripts
"""

import os
from pathlib import Path

def get_output_dir():
    """Get the output directory from environment variable or default"""
    return os.environ.get('OUTPUT_DIR', 'our_output')

def get_output_path(filename):
    """Get full path for output file"""
    output_dir = get_output_dir()
    return os.path.join(output_dir, filename)

def get_model_results_path(filename):
    """Get path for model results file"""
    output_dir = get_output_dir()
    return os.path.join(output_dir, 'model_results', filename)

def get_simulated_data_path(subdir="", filename=""):
    """Get path for simulated data"""
    output_dir = get_output_dir()
    if subdir:
        path = os.path.join(output_dir, 'simulated_data', subdir)
    else:
        path = os.path.join(output_dir, 'simulated_data')
    
    if filename:
        path = os.path.join(path, filename)
    
    return path

def get_descriptive_plots_path(filename):
    """Get path for descriptive plots"""
    output_dir = get_output_dir()
    return os.path.join(output_dir, 'descriptive_plots', filename)

def get_fire_data_path(filename):
    """Get path for fire data files"""
    # Use the root fire directory instead of creating one in our_output
    return os.path.join('fire', filename)

def get_climate_data_path(filename):
    """Get path for climate data files"""
    # Use the root climate directory instead of creating one in our_output
    return os.path.join('climate', filename)

def ensure_output_dirs():
    """Create all necessary output directories"""
    output_dir = get_output_dir()
    dirs_to_create = [
        output_dir,
        os.path.join(output_dir, 'model_results'),
        os.path.join(output_dir, 'simulated_data'),
        os.path.join(output_dir, 'descriptive_plots')
    ]
    
    for dir_path in dirs_to_create:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"Ensured directory exists: {dir_path}")
        except Exception as e:
            print(f"Error creating directory {dir_path}: {e}")
            raise
    
    return output_dir 