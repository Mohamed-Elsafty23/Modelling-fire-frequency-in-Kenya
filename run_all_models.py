from output_utils import get_output_path, get_model_results_path, get_simulated_data_path, ensure_output_dirs
#!/usr/bin/env python3
"""
Master script to run all model evaluations - Python version of all 7models_*.R
Runs standard and Bayesian NB models on all simulated datasets (5, 10, 20, 30 years)
OPTIMIZED VERSION: Enhanced with GPU acceleration, maximum parallelization, and CHECKPOINTS
"""

import pandas as pd
import numpy as np
import glob
import os
import json
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import time
import psutil
from datetime import datetime

# GPU acceleration imports
GPU_AVAILABLE = False
GPU_MESSAGE = ""

# Comprehensive warning suppression for clean output
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='CUDA path could not be detected')
warnings.filterwarnings('ignore', message='g\+\+ not available')
warnings.filterwarnings('ignore', message='g\+\+ not detected')
warnings.filterwarnings('ignore', message='.*PyTensor.*')

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', message='.*oneDNN.*')
warnings.filterwarnings('ignore', message='.*tensorflow.*')

# Configure PyTensor for better performance
try:
    import pytensor
    # Set PyTensor to use Python mode to avoid compiler warnings
    pytensor.config.gcc__cxxflags = ""
    pytensor.config.on_opt_error = 'warn'
    pytensor.config.on_shape_error = 'warn'
    # Disable compiler warnings
    pytensor.config.mode = 'FAST_COMPILE'
except ImportError:
    pass

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging

try:
    import cupy as cp
    import cudf
    GPU_AVAILABLE = True
    GPU_MESSAGE = "✅ GPU acceleration available (CuPy/cuDF)"
except ImportError as e:
    if "CUDA" in str(e) or "cuda" in str(e):
        GPU_MESSAGE = "⚠️  CUDA not properly installed - using CPU optimization only"
    else:
        try:
            import tensorflow as tf
            # Check for GPU
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            if len(physical_devices) > 0:
                GPU_AVAILABLE = True
                GPU_MESSAGE = "✅ GPU acceleration available (TensorFlow)"
            else:
                GPU_MESSAGE = "⚠️  TensorFlow found but no GPU detected - using CPU optimization only"
        except ImportError:
            try:
                import torch
                if torch.cuda.is_available():
                    GPU_AVAILABLE = True
                    GPU_MESSAGE = "✅ GPU acceleration available (PyTorch)"
                else:
                    GPU_MESSAGE = "⚠️  PyTorch found but no CUDA GPU detected - using CPU optimization only"
            except ImportError:
                GPU_MESSAGE = "💻 No GPU packages found - using optimized CPU processing"

print(GPU_MESSAGE)

# Import model functions
from model_functions import negbinner, stanbinner

class ModelCheckpointManager:
    """
    Manages model checkpoints to avoid re-running completed models
    """
    
    def __init__(self, checkpoint_dir="model_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "checkpoint_status.json"
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.load_checkpoint_status()
        
    def load_checkpoint_status(self):
        """Load existing checkpoint status"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    self.completed_jobs = json.load(f)
                print(f"📁 Loaded checkpoint status: {len(self.completed_jobs)} completed jobs")
            except Exception as e:
                print(f"⚠️  Error loading checkpoint: {e}")
                self.completed_jobs = {}
        else:
            self.completed_jobs = {}
            
        # Load metadata
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except:
                self.metadata = {}
        else:
            self.metadata = {}
    
    def save_checkpoint_status(self):
        """Save current checkpoint status"""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.completed_jobs, f, indent=2)
            
            # Update metadata
            self.metadata['last_updated'] = datetime.now().isoformat()
            self.metadata['total_completed'] = len(self.completed_jobs)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Error saving checkpoint: {e}")
    
    def get_job_key(self, time_period, theta_value, model_name):
        """Generate unique key for a job"""
        return f"{time_period}year_theta{theta_value}_{model_name.lower()}"
    
    def is_job_completed(self, time_period, theta_value, model_name):
        """Check if a specific job is already completed"""
        job_key = self.get_job_key(time_period, theta_value, model_name)
        
        # Check if job is marked as completed
        if job_key not in self.completed_jobs:
            return False
            
        # Verify output file still exists
        job_info = self.completed_jobs[job_key]
        output_file = job_info.get('output_file')
        
        if output_file and os.path.exists(output_file):
            print(f"✅ Skipping {job_key} - already completed ({output_file})")
            return True
        else:
            # File doesn't exist, remove from completed jobs
            print(f"⚠️  Output file missing for {job_key}, will re-run")
            del self.completed_jobs[job_key]
            self.save_checkpoint_status()
            return False
    
    def mark_job_completed(self, time_period, theta_value, model_name, output_file, processing_time, file_count):
        """Mark a job as completed"""
        job_key = self.get_job_key(time_period, theta_value, model_name)
        
        self.completed_jobs[job_key] = {
            'time_period': time_period,
            'theta_value': theta_value,
            'model_name': model_name,
            'output_file': output_file,
            'processing_time': processing_time,
            'file_count': file_count,
            'completed_at': datetime.now().isoformat(),
            'system_info': {
                'cpu_cores': mp.cpu_count(),
                'gpu_available': GPU_AVAILABLE
            }
        }
        
        self.save_checkpoint_status()
        print(f"💾 Checkpoint saved for {job_key}")
    
    def get_resume_summary(self):
        """Get summary of what can be resumed"""
        if not self.completed_jobs:
            return "🚀 Starting fresh - no previous checkpoints found"
        
        summary = f"📁 Found {len(self.completed_jobs)} completed jobs:\n"
        
        # Group by time period and model
        by_period = {}
        for job_key, job_info in self.completed_jobs.items():
            period = job_info['time_period']
            model = job_info['model_name']
            if period not in by_period:
                by_period[period] = {}
            if model not in by_period[period]:
                by_period[period][model] = []
            by_period[period][model].append(job_info['theta_value'])
        
        for period in sorted(by_period.keys()):
            summary += f"   • {period}-year models:\n"
            for model in sorted(by_period[period].keys()):
                thetas = sorted(by_period[period][model])
                summary += f"     - {model.upper()}: theta = {thetas}\n"
        
        return summary
    
    def clear_checkpoints(self):
        """Clear all checkpoints (start fresh)"""
        self.completed_jobs = {}
        self.save_checkpoint_status()
        print("🗑️  All checkpoints cleared")

# Global checkpoint manager
checkpoint_manager = ModelCheckpointManager()

def get_optimal_workers():
    """
    Get optimal number of workers based on system resources
    """
    # Get system info
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Base workers on CPU cores
    max_workers = cpu_count
    
    # Adjust based on memory (each worker needs ~1-2GB)
    memory_limited_workers = int(memory_gb / 2)
    max_workers = min(max_workers, memory_limited_workers)
    
    # Ensure at least 2 workers but not more than 32
    max_workers = max(2, min(max_workers, 32))
    
    print(f"🖥️  System: {cpu_count} CPUs, {memory_gb:.1f}GB RAM")
    print(f"⚡ Optimal workers: {max_workers}")
    
    return max_workers

def process_single_file_optimized(file_path, model_func, theta, n_months):
    """Process a single simulated dataset file with available optimizations"""
    try:
        # Read data with available optimization
        if GPU_AVAILABLE:
            try:
                # Try GPU-accelerated reading if available
                if 'cudf' in globals():
                    data = cudf.read_csv(file_path)
                    # Convert to pandas if model doesn't support cuDF
                    if not hasattr(model_func, '_supports_cudf'):
                        data = data.to_pandas()
                else:
                    # Fallback to pandas
                    data = pd.read_csv(file_path)
            except Exception as e:
                # If GPU fails, fallback to pandas
                print(f"GPU reading failed, using pandas: {e}")
                data = pd.read_csv(file_path)
        else:
            # Standard optimized pandas reading
            data = pd.read_csv(file_path)
        
        # Run model
        result = model_func(data, theta=theta, n=n_months)
        
        return result
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {
            'rmse_train': np.nan, 'rmse_test': np.nan,
            'mase_test': np.nan, 'bias_test': np.nan,
            'theta': theta, 'n': n_months
        }

def process_single_file(file_path, model_func, theta, n_months):
    """Process a single simulated dataset file (original function for compatibility)"""
    return process_single_file_optimized(file_path, model_func, theta, n_months)

def run_models_on_theta(theta_value, model_func, model_name, time_period, n_months, max_workers=None):
    """Run models on all datasets for a specific theta value and time period"""
    
    # Check if this job is already completed
    if checkpoint_manager.is_job_completed(time_period, theta_value, model_name):
        return pd.DataFrame()  # Return empty DataFrame, job already done
    
    if max_workers is None:
        max_workers = get_optimal_workers()
    
    print(f"\n{'='*60}")
    print(f"🚀 Running {model_name} models: {time_period} years, theta = {theta_value}")
    print(f"⚡ Using {max_workers} parallel workers")
    if GPU_AVAILABLE:
        print("🎮 GPU acceleration: ENABLED")
    print('='*60)
    
    # Get file paths
    data_path = get_simulated_data_path(f"d{time_period}year/theta_{theta_value}")
    file_pattern = f"{data_path}/*.csv"
    file_paths = glob.glob(file_pattern)
    
    if not file_paths:
        print(f"No data files found in {data_path}")
        return pd.DataFrame()
    
    print(f"📁 Found {len(file_paths)} datasets to process")
    
    # Process files
    results = []
    start_time = time.time()
    
    # Enhanced parallel processing
    if model_name.lower() == 'nb':
        # Use ProcessPoolExecutor for CPU-intensive standard models
        executor_class = ProcessPoolExecutor
        print(f"🔧 Using ProcessPoolExecutor with {max_workers} processes")
    else:
        # Use ThreadPoolExecutor for I/O intensive Bayesian models
        # but with more threads than before
        bayesian_workers = min(max_workers // 2, 8)  # Less aggressive for Bayesian
        executor_class = ThreadPoolExecutor
        max_workers = bayesian_workers
        print(f"🔧 Using ThreadPoolExecutor with {max_workers} threads (Bayesian)")
    
    with executor_class(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file_optimized, file_path, model_func, theta_value, n_months): file_path
            for file_path in file_paths
        }
            
        # Process results with progress tracking
        processed_count = 0
        for future in as_completed(future_to_file):
            result = future.result()
            results.append(result)
            processed_count += 1
            
            # Progress reporting
            if processed_count % 50 == 0 or processed_count == len(file_paths):
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                eta = (len(file_paths) - processed_count) / rate if rate > 0 else 0
                
                print(f"📊 Progress: {processed_count}/{len(file_paths)} "
                      f"({processed_count/len(file_paths)*100:.1f}%) | "
                      f"Rate: {rate:.1f}/sec | ETA: {eta/60:.1f}min")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    os.makedirs(get_model_results_path(""), exist_ok=True)
    
    time_suffix = {5: "five", 10: "ten", 20: "twenty", 30: "thirty"}[time_period]
    
    if model_name.lower() == 'nb':
        output_file = get_model_results_path(f"{time_suffix}_year_{theta_value}_metrics.csv")
    else:  # Bayesian
        output_file = get_model_results_path(f"{time_suffix}_year_{theta_value}b_metrics.csv")
    
    results_df.to_csv(output_file, index=False)
    
    elapsed = time.time() - start_time
    print(f"✅ Results saved to: {output_file}")
    print(f"⏱️  Processing time: {elapsed/60:.1f} minutes")
    print(f"🎯 Average rate: {len(file_paths)/elapsed:.1f} files/second")
    
    # Save checkpoint
    checkpoint_manager.mark_job_completed(
        time_period=time_period,
        theta_value=theta_value, 
        model_name=model_name,
        output_file=output_file,
        processing_time=elapsed,
        file_count=len(file_paths)
    )
    
    if len(results_df) > 0:
        print(f"📈 Summary statistics:")
        numeric_cols = ['rmse_train', 'rmse_test', 'mase_test', 'bias_test']
        print(results_df[numeric_cols].describe().round(4))
    
    return results_df

def run_time_period_models(time_period, n_months):
    """Run all models for a specific time period with optimizations"""
    
    print(f"\n{'='*80}")
    print(f"🚀 PROCESSING {time_period}-YEAR MODELS ({n_months} months)")
    print(f"💻 System optimization: {'GPU + ' if GPU_AVAILABLE else ''}Multi-core CPU")
    print("="*80)
    
    # Check if data exists
    data_dir = get_simulated_data_path(f"d{time_period}year")
    if not os.path.exists(data_dir):
        print(f"❌ Error: {data_dir} not found!")
        print("Please run 5_simulation_temp.py first to generate simulated datasets.")
        return 0
    
    # Parameters
    theta_values = [1.5, 5, 10, 100]
    max_workers = get_optimal_workers()
    jobs_run = 0
    
    # Run Standard Negative Binomial models (high parallelization)
    print(f"\n--- 🔥 Standard NB Models ({time_period} years) ---")
    for theta in theta_values:
        try:
            result_df = run_models_on_theta(
                theta_value=theta,
                model_func=negbinner,
                model_name='NB',
                time_period=time_period,
                n_months=n_months,
                max_workers=max_workers
            )
            if not result_df.empty:  # Job was actually run (not skipped)
                jobs_run += 1
        except Exception as e:
            print(f"❌ Error processing NB models for {time_period}yr, theta {theta}: {e}")
    
    # Run Bayesian models (moderate parallelization)
    print(f"\n--- 🧠 Bayesian NB Models ({time_period} years) ---")
    for theta in theta_values:
        try:
            # Use fewer workers for Bayesian to avoid memory issues
            bayesian_workers = min(max_workers // 2, 6)
            result_df = run_models_on_theta(
                theta_value=theta,
                model_func=stanbinner,
                model_name='Bayesian',
                time_period=time_period,
                n_months=n_months,
                max_workers=bayesian_workers
            )
            if not result_df.empty:  # Job was actually run (not skipped)
                jobs_run += 1
        except Exception as e:
            print(f"❌ Error processing Bayesian models for {time_period}yr, theta {theta}: {e}")
    
    return jobs_run

def main():
    """Main function to run all model evaluations with maximum optimization and checkpoints"""
    
    print("="*80)
    print("🚀 COMPREHENSIVE MODEL EVALUATION PIPELINE - OPTIMIZED + CHECKPOINTS")
    print("Python version of 7models_*.R scripts")
    print("⚡ Enhanced with GPU acceleration, maximum parallelization, and smart resuming")
    print("="*80)
    
    # Display system capabilities
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    print(f"💻 System Capabilities:")
    print(f"   • CPU Cores: {cpu_count}")
    print(f"   • RAM: {memory_gb:.1f} GB")
    print(f"   • GPU Status: {GPU_MESSAGE}")
    
    optimal_workers = get_optimal_workers()
    print(f"   • Optimal parallel workers: {optimal_workers}")
    
    # Display checkpoint status
    print(f"\n💾 Checkpoint Status:")
    checkpoint_summary = checkpoint_manager.get_resume_summary()
    print(checkpoint_summary)
    
    # Check if user wants to clear checkpoints
    if checkpoint_manager.completed_jobs:
        print(f"\n⚠️  Found existing checkpoints. Options:")
        print(f"   • Continue: Resume from where you left off (recommended)")
        print(f"   • Clear: Delete all checkpoints and start fresh")
        
        # For automated running, we'll continue by default
        # You can add user input here if needed:
        # choice = input("Continue (c) or Clear (x)? [c]: ").lower()
        # if choice == 'x':
        #     checkpoint_manager.clear_checkpoints()
    
    # Set seed EXACTLY like R scripts
    # R code: set.seed(76568)
    np.random.seed(76568)
    
    # Time periods and corresponding months
    time_periods = {
        5: 60,    # 5 years = 60 months
        10: 120,  # 10 years = 120 months
        20: 240,  # 20 years = 240 months
        30: 360   # 30 years = 360 months
    }
    
    # Check if simulated data exists
    if not os.path.exists(get_simulated_data_path()):
        print("❌ Error: Simulated data directory not found!")
        print("Please run 5_simulation_temp.py first to generate simulated datasets.")
        return
    
    print(f"\n📋 Processing Configuration:")
    print(f"   • Time periods: {list(time_periods.keys())} years")
    print(f"   • Theta values: [1.5, 5, 10, 100]")
    print(f"   • Model types: Standard NB + Bayesian NB")
    
    # Calculate what needs to be done
    total_combinations = len(time_periods) * 4 * 2  # periods × thetas × models
    completed_count = len(checkpoint_manager.completed_jobs)
    remaining_count = total_combinations - completed_count
    
    print(f"   • Total model combinations: {total_combinations}")
    print(f"   • Already completed: {completed_count}")
    print(f"   • Remaining to process: {remaining_count}")
    
    if remaining_count == 0:
        print(f"\n🎉 All models already completed! No work needed.")
        print(f"📁 Results available in: {get_model_results_path('')}")
        return
    
    if GPU_AVAILABLE:
        print(f"🎮 GPU Acceleration: ENABLED - Expect 2-5x speedup")
    else:
        print(f"🖥️  CPU-only processing with {optimal_workers} parallel workers")
    
    total_start_time = time.time()
    jobs_run = 0
    
    # Process each time period
    for time_period, n_months in time_periods.items():
        try:
            period_start_time = time.time()
            period_jobs_run = run_time_period_models(time_period, n_months)
            jobs_run += period_jobs_run
            
            if period_jobs_run > 0:
                period_elapsed = time.time() - period_start_time
                print(f"✅ {time_period}-year models completed in {period_elapsed/60:.1f} minutes")
            else:
                print(f"⏭️  {time_period}-year models already completed (skipped)")
                
        except Exception as e:
            print(f"❌ Error processing {time_period}-year models: {e}")
            continue
    
    # Final summary
    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*80}")
    if jobs_run > 0:
        print(f"🎉 PROCESSING SESSION COMPLETED!")
        print(f"⏱️  Session time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.1f} hours)")
        print(f"📊 Jobs processed this session: {jobs_run}")
        print(f"📊 Average time per job: {total_elapsed/jobs_run:.1f} seconds")
    else:
        print(f"✅ ALL MODELS ALREADY COMPLETED!")
        print(f"⏱️  Session time: {total_elapsed:.1f} seconds (checkpoint check only)")
    
    print(f"📁 Results saved in: {get_model_results_path('')}")
    print(f"💾 Checkpoint data: {checkpoint_manager.checkpoint_dir}")
    
    if GPU_AVAILABLE and jobs_run > 0:
        print(f"🎮 GPU acceleration was utilized for maximum performance")
    if jobs_run > 0:
        print(f"⚡ Peak parallelization: {optimal_workers} workers")
    
    # Show final checkpoint status
    final_completed = len(checkpoint_manager.completed_jobs)
    print(f"📈 Total completed jobs: {final_completed}/{total_combinations}")
    
    if final_completed == total_combinations:
        print(f"🏆 PIPELINE 100% COMPLETE!")
    else:
        remaining = total_combinations - final_completed
        print(f"⏳ {remaining} jobs remaining for future runs")
    
    print("="*80)

if __name__ == "__main__":
    # Windows multiprocessing support
    mp.freeze_support()
    main() 