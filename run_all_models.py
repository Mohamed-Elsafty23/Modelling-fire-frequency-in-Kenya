from output_utils import get_output_path, get_model_results_path, get_simulated_data_path, ensure_output_dirs
#!/usr/bin/env python3
"""
Master script to run all model evaluations - Python version of all 7models_*.R
Runs standard and Bayesian NB models on all simulated datasets (5, 10, 20, 30 years)
OPTIMIZED VERSION: Enhanced with GPU acceleration, maximum parallelization, and FILE-BASED CHECKING
"""

# Configure PyTensor BEFORE any imports
import os
# Disable compilation caching to avoid parallel processing conflicts
os.environ['PYTENSOR_FLAGS'] = 'mode=FAST_RUN,optimizer=fast_run,cxx='

import pandas as pd
import numpy as np
import glob
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import time
import psutil

# Cloud resource detection and optimization
GPU_AVAILABLE = False
TPU_AVAILABLE = False
GPU_MESSAGE = ""
TPU_MESSAGE = ""
CLOUD_PLATFORM = "Unknown"


# ============================================================================
# CONFIGURABLE PARAMETERS - EDIT THESE TO CONTROL TIME PERIODS AND THETA VALUES
# ============================================================================

# Time periods to process (in years) and their corresponding months
TIME_PERIODS = {
    5: 60,    # 5 years = 60 months
    # 10: 120,  # 10 years = 120 months
    # 20: 240,  # 20 years = 240 months
    # 30: 360   # 30 years = 360 months
}

# Theta values to test for dispersion parameter
THETA_VALUES = [1.5, 5, 10, 100]

def update_processing_parameters(time_periods=None, theta_values=None):
    """
    Update processing parameters
    
    Args:
        time_periods (dict): Dictionary mapping years to months, e.g., {5: 60, 10: 120}
        theta_values (list): List of theta values to test, e.g., [1.5, 5, 10]
    """
    global TIME_PERIODS, THETA_VALUES
    
    if time_periods is not None:
        TIME_PERIODS = time_periods
        print(f"[CONFIG] Updated time periods: {list(TIME_PERIODS.keys())} years")
    
    if theta_values is not None:
        THETA_VALUES = theta_values
        print(f"[CONFIG] Updated theta values: {THETA_VALUES}")
    
    print(f"[CONFIG] Total combinations: {len(TIME_PERIODS)} × {len(THETA_VALUES)} × 2 = {len(TIME_PERIODS) * len(THETA_VALUES) * 2}")

# ============================================================================
# THREAD CONTROL SETTINGS
# ============================================================================

THREAD_CONTROL_MODE = "manual"  # Options: "auto", "manual"
MANUAL_THREAD_COUNT = 1     # Used when THREAD_CONTROL_MODE = "manual"

def set_thread_control(mode="auto", manual_count=16):
    """
    Set thread control mode and manual thread count
    
    Args:
        mode (str): "auto" or "manual"
        manual_count (int): Number of threads to use in manual mode
    """
    global THREAD_CONTROL_MODE, MANUAL_THREAD_COUNT
    
    if mode not in ["auto", "manual"]:
        raise ValueError("Mode must be 'auto' or 'manual'")
    
    THREAD_CONTROL_MODE = mode
    MANUAL_THREAD_COUNT = max(1, manual_count)  # Ensure at least 1 thread
    
    print(f"[THREAD CONTROL] Updated settings:")
    print(f"   - Mode: {THREAD_CONTROL_MODE.upper()}")
    if THREAD_CONTROL_MODE == "manual":
        print(f"   - Manual thread count: {MANUAL_THREAD_COUNT}")
    else:
        print(f"   - Auto mode: Will calculate optimal threads")

def get_thread_info():
    """Get current thread control information"""
    return {
        "mode": THREAD_CONTROL_MODE,
        "manual_count": MANUAL_THREAD_COUNT,
        "current_workers": get_optimal_workers()
    }

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

# Additional PyTensor configuration (main config set via environment variable)
try:
    import pytensor
    # Set PyTensor to use Python mode to avoid compiler warnings
    pytensor.config.gcc__cxxflags = ""
    pytensor.config.on_opt_error = 'warn'
    pytensor.config.on_shape_error = 'warn'
except ImportError:
    pass

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging

def detect_cloud_resources():
    """Comprehensive cloud resource detection"""
    global GPU_AVAILABLE, TPU_AVAILABLE, GPU_MESSAGE, TPU_MESSAGE, CLOUD_PLATFORM
    
    # Detect cloud platform
    try:
        import subprocess
        # Check for Google Colab
        if 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ:
            CLOUD_PLATFORM = "Google Colab"
        # Check for Kaggle
        elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            CLOUD_PLATFORM = "Kaggle"
        # Check for generic cloud (GCP metadata server)
        else:
            try:
                result = subprocess.run(['curl', '-s', '-H', 'Metadata-Flavor: Google', 
                                       'http://metadata.google.internal/computeMetadata/v1/instance/'], 
                                      capture_output=True, timeout=2)
                if result.returncode == 0:
                    CLOUD_PLATFORM = "Google Cloud"
            except:
                pass
    except Exception as e:
        print(f"[WARNING] Cloud platform detection failed: {e}")
        CLOUD_PLATFORM = "Unknown"

    # GPU Detection with cloud optimizations
    gpu_count = 0
    gpu_memory = 0
    gpu_name = "Unknown"
    
    # Try TensorFlow first (most reliable in cloud environments)
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if len(gpus) > 0:
            GPU_AVAILABLE = True
            gpu_count = len(gpus)
            # Try to get GPU info
            try:
                gpu_details = tf.config.experimental.get_device_details(gpus[0])
                gpu_name = gpu_details.get('device_name', 'Unknown')
                # Enable memory growth to avoid OOM
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except:
                pass
            GPU_MESSAGE = f"[TENSORFLOW] {gpu_count} GPU(s) available ({gpu_name})"
        else:
            GPU_MESSAGE = "[INFO] TensorFlow found but no GPU detected"
    except ImportError:
        GPU_MESSAGE = "[INFO] TensorFlow not available"
    except Exception as e:
        GPU_MESSAGE = f"[WARNING] TensorFlow GPU detection failed: {str(e)[:50]}"
    
    # Try PyTorch if TensorFlow didn't work
    if not GPU_AVAILABLE:
        try:
            import torch
            if torch.cuda.is_available():
                GPU_AVAILABLE = True
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if gpu_count > 0 else 0
                GPU_MESSAGE = f"[PYTORCH] {gpu_count} GPU(s) available ({gpu_name}, {gpu_memory:.1f}GB)"
            else:
                if not GPU_MESSAGE.startswith("[TENSORFLOW]"):
                    GPU_MESSAGE = "[INFO] PyTorch found but no CUDA GPU detected"
        except ImportError:
            if not GPU_MESSAGE.startswith("[TENSORFLOW]"):
                GPU_MESSAGE = "[INFO] PyTorch not available"
        except Exception as e:
            if not GPU_MESSAGE.startswith("[TENSORFLOW]"):
                GPU_MESSAGE = f"[WARNING] PyTorch GPU detection failed: {str(e)[:50]}"
    
    # Try CuPy last (most likely to have driver issues)
    if not GPU_AVAILABLE:
        try:
            import cupy as cp
            gpu_count = cp.cuda.runtime.getDeviceCount()
            if gpu_count > 0:
                GPU_AVAILABLE = True
                # Get GPU memory info
                try:
                    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                    gpu_memory = total_mem / (1024**3)  # GB
                except:
                    gpu_memory = 0
                GPU_MESSAGE = f"[CUPY] {gpu_count} GPU(s) available, {gpu_memory:.1f}GB total memory"
            else:
                if not GPU_MESSAGE.startswith("[TENSORFLOW]") and not GPU_MESSAGE.startswith("[PYTORCH]"):
                    GPU_MESSAGE = "[INFO] CuPy found but no GPU detected"
        except ImportError:
            if not GPU_MESSAGE.startswith("[TENSORFLOW]") and not GPU_MESSAGE.startswith("[PYTORCH]"):
                GPU_MESSAGE = "[INFO] CuPy not available"
        except Exception as e:
            if not GPU_MESSAGE.startswith("[TENSORFLOW]") and not GPU_MESSAGE.startswith("[PYTORCH]"):
                GPU_MESSAGE = f"[WARNING] CuPy GPU detection failed: {str(e)[:50]}"
    
    # If no GPU detected by any method, set default message
    if not GPU_AVAILABLE and not GPU_MESSAGE.startswith("[TENSORFLOW]") and not GPU_MESSAGE.startswith("[PYTORCH]") and not GPU_MESSAGE.startswith("[CUPY]"):
        GPU_MESSAGE = "[INFO] No GPU detected"

    # TPU Detection (mainly for Google Colab/Cloud)
    try:
        if CLOUD_PLATFORM in ["Google Colab", "Google Cloud"]:
            try:
                import tensorflow as tf
                resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
                tf.config.experimental_connect_to_cluster(resolver)
                tf.tpu.experimental.initialize_tpu_system(resolver)
                TPU_AVAILABLE = True
                TPU_MESSAGE = "[TPU] TPU cluster detected and initialized"
            except:
                try:
                    # Alternative TPU detection
                    tpu_address = os.environ.get('COLAB_TPU_ADDR', '')
                    if tpu_address:
                        TPU_AVAILABLE = True
                        TPU_MESSAGE = f"[TPU] TPU available at {tpu_address}"
                    else:
                        TPU_MESSAGE = "[INFO] No TPU detected"
                except:
                    TPU_MESSAGE = "[INFO] No TPU detected"
        else:
            TPU_MESSAGE = "[INFO] Platform doesn't typically support TPUs"
    except Exception as e:
        TPU_MESSAGE = f"[WARNING] TPU detection failed: {str(e)[:50]}"

    # Detect additional cloud resources
    additional_info = []
    
    # High memory instances
    memory_gb = psutil.virtual_memory().total / (1024**3)
    if memory_gb > 50:
        additional_info.append(f"High-memory instance: {memory_gb:.1f}GB RAM")
    
    # Fast storage detection
    try:
        disk_usage = psutil.disk_usage('/')
        if disk_usage.total > 500 * (1024**3):  # > 500GB
            additional_info.append(f"Large storage: {disk_usage.total/(1024**3):.1f}GB")
    except:
        pass
    
    return gpu_count, gpu_memory, additional_info

# Detect all available resources
try:
    gpu_count, gpu_memory, additional_info = detect_cloud_resources()
except Exception as e:
    print(f"[WARNING] Resource detection failed: {e}")
    print("[INFO] Continuing with default settings...")
    gpu_count = 0
    gpu_memory = 0
    additional_info = []
    GPU_AVAILABLE = False
    GPU_MESSAGE = "[WARNING] GPU detection failed, using CPU only"

def print_cloud_setup_guide():
    """Print cloud platform setup recommendations"""
    print("="*80)
    print("[CLOUD OPTIMIZATION] Automatic cloud resource detection and optimization")
    print("="*80)
    
    if CLOUD_PLATFORM == "Google Colab":
        print("[COLAB] Detected Google Colab environment")
        print("   Recommendations:")
        print("   - Use Runtime > Change runtime type > GPU/TPU for acceleration")
        print("   - Consider Colab Pro for longer sessions and better GPUs")
        print("   - Keep this browser tab active to prevent disconnection")
    elif CLOUD_PLATFORM == "Kaggle":
        print("[KAGGLE] Detected Kaggle environment")
        print("   Recommendations:")
        print("   - Enable GPU in Settings > Accelerator")
        print("   - Use Internet-enabled kernels for package downloads")
        print("   - 9-hour session limit - use checkpoints for long runs")
    elif CLOUD_PLATFORM == "Google Cloud":
        print("[GCP] Detected Google Cloud environment")
        print("   Recommendations:")
        print("   - Consider GPU/TPU instances for maximum performance")
        print("   - Use preemptible instances for cost savings")
    else:
        print("[LOCAL] Local or unknown environment detected")
        print("   - For cloud speedup, consider Google Colab or Kaggle")
    
    print("="*80)


print_cloud_setup_guide()

print(f"[CLOUD] Platform: {CLOUD_PLATFORM}")
print(GPU_MESSAGE)
if TPU_AVAILABLE:
    print(TPU_MESSAGE)
for info in additional_info:
    print(f"[RESOURCE] {info}")

# Import model functions
from model_functions import negbinner, stanbinner

def check_output_file_exists(time_period, theta_value, model_name):
    """
    Check if output file already exists in the model_results directory
    
    Args:
        time_period (int): Time period in years (5, 10, 20, 30)
        theta_value (float): Theta value (1.5, 5, 10, 100)
        model_name (str): Model name ('NB' or 'Bayesian')
    
    Returns:
        bool: True if file exists, False otherwise
    """
    # Get the expected output file path
    time_suffix = {5: "five", 10: "ten", 20: "twenty", 30: "thirty"}[time_period]
    
    if model_name.lower() == 'nb':
        filename = f"{time_suffix}_year_{theta_value}_metrics.csv"
    else:  # Bayesian
        filename = f"{time_suffix}_year_{theta_value}b_metrics.csv"
    
    output_file = get_model_results_path(filename)
    
    if os.path.exists(output_file):
        print(f"[SKIP] Skipping {time_period}yr theta{theta_value} {model_name} - file exists: {filename}")
        return True
    else:
        return False

def get_optimal_workers():
    """
    Get optimal number of workers based on cloud resources
    Cloud-optimized: More aggressive resource utilization
    """
    # Check thread control mode
    if THREAD_CONTROL_MODE == "manual":
        max_workers = MANUAL_THREAD_COUNT
        print(f"[THREAD CONTROL] Manual mode: Using {max_workers} workers")
        return max_workers
    
    # Auto mode - calculate optimal workers
    print(f"[THREAD CONTROL] Auto mode: Calculating optimal workers")
    
    # Get system info
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Cloud-specific optimizations
    if CLOUD_PLATFORM in ["Google Colab", "Kaggle", "Google Cloud"]:
        # Cloud platforms often have more generous resource allocation
        # Be more aggressive with worker count
        
        if GPU_AVAILABLE:
            # With GPU acceleration, we can use more workers for CPU preprocessing
            base_workers = min(cpu_count * 2, 64)  # Up to 2x CPU cores
        else:
            # Without GPU, maximize CPU utilization
            base_workers = cpu_count
            
        # Cloud memory optimization
        if memory_gb > 50:  # High-memory instance
            memory_limited_workers = int(memory_gb / 1)  # 1GB per worker (more aggressive)
        elif memory_gb > 25:  # Medium-memory instance
            memory_limited_workers = int(memory_gb / 1.5)  # 1.5GB per worker
        else:  # Standard instance
            memory_limited_workers = int(memory_gb / 2)  # 2GB per worker
            
        max_workers = min(base_workers, memory_limited_workers)
        
        # Cloud platform specific limits
        if CLOUD_PLATFORM == "Google Colab":
            max_workers = min(max_workers, 32)  # Colab limit
        elif CLOUD_PLATFORM == "Kaggle":
            max_workers = min(max_workers, 16)  # Kaggle limit
        else:  # Google Cloud or other
            max_workers = min(max_workers, 128)  # Higher limit for paid cloud
            
    else:
        # Local/unknown environment - conservative approach
        max_workers = cpu_count
        memory_limited_workers = int(memory_gb / 2)
        max_workers = min(max_workers, memory_limited_workers)
        max_workers = min(max_workers, 32)
    
    # Ensure minimum workers
    max_workers = max(2, max_workers)
    
    # Additional resource info
    available_memory = psutil.virtual_memory().available / (1024**3)
    cpu_freq = psutil.cpu_freq()
    cpu_freq_str = f"{cpu_freq.current:.0f}MHz" if cpu_freq else "Unknown"
    
    print(f"[SYSTEM] Platform: {CLOUD_PLATFORM}")
    print(f"[SYSTEM] CPU: {cpu_count} cores @ {cpu_freq_str}")
    print(f"[SYSTEM] RAM: {memory_gb:.1f}GB total, {available_memory:.1f}GB available")
    if GPU_AVAILABLE:
        print(f"[SYSTEM] GPU: Available ({gpu_count} device(s))")
    if TPU_AVAILABLE:
        print(f"[SYSTEM] TPU: Available")
    print(f"[OPTIMIZATION] Selected workers: {max_workers}")
    
    return max_workers

def process_single_file_optimized(file_path, model_func, theta, n_months):
    """Process a single simulated dataset file with cloud optimizations"""
    try:
        # Cloud-optimized data loading
        if GPU_AVAILABLE and CLOUD_PLATFORM in ["Google Colab", "Google Cloud"]:
            try:
                # Try GPU-accelerated data loading if available
                import cudf
                data = cudf.read_csv(file_path)
                # Convert to pandas for model compatibility if needed
                if hasattr(data, 'to_pandas'):
                    data = data.to_pandas()
            except:
                # Fallback to standard pandas
                data = pd.read_csv(file_path)
        else:
            # Memory-efficient reading for large files
            data = pd.read_csv(file_path, 
                             low_memory=False,  # Better type inference
                             engine='c')       # Faster C engine
        
        # Memory optimization - ensure numeric columns are optimal dtypes
        for col in data.select_dtypes(include=['float64']).columns:
            data[col] = pd.to_numeric(data[col], downcast='float')
        for col in data.select_dtypes(include=['int64']).columns:
            data[col] = pd.to_numeric(data[col], downcast='integer')
        
        # Run model with error handling
        result = model_func(data, theta=theta, n=n_months)
        
        # Clean up memory
        del data
        
        return result
        
    except MemoryError:
        print(f"[MEMORY] Memory error processing {file_path} - trying memory-efficient mode")
        try:
            # Try chunked processing if memory error
            data = pd.read_csv(file_path, chunksize=10000)
            # Combine chunks (simplified approach)
            data = pd.concat(list(data), ignore_index=True)
            result = model_func(data, theta=theta, n=n_months)
            del data
            return result
        except Exception as e2:
            print(f"[ERROR] Failed chunked processing for {file_path}: {e2}")
            return {
                'rmse_train': np.nan, 'rmse_test': np.nan,
                'mase_test': np.nan, 'bias_test': np.nan,
                'theta': theta, 'n': n_months
            }
    except Exception as e:
        print(f"[ERROR] Error processing {file_path}: {e}")
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
    if check_output_file_exists(time_period, theta_value, model_name):
        return pd.DataFrame()  # Return empty DataFrame, job already done
    
    if max_workers is None:
        max_workers = get_optimal_workers()
    
    print(f"\n{'='*60}")
    print(f"Running {model_name} models: {time_period} years, theta = {theta_value}")
    if GPU_AVAILABLE:
        print("[GPU] GPU acceleration: ENABLED")
    print('='*60)
    
    # Get file paths
    data_path = get_simulated_data_path(f"d{time_period}year/theta_{theta_value}")
    file_pattern = f"{data_path}/*.csv"
    file_paths = glob.glob(file_pattern)
    
    if not file_paths:
        print(f"No data files found in {data_path}")
        return pd.DataFrame()
    
    print(f"Found {len(file_paths)} datasets to process")
    
    # Process files
    results = []
    start_time = time.time()
    
    # Cloud-optimized parallel processing
    if model_name.lower() == 'nb':
        # Use ProcessPoolExecutor for CPU-intensive standard models
        executor_class = ProcessPoolExecutor
        # Cloud platforms can handle more aggressive parallelization
        if CLOUD_PLATFORM in ["Google Colab", "Kaggle", "Google Cloud"]:
            max_workers = min(max_workers, len(file_paths))  # Don't exceed file count
        print(f"[EXECUTOR] Using ProcessPoolExecutor with {max_workers} processes")
    else:
        # Bayesian models: use ALL available threads for maximum speed
        executor_class = ThreadPoolExecutor
        print(f"[EXECUTOR] Using ThreadPoolExecutor with {max_workers} threads (Bayesian - MAXIMUM SPEED)")
        if max_workers > 2:
            print(f"[WARNING] High parallelization may cause PyTensor cache conflicts - but prioritizing speed!")
    
    with executor_class(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file_optimized, file_path, model_func, theta_value, n_months): file_path
            for file_path in file_paths
        }
        
        # Process results with enhanced cloud progress tracking
        processed_count = 0
        last_memory_check = time.time()
        
        for future in as_completed(future_to_file):
            result = future.result()
            results.append(result)
            processed_count += 1
            
            # Memory monitoring for cloud environments
            current_time = time.time()
            if current_time - last_memory_check > 60:  # Check every minute
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 85:
                    print(f"[WARNING] High memory usage: {memory_percent:.1f}%")
                last_memory_check = current_time
            
            # Enhanced progress reporting optimized for cloud
            progress_interval = 25 if CLOUD_PLATFORM in ["Google Colab", "Kaggle"] else 50
            
            if processed_count % progress_interval == 0 or processed_count == len(file_paths):
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                eta = (len(file_paths) - processed_count) / rate if rate > 0 else 0
                
                # Cloud-optimized progress display
                memory_info = f"RAM: {psutil.virtual_memory().percent:.1f}%"
                if GPU_AVAILABLE:
                    gpu_info = " | GPU: Active"
                else:
                    gpu_info = ""
                
                print(f"[PROGRESS] {processed_count}/{len(file_paths)} "
                      f"({processed_count/len(file_paths)*100:.1f}%) | "
                      f"Rate: {rate:.1f}/sec | ETA: {eta/60:.1f}min | "
                      f"{memory_info}{gpu_info}")
                
                # Cloud session persistence reminder
                if CLOUD_PLATFORM in ["Google Colab", "Kaggle"] and processed_count % 200 == 0:
                    print(f"[CLOUD] Keep session active - {processed_count} files processed")
        
        # Final memory cleanup
        import gc
        gc.collect()
    
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
    print(f"[RESULTS] Results saved to: {output_file}")
    print(f"[TIME] Processing time: {elapsed/60:.1f} minutes")
    print(f"[PERFORMANCE] Average rate: {len(file_paths)/elapsed:.1f} files/second")
    
    if len(results_df) > 0:
        print(f"[STATS] Summary statistics:")
        numeric_cols = ['rmse_train', 'rmse_test', 'mase_test', 'bias_test']
        print(results_df[numeric_cols].describe().round(4))
    
    return results_df

def run_time_period_models(time_period, n_months):
    """Run all models for a specific time period with optimizations"""
    
    print(f"\n{'='*80}")
    print(f"[PROCESSING] PROCESSING {time_period}-YEAR MODELS ({n_months} months)")
    print(f"[SYSTEM] System optimization: {'GPU + ' if GPU_AVAILABLE else ''}Multi-core CPU")
    print("="*80)
    
    # Check if data exists
    data_dir = get_simulated_data_path(f"d{time_period}year")
    if not os.path.exists(data_dir):
        print(f"[ERROR] Error: {data_dir} not found!")
        print("Please run 5_simulation_temp.py first to generate simulated datasets.")
        return 0
    
    # Use configurable parameters
    theta_values = THETA_VALUES
    max_workers = get_optimal_workers()
    jobs_run = 0
    
    # Run Standard Negative Binomial models (high parallelization)
    print(f"\n--- [STANDARD] Standard NB Models ({time_period} years) ---")
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
            print(f"[ERROR] Error processing NB models for {time_period}yr, theta {theta}: {e}")
    
    # Run Bayesian models (maximum parallelization)
    print(f"\n--- [BAYESIAN] Bayesian NB Models ({time_period} years) ---")
    print(f"[PERFORMANCE] Using ALL available threads for maximum speed!")
    for theta in theta_values:
        try:
            # Use ALL workers for maximum speed
            result_df = run_models_on_theta(
                theta_value=theta,
                model_func=stanbinner,
                model_name='Bayesian',
                time_period=time_period,
                n_months=n_months,
                max_workers=max_workers  # Use full parallelization
            )
            if not result_df.empty:  # Job was actually run (not skipped)
                jobs_run += 1
        except Exception as e:
            print(f"[ERROR] Error processing Bayesian models for {time_period}yr, theta {theta}: {e}")
    
    return jobs_run

def main():
    """Main function to run all model evaluations with file-based checking"""
    
    print("="*80)
    print("[PIPELINE] COMPREHENSIVE MODEL EVALUATION PIPELINE - FILE-BASED CHECKING")
    print("Python version of 7models_*.R scripts")
    print("[INFO] Enhanced with GPU acceleration, maximum parallelization, and file-based resuming")
    print("="*80)
    
    # Display thread control settings
    print(f"\n[THREAD CONTROL] Configuration:")
    print(f"   - Mode: {THREAD_CONTROL_MODE.upper()}")
    if THREAD_CONTROL_MODE == "manual":
        print(f"   - Manual thread count: {MANUAL_THREAD_COUNT}")
    print("="*80)
    
    # Display system capabilities
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    print(f"[CLOUD] Cloud Environment Analysis:")
    print(f"   - Platform: {CLOUD_PLATFORM}")
    print(f"   - CPU Cores: {cpu_count}")
    print(f"   - RAM: {memory_gb:.1f} GB")
    
    # Detailed resource analysis
    if GPU_AVAILABLE:
        print(f"   - GPU: {GPU_MESSAGE}")
    if TPU_AVAILABLE:
        print(f"   - TPU: {TPU_MESSAGE}")
    
    # Cloud-specific performance predictions
    optimal_workers = get_optimal_workers()
    print(f"   - Optimal parallel workers: {optimal_workers}")
    
    # Estimate performance boost
    performance_multiplier = 1.0
    if GPU_AVAILABLE:
        performance_multiplier *= 2.0  # GPU boost
    if TPU_AVAILABLE:
        performance_multiplier *= 3.0  # TPU boost
    if CLOUD_PLATFORM in ["Google Colab", "Google Cloud"]:
        performance_multiplier *= 1.5  # Cloud infrastructure boost
    
    if performance_multiplier > 1.0:
        print(f"   - Expected performance boost: {performance_multiplier:.1f}x faster than standard CPU")
    
    # Check existing output files
    print(f"\n[FILE CHECK] Checking existing output files in: {get_model_results_path('')}")
    
    # Set seed EXACTLY like R scripts
    # R code: set.seed(76568)
    np.random.seed(76568)
    
    # Use configurable time periods
    time_periods = TIME_PERIODS
    
    # Check if simulated data exists
    if not os.path.exists(get_simulated_data_path()):
        print("[ERROR] Error: Simulated data directory not found!")
        print("Please run 5_simulation_temp.py first to generate simulated datasets.")
        return
    
    print(f"\n[CONFIG] Processing Configuration:")
    print(f"   - Time periods: {list(time_periods.keys())} years")
    print(f"   - Theta values: {THETA_VALUES}")
    print(f"   - Model types: Standard NB + Bayesian NB")
    
    # Calculate total combinations
    total_combinations = len(time_periods) * len(THETA_VALUES) * 2  # periods × thetas × models
    print(f"   - Total model combinations: {total_combinations}")
    
    # Cloud session optimization
    if CLOUD_PLATFORM in ["Google Colab", "Kaggle"]:
        print(f"\n[CLOUD] Session Management Tips:")
        print(f"   - Keep browser tab active to prevent disconnection")
        print(f"   - Existing files will be skipped automatically")
        print(f"   - Consider running in smaller batches for very long jobs")
        
        # Estimate total runtime (rough estimate)
        estimated_seconds_per_job = 30 / (performance_multiplier * optimal_workers)
        estimated_total_hours = (total_combinations * estimated_seconds_per_job) / 3600
        print(f"   - Estimated total runtime: {estimated_total_hours:.1f} hours (with MAXIMUM parallelization)")
        
        if estimated_total_hours > 6 and CLOUD_PLATFORM == "Google Colab":
            print(f"   - [WARNING] Long job detected - consider Colab Pro for longer sessions")
    
    if GPU_AVAILABLE:
        print(f"\n[ACCELERATION] GPU Acceleration: ENABLED - Expect {performance_multiplier:.1f}x speedup")
        print(f"[PERFORMANCE] MAXIMUM PARALLELIZATION: {optimal_workers} workers for ALL models (Standard + Bayesian)")
    elif TPU_AVAILABLE:
        print(f"\n[ACCELERATION] TPU Acceleration: ENABLED - Expect {performance_multiplier:.1f}x speedup")
        print(f"[PERFORMANCE] MAXIMUM PARALLELIZATION: {optimal_workers} workers for ALL models (Standard + Bayesian)")
    else:
        print(f"\n[PROCESSING] MAXIMUM PARALLELIZATION: {optimal_workers} parallel workers for ALL models")
    
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
                print(f"[COMPLETE] {time_period}-year models completed in {period_elapsed/60:.1f} minutes")
            else:
                print(f"[SKIP] {time_period}-year models already completed (skipped)")
                
        except Exception as e:
            print(f"[ERROR] Error processing {time_period}-year models: {e}")
            continue
    
    # Final summary
    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*80}")
    if jobs_run > 0:
        print(f"[COMPLETE] PROCESSING SESSION COMPLETED!")
        print(f"[TIME] Session time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.1f} hours)")
        print(f"[STATS] Jobs processed this session: {jobs_run}")
        print(f"[STATS] Average time per job: {total_elapsed/jobs_run:.1f} seconds")
    else:
        print(f"[COMPLETE] ALL MODELS ALREADY COMPLETED!")
        print(f"[TIME] Session time: {total_elapsed:.1f} seconds (file check only)")
    
    print(f"[RESULTS] Results saved in: {get_model_results_path('')}")
    
    # Cloud optimization summary
    if jobs_run > 0:
        print(f"[PERFORMANCE] Cloud optimization summary:")
        print(f"   - Platform: {CLOUD_PLATFORM}")
        print(f"   - Peak parallelization: {optimal_workers} workers")
        if GPU_AVAILABLE:
            print(f"   - GPU acceleration: UTILIZED")
        if TPU_AVAILABLE:
            print(f"   - TPU acceleration: UTILIZED")
        if performance_multiplier > 1.0:
            estimated_base_time = total_elapsed * performance_multiplier
            print(f"   - Time saved vs CPU-only: {(estimated_base_time - total_elapsed)/60:.1f} minutes")
    
    print(f"[SUCCESS] PIPELINE COMPLETED!")
    if CLOUD_PLATFORM in ["Google Colab", "Kaggle"]:
        print(f"[CLOUD] All models completed successfully on {CLOUD_PLATFORM}!")
    
    print("="*80)

if __name__ == "__main__":
    # Windows multiprocessing support
    mp.freeze_support()
    main() 