#!/usr/bin/env python3
"""
Main Fire Frequency Modeling Pipeline
=====================================

Comprehensive orchestrator for the Kenya Fire Frequency Modeling Project
Runs all steps in sequence, skipping completed steps automatically.

Usage:
    python main.py [--force] [--skip-optional]
    
Options:
    --force: Force re-run all steps (ignore existing outputs)
    --skip-optional: Skip optional model building and fitting steps
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

class FireModelingPipeline:
    """Main pipeline orchestrator for fire frequency modeling project"""
    
    def __init__(self, force_rerun=False, skip_optional=False):
        self.force_rerun = force_rerun
        self.skip_optional = skip_optional
        self.start_time = datetime.now()
        self.completed_steps = []
        self.failed_steps = []
        
        # Create output directory
        self.output_dir = "our_output"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/model_results", exist_ok=True)
        os.makedirs(f"{self.output_dir}/simulated_data", exist_ok=True)
        os.makedirs(f"{self.output_dir}/descriptive_plots", exist_ok=True)
        
        # Define the pipeline steps
        self.steps = [
            {
                'name': 'Data Import & Merge',
                'script': '1_import_merge.py',
                'output_files': [f'{self.output_dir}/fire_data_merged.csv'],
                'description': 'Import MODIS fire data and climate rasters, perform spatial intersection'
            },
            {
                'name': 'Data Aggregation',
                'script': '2_data_aggregate.py',
                'output_files': [f'{self.output_dir}/fire_data_2000-18.csv'],
                'description': 'Aggregate monthly fire-climate data into final analysis dataset'
            },
            {
                'name': 'Seasonality Check',
                'script': '3_check_seasonality.py',
                'output_files': [f'{self.output_dir}/seasonality_analysis.png', f'{self.output_dir}/seasonal_decomposition.csv'],
                'description': 'Analyze seasonal patterns in climate variables'
            },
            {
                'name': 'Data Simulation',
                'script': '5_simulation_temp.py',
                'output_files': [f'{self.output_dir}/simulated_data'],
                'description': 'Generate synthetic fire-climate datasets for model testing'
            },
            {
                'name': 'Model Evaluation',
                'script': 'run_all_models.py',
                'output_files': [f'{self.output_dir}/model_results'],
                'description': 'Run Standard and Bayesian NB models on all time periods'
            },
            {
                'name': 'Descriptive Statistics',
                'script': '8_descriptive_stats.py',
                'output_files': [f'{self.output_dir}/descriptive_plots', f'{self.output_dir}/summary_statistics.csv'],
                'description': 'Generate comprehensive data visualizations and statistics'
            },
            {
                'name': 'Simulation Visualization',
                'script': '9_sim_visualization.py',
                'output_files': [f'{self.output_dir}/simulation_comparison_plots.png', f'{self.output_dir}/model_performance_summary.csv'],
                'description': 'Create model comparison visualizations and performance analysis'
            }
        ]
        
        # Optional steps (run only if not skipped)
        self.optional_steps = [
            {
                'name': 'Model Building Analysis',
                'script': 'model_building.py',
                'output_files': [f'{self.output_dir}/simulation_results.csv', f'{self.output_dir}/fire_count_distribution.png'],
                'description': 'Comprehensive model building and overdispersion analysis'
            },
            {
                'name': 'Advanced Model Fitting',
                'script': 'model_fitting.py',
                'output_files': [f'{self.output_dir}/model_fitting_results.csv', f'{self.output_dir}/model_comparison.png', f'{self.output_dir}/model_results/fitted_models'],
                'description': 'Advanced Bayesian modeling with MCMC and comprehensive fitting'
            }
        ]
        
        if not skip_optional:
            self.steps.extend(self.optional_steps)
    

    
    def print_header(self):
        """Print pipeline header"""
        print("=" * 80)
        print("🔥 KENYA FIRE FREQUENCY MODELING PIPELINE")
        print("=" * 80)
        print(f"📅 Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📂 Working Directory: {os.getcwd()}")
        print(f"🔧 Force Rerun: {'Yes' if self.force_rerun else 'No'}")
        print(f"⏭️  Skip Optional: {'Yes' if self.skip_optional else 'No'}")
        print(f"📊 Total Steps: {len(self.steps)}")
        print("=" * 80)
    
    def check_step_completed(self, step):
        """Check if a step has been completed by looking for output files"""
        if self.force_rerun:
            return False
        
        output_files = step['output_files']
        
        for file_pattern in output_files:
            # Check for exact file
            if os.path.exists(file_pattern):
                continue
            
            # Check for directory
            if os.path.isdir(file_pattern):
                continue
                
            # Check for pattern match (simple glob)
            if '*' in file_pattern:
                import glob
                matches = glob.glob(file_pattern)
                if matches:
                    continue
            
            # If any required output is missing, step is not completed
            return False
        
        return True
    
    def run_step(self, step, step_num, total_steps):
        """Run a single pipeline step"""
        step_name = step['name']
        script = step['script']
        description = step['description']
        
        print(f"\n📍 STEP {step_num}/{total_steps}: {step_name}")
        print("─" * 60)
        print(f"📝 Description: {description}")
        print(f"🐍 Script: {script}")
        
        # Check if step is already completed
        if self.check_step_completed(step):
            print("✅ SKIPPED - Step already completed")
            print(f"   Found outputs: {', '.join(step['output_files'])}")
            self.completed_steps.append(step_name)
            return True
        
        # Check if script exists
        if not os.path.exists(script):
            print(f"❌ ERROR - Script not found: {script}")
            self.failed_steps.append(step_name)
            return False
        
        print(f"🚀 RUNNING - {script}")
        
        try:
            # Run the script with output directory environment variable
            start_time = time.time()
            env = os.environ.copy()
            env['OUTPUT_DIR'] = self.output_dir
            
            result = subprocess.run([
                sys.executable, script
            ], capture_output=True, text=True, env=env)
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ COMPLETED in {duration:.1f}s")
                self.completed_steps.append(step_name)
                
                # Show some output (last few lines)
                if result.stdout:
                    output_lines = result.stdout.strip().split('\n')
                    if len(output_lines) > 3:
                        print("📋 Last output lines:")
                        for line in output_lines[-3:]:
                            print(f"   {line}")
                
                return True
            else:
                print(f"❌ FAILED with return code {result.returncode}")
                print("📋 Error output:")
                if result.stderr:
                    error_lines = result.stderr.strip().split('\n')
                    for line in error_lines[-5:]:  # Show last 5 error lines
                        print(f"   {line}")
                
                self.failed_steps.append(step_name)
                return False
                
        except Exception as e:
            print(f"❌ EXCEPTION - {str(e)}")
            self.failed_steps.append(step_name)
            return False
    
    def create_project_summary(self):
        """Create a summary of the project results"""
        summary_file = f"{self.output_dir}/project_summary.md"
        
        with open(summary_file, 'w') as f:
            f.write("# Fire Frequency Modeling Project - Execution Summary\n\n")
            f.write(f"**Execution Date:** {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Execution Time:** {(datetime.now() - self.start_time).total_seconds():.1f} seconds\n\n")
            
            f.write("## Completed Steps\n\n")
            for step in self.completed_steps:
                f.write(f"- ✅ {step}\n")
            
            if self.failed_steps:
                f.write("\n## Failed Steps\n\n")
                for step in self.failed_steps:
                    f.write(f"- ❌ {step}\n")
            
            f.write("\n## Generated Files\n\n")
            
            # List key output files
            key_files = [
                f'{self.output_dir}/fire_data_2000-18.csv',
                f'{self.output_dir}/seasonality_analysis.png',
                f'{self.output_dir}/model_performance_summary.csv',
                f'{self.output_dir}/simulation_comparison_plots.png'
            ]
            
            for file in key_files:
                if os.path.exists(file):
                    file_size = os.path.getsize(file)
                    f.write(f"- 📄 {file} ({file_size:,} bytes)\n")
            
            # Check for directories
            key_dirs = [f'{self.output_dir}/simulated_data', f'{self.output_dir}/model_results', f'{self.output_dir}/descriptive_plots']
            for dir_name in key_dirs:
                if os.path.isdir(dir_name):
                    file_count = len([f for f in os.listdir(dir_name) 
                                    if os.path.isfile(os.path.join(dir_name, f))])
                    f.write(f"- 📁 {dir_name}/ ({file_count} files)\n")
        
        print(f"📋 Project summary saved to: {summary_file}")
    
    def run_pipeline(self):
        """Run the complete pipeline"""
        self.print_header()
        
        total_steps = len(self.steps)
        success_count = 0
        
        # Check dependencies
        self.check_dependencies()
        
        # Run each step
        for i, step in enumerate(self.steps, 1):
            success = self.run_step(step, i, total_steps)
            if success:
                success_count += 1
            else:
                # Ask user if they want to continue
                response = input(f"\n⚠️  Step failed. Continue with remaining steps? [y/N]: ")
                if response.lower() != 'y':
                    break
        
        # Print final summary
        self.print_final_summary(success_count, total_steps)
        
        # Create project summary
        self.create_project_summary()
        
        return len(self.failed_steps) == 0
    
    def check_dependencies(self):
        """Check for required dependencies"""
        print("\n🔍 CHECKING DEPENDENCIES")
        print("─" * 40)
        
        # Check Python version
        python_version = sys.version_info
        print(f"🐍 Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check key packages
        # required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn', 'statsmodels']
        # missing_packages = []
        
        # for package in required_packages:
        #     try:
        #         __import__(package)
        #         print(f"✅ {package}")
        #     except ImportError:
        #         print(f"❌ {package} - MISSING")
        #         missing_packages.append(package)
        
        # if missing_packages:
        #     print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        #     print("💡 Install with: pip install -r requirements.txt")
        #     response = input("Continue anyway? [y/N]: ")
        #     if response.lower() != 'y':
                # sys.exit(1)
        
        # Check for data directories
        data_dirs = ['fire', 'climate', 'tmax', 'tmin', 'rain']
        for dir_name in data_dirs:
            if os.path.isdir(dir_name):
                file_count = len(os.listdir(dir_name))
                print(f"📁 {dir_name}/ ({file_count} files)")
            else:
                print(f"❌ {dir_name}/ - Directory not found")
    
    def print_final_summary(self, success_count, total_steps):
        """Print final execution summary"""
        duration = datetime.now() - self.start_time
        
        print("\n" + "=" * 80)
        print("📊 PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        print(f"⏱️  Total Time: {duration.total_seconds():.1f} seconds")
        print(f"✅ Successful Steps: {success_count}/{total_steps}")
        print(f"❌ Failed Steps: {len(self.failed_steps)}")
        
        if self.completed_steps:
            print(f"\n🎉 Completed Steps:")
            for step in self.completed_steps:
                print(f"   ✅ {step}")
        
        if self.failed_steps:
            print(f"\n💥 Failed Steps:")
            for step in self.failed_steps:
                print(f"   ❌ {step}")
        
        if success_count == total_steps:
            print(f"\n🎊 SUCCESS! All {total_steps} steps completed successfully!")
            print("🔬 Your fire frequency modeling analysis is ready!")
        else:
            print(f"\n⚠️  PARTIAL SUCCESS: {success_count}/{total_steps} steps completed")
        
        print("=" * 80)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run the complete Fire Frequency Modeling Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run pipeline with force=True (default)
  python main.py --no-force         # Run without forcing re-runs
  python main.py --skip-optional    # Skip optional model building steps
        """
    )
    
    parser.add_argument('--no-force', action='store_true',
                       help='Do not force re-run all steps (ignore existing outputs)')
    parser.add_argument('--skip-optional', action='store_true',
                       help='Skip optional model building and fitting steps')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = FireModelingPipeline(
        force_rerun=not args.no_force,  # Invert the no_force flag
        skip_optional=args.skip_optional
    )
    success = pipeline.run_pipeline()
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 