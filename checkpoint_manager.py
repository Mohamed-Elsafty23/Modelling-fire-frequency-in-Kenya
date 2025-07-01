#!/usr/bin/env python3
"""
Checkpoint Management Utility
Manage model checkpoints for the fire frequency modeling pipeline
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import argparse

def load_checkpoint_data(checkpoint_dir="model_checkpoints"):
    """Load checkpoint data"""
    checkpoint_file = Path(checkpoint_dir) / "checkpoint_status.json"
    metadata_file = Path(checkpoint_dir) / "checkpoint_metadata.json"
    
    if not checkpoint_file.exists():
        return {}, {}
    
    try:
        with open(checkpoint_file, 'r') as f:
            completed_jobs = json.load(f)
        
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        return completed_jobs, metadata
    except Exception as e:
        print(f"Error loading checkpoint data: {e}")
        return {}, {}

def show_status():
    """Show current checkpoint status"""
    completed_jobs, metadata = load_checkpoint_data()
    
    if not completed_jobs:
        print("📝 No checkpoints found - starting fresh")
        return
    
    print(f"💾 Checkpoint Status Summary")
    print("="*50)
    
    # Show metadata
    if metadata:
        print(f"Last updated: {metadata.get('last_updated', 'Unknown')}")
        print(f"Total completed: {metadata.get('total_completed', len(completed_jobs))}")
        print()
    
    # Group by time period and model
    by_period = {}
    total_time = 0
    
    for job_key, job_info in completed_jobs.items():
        period = job_info['time_period']
        model = job_info['model_name']
        
        if period not in by_period:
            by_period[period] = {}
        if model not in by_period[period]:
            by_period[period][model] = []
        
        by_period[period][model].append({
            'theta': job_info['theta_value'],
            'time': job_info['processing_time'],
            'files': job_info['file_count'],
            'completed': job_info['completed_at']
        })
        
        total_time += job_info['processing_time']
    
    # Display summary
    for period in sorted(by_period.keys()):
        print(f"📊 {period}-year models:")
        for model in sorted(by_period[period].keys()):
            jobs = by_period[period][model]
            thetas = [str(job['theta']) for job in jobs]
            avg_time = sum(job['time'] for job in jobs) / len(jobs)
            total_files = sum(job['files'] for job in jobs)
            
            print(f"   • {model.upper()}: θ = [{', '.join(thetas)}]")
            print(f"     - Avg time: {avg_time/60:.1f} min")
            print(f"     - Total files: {total_files:,}")
        print()
    
    print(f"⏱️  Total processing time saved: {total_time/3600:.1f} hours")
    print(f"📁 Checkpoint files located in: model_checkpoints/")

def clear_checkpoints():
    """Clear all checkpoints"""
    checkpoint_dir = Path("model_checkpoints")
    
    if not checkpoint_dir.exists():
        print("📝 No checkpoint directory found")
        return
    
    try:
        # Remove checkpoint files
        for file in checkpoint_dir.glob("*.json"):
            file.unlink()
        
        print("🗑️  All checkpoints cleared successfully")
        print("Next run will start from scratch")
        
    except Exception as e:
        print(f"❌ Error clearing checkpoints: {e}")

def remove_specific_checkpoint(job_pattern):
    """Remove specific checkpoint matching pattern"""
    completed_jobs, metadata = load_checkpoint_data()
    
    if not completed_jobs:
        print("📝 No checkpoints found")
        return
    
    # Find matching jobs
    matching_jobs = []
    for job_key, job_info in completed_jobs.items():
        if job_pattern.lower() in job_key.lower():
            matching_jobs.append((job_key, job_info))
    
    if not matching_jobs:
        print(f"❌ No checkpoints found matching: {job_pattern}")
        return
    
    print(f"🎯 Found {len(matching_jobs)} matching checkpoint(s):")
    for job_key, job_info in matching_jobs:
        print(f"   • {job_key}")
        print(f"     - Completed: {job_info['completed_at']}")
        print(f"     - Time: {job_info['processing_time']/60:.1f} min")
    
    # Remove matching jobs
    for job_key, _ in matching_jobs:
        del completed_jobs[job_key]
    
    # Save updated checkpoints
    checkpoint_file = Path("model_checkpoints") / "checkpoint_status.json"
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(completed_jobs, f, indent=2)
        
        print(f"✅ Removed {len(matching_jobs)} checkpoint(s)")
        print(f"These jobs will be re-run on next execution")
        
    except Exception as e:
        print(f"❌ Error saving updated checkpoints: {e}")

def validate_checkpoints():
    """Validate that checkpoint files still exist"""
    completed_jobs, metadata = load_checkpoint_data()
    
    if not completed_jobs:
        print("📝 No checkpoints to validate")
        return
    
    print("🔍 Validating checkpoint integrity...")
    
    missing_files = []
    valid_jobs = {}
    
    for job_key, job_info in completed_jobs.items():
        output_file = job_info.get('output_file')
        
        if output_file and os.path.exists(output_file):
            valid_jobs[job_key] = job_info
        else:
            missing_files.append((job_key, output_file))
    
    if missing_files:
        print(f"⚠️  Found {len(missing_files)} checkpoint(s) with missing output files:")
        for job_key, output_file in missing_files:
            print(f"   • {job_key}: {output_file}")
        
        # Update checkpoint file to remove invalid entries
        checkpoint_file = Path("model_checkpoints") / "checkpoint_status.json"
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(valid_jobs, f, indent=2)
            
            print(f"🔧 Cleaned {len(missing_files)} invalid checkpoint(s)")
            print(f"✅ {len(valid_jobs)} valid checkpoint(s) remaining")
            
        except Exception as e:
            print(f"❌ Error updating checkpoints: {e}")
    else:
        print(f"✅ All {len(completed_jobs)} checkpoint(s) are valid")

def main():
    parser = argparse.ArgumentParser(description="Manage model checkpoints")
    parser.add_argument('action', choices=['status', 'clear', 'remove', 'validate'], 
                       help='Action to perform')
    parser.add_argument('--pattern', help='Pattern to match for remove action')
    
    args = parser.parse_args()
    
    if args.action == 'status':
        show_status()
    elif args.action == 'clear':
        clear_checkpoints()
    elif args.action == 'remove':
        if not args.pattern:
            print("❌ --pattern required for remove action")
            sys.exit(1)
        remove_specific_checkpoint(args.pattern)
    elif args.action == 'validate':
        validate_checkpoints()

if __name__ == "__main__":
    main() 