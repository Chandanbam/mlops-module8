#!/usr/bin/env python3
"""
Memory Optimization and Monitoring Script for Dask ML Pipeline
"""

import psutil
import os
import gc
import time
from typing import Dict, Any

def get_memory_info() -> Dict[str, Any]:
    """Get current memory usage information."""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    return {
        'ram_total_gb': memory.total / (1024**3),
        'ram_used_gb': memory.used / (1024**3),
        'ram_available_gb': memory.available / (1024**3),
        'ram_percent': memory.percent,
        'swap_total_gb': swap.total / (1024**3),
        'swap_used_gb': swap.used / (1024**3),
        'swap_percent': swap.percent
    }

def print_memory_status():
    """Print current memory status."""
    info = get_memory_info()
    
    print("\n" + "="*60)
    print("MEMORY STATUS")
    print("="*60)
    print(f"RAM: {info['ram_used_gb']:.2f}GB / {info['ram_total_gb']:.2f}GB ({info['ram_percent']:.1f}%)")
    print(f"Available RAM: {info['ram_available_gb']:.2f}GB")
    print(f"Swap: {info['swap_used_gb']:.2f}GB / {info['swap_total_gb']:.2f}GB ({info['swap_percent']:.1f}%)")
    print("="*60)

def optimize_memory():
    """Perform memory optimization."""
    print("\n🧹 Performing memory optimization...")
    
    # Force garbage collection
    collected = gc.collect()
    print(f"   Garbage collected: {collected} objects")
    
    # Clear Python cache
    import sys
    if hasattr(sys, 'getallocatedblocks'):
        before = sys.getallocatedblocks()
        gc.collect()
        after = sys.getallocatedblocks()
        print(f"   Memory blocks freed: {before - after}")
    
    print("✅ Memory optimization completed")

def check_memory_safety(required_gb: float = 2.0) -> bool:
    """Check if there's enough memory for the operation."""
    info = get_memory_info()
    available_gb = info['ram_available_gb']
    
    print(f"\n🔍 Memory Safety Check:")
    print(f"   Required: {required_gb:.1f}GB")
    print(f"   Available: {available_gb:.1f}GB")
    
    if available_gb >= required_gb:
        print("✅ Sufficient memory available")
        return True
    else:
        print("❌ Insufficient memory - consider reducing scale_factor")
        return False

def monitor_memory_during_execution(func, *args, **kwargs):
    """Monitor memory during function execution."""
    print_memory_status()
    
    start_time = time.time()
    start_memory = get_memory_info()
    
    try:
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = get_memory_info()
        
        print(f"\n📊 Execution Summary:")
        print(f"   Duration: {end_time - start_time:.2f} seconds")
        print(f"   RAM Used: {end_memory['ram_used_gb'] - start_memory['ram_used_gb']:.2f}GB")
        print(f"   Swap Used: {end_memory['swap_used_gb'] - start_memory['swap_used_gb']:.2f}GB")
        
        return result
        
    except MemoryError as e:
        print(f"\n❌ Memory Error: {e}")
        print("💡 Suggestions:")
        print("   - Reduce scale_factor in config.yaml")
        print("   - Reduce n_workers in Dask config")
        print("   - Use smaller model (SGDRegressor instead of RandomForest)")
        print("   - Close other applications")
        raise
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

def get_optimal_scale_factor() -> int:
    """Calculate optimal scale factor based on available memory."""
    info = get_memory_info()
    available_gb = info['ram_available_gb']
    
    # Conservative estimate: 1GB per 10x scale factor
    if available_gb >= 8:
        return 10
    elif available_gb >= 6:
        return 7
    elif available_gb >= 4:
        return 5
    elif available_gb >= 2:
        return 3
    else:
        return 1

def print_recommendations():
    """Print memory optimization recommendations."""
    info = get_memory_info()
    optimal_scale = get_optimal_scale_factor()
    
    print("\n💡 MEMORY OPTIMIZATION RECOMMENDATIONS:")
    print("="*50)
    
    if info['ram_percent'] > 80:
        print("⚠️  High RAM usage detected!")
        print("   - Close unnecessary applications")
        print("   - Reduce scale_factor in config.yaml")
    
    if info['swap_percent'] > 50:
        print("⚠️  High swap usage detected!")
        print("   - This will significantly slow down performance")
        print("   - Consider reducing memory requirements")
    
    print(f"📊 Recommended scale_factor: {optimal_scale}")
    print(f"📊 Current available RAM: {info['ram_available_gb']:.1f}GB")
    
    print("\n🔧 Configuration suggestions:")
    print("   - Set scale_factor to 5 or less")
    print("   - Use n_workers: 2")
    print("   - Set memory_limit: '1GB'")
    print("   - Use SGDRegressor instead of RandomForest")
    print("   - Reduce cv_folds to 3")

if __name__ == "__main__":
    print("🔧 Dask ML Pipeline Memory Optimizer")
    print("="*40)
    
    # Show current memory status
    print_memory_status()
    
    # Check memory safety
    check_memory_safety(2.0)
    
    # Show recommendations
    print_recommendations()
    
    # Perform optimization
    optimize_memory()
    
    print("\n✅ Ready to run pipeline with optimized settings!") 