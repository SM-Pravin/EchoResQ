#!/usr/bin/env python3
"""
Simple benchmark runner for Emergency AI performance testing.
Run this script to quickly test all performance improvements.
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run Emergency AI benchmarks")
    parser.add_argument("--type", choices=["quick", "full", "stress"], default="quick",
                       help="Type of benchmark to run")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers to test")
    parser.add_argument("--batch", action="store_true",
                       help="Enable batch processing")
    parser.add_argument("--gpu", action="store_true",
                       help="Enable GPU processing")
    
    args = parser.parse_args()
    
    # Set environment based on arguments
    if args.workers:
        os.environ["PARALLEL_MAX_WORKERS"] = str(args.workers)
    
    if args.batch:
        os.environ["ENABLE_BATCH_PROCESSING"] = "true"
    else:
        os.environ["ENABLE_BATCH_PROCESSING"] = "false"
    
    if not args.gpu:
        os.environ["FORCE_CPU"] = "true"
    
    # Suppress verbose logging
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("VOSK_LOG_LEVEL", "-1")
    
    print("Emergency AI Performance Benchmarks")
    print("="*50)
    print(f"Benchmark Type: {args.type}")
    print(f"Workers: {args.workers or 'auto'}")
    print(f"Batch Processing: {args.batch}")
    print(f"GPU Enabled: {args.gpu}")
    print("="*50)
    
    if args.type == "quick":
        print("Running quick parallel benchmark...")
        from tests.bench_parallel import benchmark_configurations
        benchmark_configurations()
        
    elif args.type == "full":
        print("Running comprehensive performance benchmark...")
        from benchmarks.performance_profiler import run_full_benchmark
        run_full_benchmark()
        
    elif args.type == "stress":
        print("Running stress testing suite...")
        from benchmarks.stress_test import StressTestRunner
        runner = StressTestRunner()
        runner.run_full_stress_test()
    
    print("\nBenchmarking completed!")

if __name__ == "__main__":
    main()
