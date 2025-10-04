# Emergency AI CLI Module
# Command-line interface for Emergency AI operations

import click
import sys
import os
from pathlib import Path
from typing import Optional
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from modules.enhanced_logger import get_logger
from modules.config_manager import get_config_manager
from analysis_pipeline import process_audio_file
from tests.stress_test_suite import run_stress_tests, run_regression_tests
from benchmarks.performance_profiler import run_performance_benchmarks, run_memory_profiling


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), 
              default='INFO', help='Set logging level')
@click.pass_context
def cli(ctx, config, verbose, log_level):
    """Emergency AI - Command Line Interface
    
    Advanced real-time emergency audio analysis system with AI-powered distress detection.
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['log_level'] = log_level
    
    # Initialize configuration
    config_manager = get_config_manager()
    if config:
        config_manager.load_config(config)
    
    # Initialize logger
    logger = get_logger()
    logger.info(f"Emergency AI CLI started with log level: {log_level}")


@cli.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for results (JSON)')
@click.option('--format', 'output_format', type=click.Choice(['json', 'text']), 
              default='text', help='Output format')
@click.option('--confidence-threshold', type=float, default=0.5, 
              help='Confidence threshold for results')
@click.pass_context
def analyze(ctx, audio_file, output, output_format, confidence_threshold):
    """Analyze an audio file for emergency indicators.
    
    AUDIO_FILE: Path to the audio file to analyze
    """
    logger = get_logger()
    logger.info(f"Analyzing audio file: {audio_file}")
    
    try:
        # Process the audio file
        result = process_audio_file(audio_file)
        
        # Filter results by confidence threshold
        if result.get('confidence', 0) < confidence_threshold:
            if ctx.obj['verbose']:
                click.echo(f"Warning: Confidence {result.get('confidence', 0):.3f} below threshold {confidence_threshold}")
        
        # Format output
        if output_format == 'json':
            output_data = json.dumps(result, indent=2)
        else:
            output_data = format_text_output(result)
        
        # Save or display results
        if output:
            with open(output, 'w') as f:
                f.write(output_data)
            click.echo(f"Results saved to: {output}")
        else:
            click.echo(output_data)
        
        # Exit code based on distress level
        distress_score = result.get('distress_score', 0)
        if distress_score > 0.8:
            sys.exit(2)  # High distress
        elif distress_score > 0.5:
            sys.exit(1)  # Medium distress
        else:
            sys.exit(0)  # Low/no distress
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(3)


@cli.command()
@click.option('--duration', '-d', type=int, default=60, help='Recording duration in seconds')
@click.option('--output-dir', type=click.Path(), help='Directory to save recordings')
@click.option('--real-time', is_flag=True, help='Enable real-time analysis')
@click.pass_context
def stream(ctx, duration, output_dir, real_time):
    """Stream audio from microphone and analyze in real-time.
    
    Records audio from the default microphone and provides live analysis.
    """
    logger = get_logger()
    logger.info(f"Starting audio streaming for {duration} seconds")
    
    try:
        from modules.streaming_audio import StreamingProcessor
        
        processor = StreamingProcessor()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Start streaming
        click.echo(f"Recording for {duration} seconds...")
        
        if real_time:
            click.echo("Real-time analysis enabled. Press Ctrl+C to stop.")
            # Implement real-time streaming
            processor.start_real_time_analysis(duration)
        else:
            # Record and then analyze
            audio_file = processor.record_audio(duration, output_dir)
            click.echo(f"Recording saved: {audio_file}")
            
            # Analyze the recording
            result = process_audio_file(audio_file)
            click.echo(format_text_output(result))
        
    except KeyboardInterrupt:
        click.echo("\nStreaming stopped by user")
    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--suite', type=click.Choice(['stress', 'regression', 'benchmark', 'all']), 
              default='all', help='Test suite to run')
@click.option('--output', '-o', type=click.Path(), help='Output file for test results')
@click.option('--parallel', is_flag=True, help='Run tests in parallel')
@click.pass_context
def test(ctx, suite, output, parallel):
    """Run test suites for Emergency AI system.
    
    Executes comprehensive testing including stress tests, regression tests, and benchmarks.
    """
    logger = get_logger()
    logger.info(f"Running {suite} test suite")
    
    results = {}
    
    try:
        if suite in ['stress', 'all']:
            click.echo("Running stress tests...")
            stress_results = run_stress_tests()
            results['stress_tests'] = stress_results
        
        if suite in ['regression', 'all']:
            click.echo("Running regression tests...")
            regression_results = run_regression_tests()
            results['regression_tests'] = regression_results
        
        if suite in ['benchmark', 'all']:
            click.echo("Running performance benchmarks...")
            
            # Find sample audio files
            audio_samples = []
            sample_dir = Path("WORKING_FILES/audio_samples")
            if sample_dir.exists():
                audio_samples = list(sample_dir.glob("*.wav"))
            
            if audio_samples:
                benchmark_results = run_performance_benchmarks([str(f) for f in audio_samples])
                results['benchmarks'] = benchmark_results
            else:
                click.echo("Warning: No audio samples found for benchmarking")
        
        # Display summary
        click.echo("\n" + "="*50)
        click.echo("TEST RESULTS SUMMARY")
        click.echo("="*50)
        
        for test_type, result in results.items():
            if hasattr(result, 'passed') and hasattr(result, 'total_tests'):
                success_rate = (result.passed / result.total_tests) * 100
                click.echo(f"{test_type}: {result.passed}/{result.total_tests} passed ({success_rate:.1f}%)")
        
        # Save results if requested
        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            click.echo(f"\nDetailed results saved to: {output}")
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--type', 'profile_type', type=click.Choice(['performance', 'memory', 'all']), 
              default='performance', help='Type of profiling to run')
@click.option('--audio-files', multiple=True, type=click.Path(exists=True), 
              help='Audio files to profile (can specify multiple)')
@click.option('--output', '-o', type=click.Path(), help='Output file for profiling results')
@click.pass_context
def profile(ctx, profile_type, audio_files, output):
    """Profile system performance and resource usage.
    
    Provides detailed analysis of processing performance, memory usage, and bottlenecks.
    """
    logger = get_logger()
    logger.info(f"Running {profile_type} profiling")
    
    try:
        results = {}
        
        if profile_type in ['performance', 'all']:
            click.echo("Running performance profiling...")
            if audio_files:
                perf_results = run_performance_benchmarks(list(audio_files))
                results['performance'] = perf_results
            else:
                click.echo("Warning: No audio files specified for performance profiling")
        
        if profile_type in ['memory', 'all']:
            click.echo("Running memory profiling...")
            memory_results = run_memory_profiling()
            results['memory'] = memory_results
        
        # Display summary
        click.echo("\n" + "="*50)
        click.echo("PROFILING RESULTS")
        click.echo("="*50)
        
        if 'performance' in results:
            perf = results['performance']
            click.echo(f"Average processing time: {perf.get('avg_processing_time_ms', 0):.1f}ms")
            click.echo(f"Average memory usage: {perf.get('avg_memory_usage_mb', 0):.1f}MB")
        
        if 'memory' in results:
            memory = results['memory']
            if 'model_loading' in memory:
                model_mem = memory['model_loading']
                click.echo(f"Model loading memory: {model_mem.get('total_model_memory_mb', 0):.1f}MB")
        
        # Save results if requested
        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            click.echo(f"\nDetailed results saved to: {output}")
        
    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--port', '-p', type=int, default=8501, help='Port for Streamlit interface')
@click.option('--host', '-h', default='localhost', help='Host address to bind')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.pass_context
def serve(ctx, port, host, debug):
    """Start the Streamlit web interface.
    
    Launches the Emergency AI web interface for interactive analysis.
    """
    logger = get_logger()
    logger.info(f"Starting Streamlit interface on {host}:{port}")
    
    try:
        import subprocess
        
        cmd = [
            "streamlit", "run", 
            str(Path(__file__).parent / "app_streamlit.py"),
            "--server.port", str(port),
            "--server.address", host,
            "--server.headless", "true" if not debug else "false"
        ]
        
        if debug:
            cmd.extend(["--server.runOnSave", "true"])
        
        click.echo(f"Starting web interface at http://{host}:{port}")
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        click.echo("\nWeb interface stopped")
    except Exception as e:
        logger.error(f"Failed to start web interface: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def doctor(ctx):
    """Run system diagnostics and health checks.
    
    Performs comprehensive system analysis to identify potential issues.
    """
    logger = get_logger()
    logger.info("Running system diagnostics")
    
    click.echo("Emergency AI System Diagnostics")
    click.echo("=" * 40)
    
    # Check Python environment
    click.echo(f"Python version: {sys.version}")
    click.echo(f"Python executable: {sys.executable}")
    
    # Check dependencies
    try:
        import numpy as np
        import librosa
        import tensorflow as tf
        import streamlit as st
        
        click.echo("✓ Core dependencies installed")
        click.echo(f"  - NumPy: {np.__version__}")
        click.echo(f"  - Librosa: {librosa.__version__}")
        click.echo(f"  - TensorFlow: {tf.__version__}")
        click.echo(f"  - Streamlit: {st.__version__}")
        
    except ImportError as e:
        click.echo(f"✗ Missing dependency: {e}")
    
    # Check models
    models_dir = Path("WORKING_FILES/models")
    if models_dir.exists():
        model_count = len(list(models_dir.rglob("*")))
        click.echo(f"✓ Models directory found ({model_count} files)")
    else:
        click.echo("✗ Models directory not found")
    
    # Check configuration
    try:
        config_manager = get_config_manager()
        click.echo("✓ Configuration loaded successfully")
    except Exception as e:
        click.echo(f"✗ Configuration error: {e}")
    
    # Check system resources
    import psutil
    memory = psutil.virtual_memory()
    click.echo(f"System memory: {memory.total // (1024**3)}GB total, {memory.available // (1024**3)}GB available")
    click.echo(f"CPU cores: {psutil.cpu_count()}")
    
    click.echo("\nDiagnostics completed")


def format_text_output(result):
    """Format analysis results as human-readable text."""
    output = []
    output.append("Emergency AI Analysis Results")
    output.append("=" * 40)
    
    # Basic metrics
    output.append(f"Confidence Score: {result.get('confidence', 0):.3f}")
    output.append(f"Distress Level: {result.get('distress_score', 0):.3f}")
    
    # Transcript
    transcript = result.get('transcript', '')
    if transcript:
        output.append(f"Transcript: {transcript}")
    
    # Emotions
    emotions = result.get('emotions', {})
    if emotions:
        output.append("Detected Emotions:")
        for emotion, score in emotions.items():
            if score > 0.1:  # Only show significant emotions
                output.append(f"  - {emotion.title()}: {score:.3f}")
    
    # Processing time
    processing_time = result.get('processing_time_ms', 0)
    if processing_time:
        output.append(f"Processing Time: {processing_time:.1f}ms")
    
    return "\n".join(output)


def main():
    """Main CLI entry point."""
    cli()


if __name__ == '__main__':
    main()