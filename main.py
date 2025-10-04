#!/usr/bin/env python3
"""
Emergency AI - Complete Production System Entry Point
Advanced real-time emergency audio analysis system with AI-powered distress detection.

This is the main production entry point that provides a unified interface
to all Emergency AI functionality including CLI, GUI, web interface, and validation tools.
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "WORKING_FILES"))

def setup_environment():
    """Set up the runtime environment."""
    # Ensure models directory exists
    models_dir = PROJECT_ROOT / "WORKING_FILES" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure logs directory exists
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables for optimal performance
    os.environ.setdefault('PYTHONPATH', str(PROJECT_ROOT))
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')  # Reduce TensorFlow verbosity
    
    # GPU memory growth for TensorFlow
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except ImportError:
        pass  # TensorFlow not available


def run_cli():
    """Run the command-line interface."""
    try:
        from WORKING_FILES.cli import main
        main()
    except ImportError as e:
        print(f"Error: CLI module not available: {e}")
        sys.exit(1)


def run_gui():
    """Run the graphical user interface."""
    try:
        from WORKING_FILES.gui import main
        main()
    except ImportError as e:
        print(f"Error: GUI module not available: {e}")
        print("Make sure tkinter is installed: pip install tkinter")
        sys.exit(1)


def run_web(port: int = 8501, host: str = "localhost"):
    """Run the Streamlit web interface."""
    try:
        streamlit_script = PROJECT_ROOT / "WORKING_FILES" / "app_streamlit.py"
        if not streamlit_script.exists():
            print(f"Error: Streamlit app not found at {streamlit_script}")
            sys.exit(1)
        
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(streamlit_script),
            "--server.port", str(port),
            "--server.address", host,
            "--server.headless", "true"
        ]
        
        print(f"Starting Emergency AI web interface at http://{host}:{port}")
        subprocess.run(cmd)
        
    except FileNotFoundError:
        print("Error: Streamlit not installed. Install with: pip install streamlit")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nWeb interface stopped")


def run_validate():
    """Run system validation."""
    try:
        from WORKING_FILES.validate import main
        main()
    except ImportError as e:
        print(f"Error: Validation module not available: {e}")
        sys.exit(1)


def run_analyze(audio_file: str, output: Optional[str] = None):
    """Run audio analysis on a single file."""
    try:
        from WORKING_FILES.analysis_pipeline import process_audio_file
        
        print(f"Analyzing: {audio_file}")
        result = process_audio_file(audio_file)
        
        if output:
            import json
            with open(output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"Results saved to: {output}")
        else:
            # Print summary
            print("\nAnalysis Results:")
            print(f"  Confidence: {result.get('confidence', 0):.3f}")
            print(f"  Distress Level: {result.get('distress_score', 0):.3f}")
            
            transcript = result.get('transcript', '')
            if transcript:
                print(f"  Transcript: {transcript}")
            
            emotions = result.get('emotions', {})
            if emotions:
                print("  Top Emotions:")
                for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]:
                    print(f"    {emotion}: {score:.3f}")
    
    except ImportError as e:
        print(f"Error: Analysis module not available: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Analysis failed: {e}")
        sys.exit(1)


def main():
    """Main entry point with command routing."""
    setup_environment()
    
    parser = argparse.ArgumentParser(
        description="Emergency AI - Advanced Real-time Emergency Audio Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  emergency-ai cli analyze audio.wav        # Analyze single file via CLI
  emergency-ai gui                          # Launch graphical interface
  emergency-ai web --port 8080              # Start web interface on port 8080
  emergency-ai validate                     # Run system validation
  emergency-ai analyze audio.wav -o out.json # Quick analysis with JSON output

For more detailed usage, run:
  emergency-ai cli --help                   # CLI help
  emergency-ai gui --help                   # GUI help
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # CLI command
    cli_parser = subparsers.add_parser('cli', help='Command-line interface')
    cli_parser.set_defaults(func=lambda args: run_cli())
    
    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Graphical user interface')
    gui_parser.set_defaults(func=lambda args: run_gui())
    
    # Web command
    web_parser = subparsers.add_parser('web', help='Web interface (Streamlit)')
    web_parser.add_argument('--port', '-p', type=int, default=8501, help='Port number')
    web_parser.add_argument('--host', default='localhost', help='Host address')
    web_parser.set_defaults(func=lambda args: run_web(args.port, args.host))
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='System validation')
    validate_parser.set_defaults(func=lambda args: run_validate())
    
    # Quick analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Quick audio analysis')
    analyze_parser.add_argument('audio_file', help='Audio file to analyze')
    analyze_parser.add_argument('--output', '-o', help='Output file for results')
    analyze_parser.set_defaults(func=lambda args: run_analyze(args.audio_file, args.output))
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        # No command specified, show help and suggest default
        parser.print_help()
        print("\n" + "="*60)
        print("Quick Start:")
        print("  emergency-ai gui        # Launch graphical interface")
        print("  emergency-ai web        # Launch web interface")
        print("  emergency-ai validate   # Check system setup")
        print("="*60)
        return
    
    # Execute the selected command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()