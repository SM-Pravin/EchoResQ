#!/usr/bin/env python3
"""
Emergency AI - Dependency Installation and Verification Script
Checks for and installs missing dependencies required for Emergency AI.
"""

import sys
import subprocess
import importlib
import pkg_resources
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8+ required. Current version:", sys.version)
        return False
    print("[OK] Python version:", sys.version.split()[0])
    return True


def install_package(package_name):
    """Install a package using pip."""
    try:
        print(f"📦 Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"[OK] Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install {package_name}: {e}")
        return False


def check_and_install_package(package_name, import_name=None):
    """Check if a package is installed, install if missing."""
    if import_name is None:
        import_name = package_name.replace("-", "_")
    
    try:
        importlib.import_module(import_name)
        print(f"[OK] {package_name} is available")
        return True
    except ImportError:
        print(f"[WARNING] {package_name} not found, attempting to install...")
        return install_package(package_name)


def main():
    """Main dependency check and installation function."""
    print("[EMERGENCY] Emergency AI - Dependency Verification")
    print("=" * 50)
    
    if not check_python_version():
        sys.exit(1)
    
    # Core dependencies that are critical
    core_dependencies = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("streamlit", "streamlit"),
        ("plotly", "plotly"),
        ("loguru", "loguru"),
        ("psutil", "psutil"),
        ("pyyaml", "yaml"),
        ("click", "click"),
        ("rich", "rich"),
    ]
    
    # Optional dependencies that can fail gracefully
    optional_dependencies = [
        ("plotly", "plotly.express"),
        ("plotly", "plotly.graph_objects"),
        ("memory-profiler", "memory_profiler"),
        ("streamlit-webrtc", "streamlit_webrtc"),
        ("pyaudio", "pyaudio"),
    ]
    
    # Audio/ML dependencies
    ml_dependencies = [
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("scikit-learn", "sklearn"),
        ("transformers", "transformers"),
    ]
    
    print("\n[CONFIG] Checking Core Dependencies...")
    core_success = True
    for package, import_name in core_dependencies:
        if not check_and_install_package(package, import_name):
            core_success = False
    
    print("\n[TARGET] Checking ML/Audio Dependencies...")
    ml_success = True
    for package, import_name in ml_dependencies:
        if not check_and_install_package(package, import_name):
            ml_success = False
    
    print("\n[STAR] Checking Optional Dependencies...")
    for package, import_name in optional_dependencies:
        check_and_install_package(package, import_name)
    
    print("\n" + "=" * 50)
    if core_success and ml_success:
        print("[OK] All critical dependencies are available!")
        print("\n[ROCKET] Emergency AI is ready to run!")
        print("\nQuick start commands:")
        print("  python main.py web      # Start web interface")
        print("  python main.py gui      # Start GUI application")
        print("  python main.py validate # Run system validation")
    else:
        print("[ERROR] Some critical dependencies are missing.")
        print("\n💡 Try installing manually:")
        print("  pip install -e \".[all]\"")
        print("  # or")
        print("  pip install emergency-ai[all]")
    
    # Check if we're in the right directory
    if not Path("WORKING_FILES").exists():
        print("\n[WARNING] Warning: WORKING_FILES directory not found.")
        print("Make sure you're running this from the Emergency AI root directory.")


if __name__ == "__main__":
    main()