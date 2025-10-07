# Emergency AI Installation Verification Script
# Comprehensive system validation for Emergency AI deployment

import sys
import os
import subprocess
import importlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import time
import tempfile
import shutil

def colorize(text: str, color: str) -> str:
    """Add color to terminal text."""
    colors = {
        'green': '\033[92m',
        'red': '\033[91m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'bold': '\033[1m',
        'end': '\033[0m'
    }
    return f"{colors.get(color, '')}{text}{colors.get('end', '')}"

class SystemValidator:
    """Comprehensive system validation for Emergency AI."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {
            'python_env': {},
            'dependencies': {},
            'models': {},
            'configuration': {},
            'functionality': {},
            'performance': {},
            'overall': {'status': 'unknown', 'score': 0}
        }
        self.issues = []
        self.warnings = []
    
    def print_status(self, message: str, status: str = 'info'):
        """Print formatted status message."""
        if status == 'success':
            print(f"✓ {colorize(message, 'green')}")
        elif status == 'error':
            print(f"✗ {colorize(message, 'red')}")
        elif status == 'warning':
            print(f"⚠ {colorize(message, 'yellow')}")
        elif status == 'info':
            print(f"ℹ {colorize(message, 'blue')}")
        else:
            print(f"  {message}")
    
    def check_python_environment(self) -> bool:
        """Check Python environment and version."""
        self.print_status("Checking Python environment...", 'info')
        
        # Python version
        python_version = sys.version_info
        version_str = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
        
        if python_version >= (3, 8):
            self.print_status(f"Python version: {version_str}", 'success')
            self.results['python_env']['version'] = version_str
            self.results['python_env']['version_ok'] = True
        else:
            self.print_status(f"Python version {version_str} too old (require >= 3.8)", 'error')
            self.results['python_env']['version'] = version_str
            self.results['python_env']['version_ok'] = False
            self.issues.append("Python version too old")
            return False
        
        # Python executable
        executable = sys.executable
        self.print_status(f"Python executable: {executable}")
        self.results['python_env']['executable'] = executable
        
        # Virtual environment check
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        if in_venv:
            self.print_status("Virtual environment detected", 'success')
        else:
            self.print_status("Not in virtual environment", 'warning')
            self.warnings.append("Consider using virtual environment")
        
        self.results['python_env']['virtual_env'] = in_venv
        
        return True
    
    def check_dependencies(self) -> bool:
        """Check required dependencies."""
        self.print_status("Checking dependencies...", 'info')
        
        # Core dependencies
        core_deps = {
            'numpy': '1.21.0',
            'librosa': '0.9.0',
            'soundfile': '0.10.0',
            'scipy': '1.7.0',
            'scikit-learn': '1.0.0',
            'tensorflow': '2.8.0',
            'torch': '1.10.0',
            'transformers': '4.15.0',
            'streamlit': '1.15.0',
            'plotly': '5.0.0',
            'pandas': '1.3.0',
            'psutil': '5.8.0',
            'pyyaml': '6.0',
            'click': '8.0.0',
            'loguru': '0.6.0',
            'pytest': '7.0.0'
        }
        
        missing_deps = []
        outdated_deps = []
        
        for package, min_version in core_deps.items():
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                
                self.print_status(f"{package}: {version}", 'success')
                self.results['dependencies'][package] = {
                    'installed': True,
                    'version': version,
                    'required': min_version
                }
                
                # Version check (simplified)
                if version != 'unknown' and self._compare_versions(version, min_version) < 0:
                    outdated_deps.append(f"{package} {version} < {min_version}")
                    self.warnings.append(f"{package} version may be outdated")
                
            except ImportError:
                self.print_status(f"{package}: not installed", 'error')
                self.results['dependencies'][package] = {
                    'installed': False,
                    'version': None,
                    'required': min_version
                }
                missing_deps.append(package)
        
        if missing_deps:
            self.issues.append(f"Missing dependencies: {', '.join(missing_deps)}")
        
        if outdated_deps:
            self.warnings.append(f"Potentially outdated: {', '.join(outdated_deps)}")
        
        return len(missing_deps) == 0
    
    def check_models(self) -> bool:
        """Check AI models availability."""
        self.print_status("Checking AI models...", 'info')
        
        models_dir = Path("WORKING_FILES/models")
        if not models_dir.exists():
            self.print_status("Models directory not found", 'error')
            self.issues.append("Models directory missing")
            self.results['models']['directory_exists'] = False
            return False
        
        self.results['models']['directory_exists'] = True
        
        # Expected models
        expected_models = {
            'yamnet': ['yamnet.h5', 'yamnet/saved_model.pb'],
            'wav2vec2': ['wav2vec2/config.json', 'wav2vec2/model.safetensors'],
            'distilroberta': ['distilroberta/config.json', 'distilroberta/model.safetensors'],
            'vosk': ['vosk-model-small-en-us-0.15', 'vosk-model-medium-en-us', 'vosk-model-large-en-us'],
            'whisper': ['whisper-medium', 'whisper-medium.en']
        }
        
        model_status = {}
        
        for model_name, paths in expected_models.items():
            found = False
            for path in paths:
                full_path = models_dir / path
                if full_path.exists():
                    found = True
                    self.print_status(f"{model_name}: found", 'success')
                    break
            
            if not found:
                self.print_status(f"{model_name}: missing", 'error')
                self.issues.append(f"Missing model: {model_name}")
            
            model_status[model_name] = found
        
        self.results['models']['models'] = model_status
        
        # Check model sizes
        total_size = sum(f.stat().st_size for f in models_dir.rglob('*') if f.is_file())
        total_size_gb = total_size / (1024**3)
        
        self.print_status(f"Total model size: {total_size_gb:.2f} GB")
        self.results['models']['total_size_gb'] = total_size_gb
        
        if total_size_gb < 1.0:
            self.warnings.append("Models directory seems small - models may be incomplete")
        
        return all(model_status.values())
    
    def check_configuration(self) -> bool:
        """Check configuration files."""
        self.print_status("Checking configuration...", 'info')
        
        config_file = Path("WORKING_FILES/config.yaml")
        if config_file.exists():
            self.print_status("Configuration file found", 'success')
            self.results['configuration']['config_exists'] = True
            
            try:
                import yaml
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                self.print_status("Configuration file is valid YAML", 'success')
                self.results['configuration']['config_valid'] = True
                self.results['configuration']['config'] = config
            except Exception as e:
                self.print_status(f"Configuration file invalid: {e}", 'error')
                self.issues.append("Invalid configuration file")
                self.results['configuration']['config_valid'] = False
                return False
        else:
            self.print_status("Configuration file not found", 'warning')
            self.warnings.append("No configuration file found")
            self.results['configuration']['config_exists'] = False
        
        # Check pyproject.toml
        pyproject_file = Path("pyproject.toml")
        if pyproject_file.exists():
            self.print_status("pyproject.toml found", 'success')
            self.results['configuration']['pyproject_exists'] = True
        else:
            self.print_status("pyproject.toml not found", 'warning')
            self.results['configuration']['pyproject_exists'] = False
        
        return True
    
    def check_functionality(self) -> bool:
        """Check basic functionality."""
        self.print_status("Checking functionality...", 'info')
        
        try:
            # Test imports
            sys.path.append(str(Path("WORKING_FILES")))
            
            from modules.config_manager import get_config_manager
            config_manager = get_config_manager()
            self.print_status("Configuration manager: OK", 'success')
            
            from modules.enhanced_logger import get_logger
            logger = get_logger()
            self.print_status("Logger: OK", 'success')
            
            from modules.model_manager import ModelManager
            self.print_status("Model manager: OK", 'success')
            
            # Test basic audio processing
            if Path("WORKING_FILES/audio_samples").exists():
                sample_files = list(Path("WORKING_FILES/audio_samples").glob("*.wav"))
                if sample_files:
                    try:
                        from analysis_pipeline import process_audio_file
                        result = process_audio_file(str(sample_files[0]))
                        if result and isinstance(result, dict):
                            self.print_status("Audio processing: OK", 'success')
                            self.results['functionality']['audio_processing'] = True
                        else:
                            self.print_status("Audio processing returned invalid result", 'error')
                            self.issues.append("Audio processing test failed")
                            self.results['functionality']['audio_processing'] = False
                    except Exception as e:
                        self.print_status(f"Audio processing failed: {e}", 'error')
                        self.issues.append("Audio processing test failed")
                        self.results['functionality']['audio_processing'] = False
                else:
                    self.print_status("No audio samples for testing", 'warning')
                    self.warnings.append("No audio samples available for testing")
            else:
                self.print_status("Audio samples directory not found", 'warning')
                self.warnings.append("No audio samples directory")
            
            self.results['functionality']['imports'] = True
            return True
            
        except ImportError as e:
            self.print_status(f"Import failed: {e}", 'error')
            self.issues.append(f"Module import failed: {e}")
            self.results['functionality']['imports'] = False
            return False
        except Exception as e:
            self.print_status(f"Functionality check failed: {e}", 'error')
            self.issues.append(f"Functionality test failed: {e}")
            return False
    
    def check_performance(self) -> bool:
        """Check performance indicators."""
        self.print_status("Checking performance...", 'info')
        
        try:
            import psutil
            
            # Memory
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)
            
            self.print_status(f"Total memory: {memory_gb:.1f} GB", 'success')
            self.print_status(f"Available memory: {available_gb:.1f} GB")
            
            if available_gb < 4.0:
                self.warnings.append("Low available memory (< 4GB)")
            
            # CPU
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            self.print_status(f"CPU cores: {cpu_count}", 'success')
            self.print_status(f"CPU usage: {cpu_percent:.1f}%")
            
            # Disk space
            disk_usage = psutil.disk_usage('.')
            free_gb = disk_usage.free / (1024**3)
            
            self.print_status(f"Free disk space: {free_gb:.1f} GB")
            
            if free_gb < 10.0:
                self.warnings.append("Low disk space (< 10GB)")
            
            self.results['performance'] = {
                'memory_total_gb': memory_gb,
                'memory_available_gb': available_gb,
                'cpu_cores': cpu_count,
                'cpu_usage_percent': cpu_percent,
                'disk_free_gb': free_gb
            }
            
            return True
            
        except ImportError:
            self.print_status("psutil not available for performance check", 'warning')
            return True
        except Exception as e:
            self.print_status(f"Performance check failed: {e}", 'warning')
            return True
    
    def run_validation(self) -> Dict:
        """Run complete validation."""
        print(colorize("Emergency AI - System Validation", 'bold'))
        print("=" * 50)
        
        start_time = time.time()
        
        # Run all checks
        checks = [
            ('Python Environment', self.check_python_environment),
            ('Dependencies', self.check_dependencies),
            ('AI Models', self.check_models),
            ('Configuration', self.check_configuration),
            ('Functionality', self.check_functionality),
            ('Performance', self.check_performance)
        ]
        
        passed_checks = 0
        total_checks = len(checks)
        
        for check_name, check_func in checks:
            print(f"\n{colorize(check_name, 'bold')}")
            print("-" * 20)
            
            try:
                result = check_func()
                if result:
                    passed_checks += 1
            except Exception as e:
                self.print_status(f"Check failed with exception: {e}", 'error')
                self.issues.append(f"{check_name} check failed: {e}")
        
        # Calculate overall score
        score = (passed_checks / total_checks) * 100
        self.results['overall']['score'] = score
        
        # Determine overall status
        if score >= 90 and len(self.issues) == 0:
            status = 'excellent'
            status_color = 'green'
        elif score >= 75 and len(self.issues) <= 2:
            status = 'good'
            status_color = 'green'
        elif score >= 50:
            status = 'acceptable'
            status_color = 'yellow'
        else:
            status = 'poor'
            status_color = 'red'
        
        self.results['overall']['status'] = status
        
        # Print summary
        elapsed_time = time.time() - start_time
        
        print(f"\n{colorize('VALIDATION SUMMARY', 'bold')}")
        print("=" * 50)
        print(f"Overall Status: {colorize(status.upper(), status_color)}")
        print(f"Score: {colorize(f'{score:.1f}%', status_color)}")
        print(f"Checks Passed: {passed_checks}/{total_checks}")
        print(f"Validation Time: {elapsed_time:.2f}s")
        
        if self.issues:
            print(f"\n{colorize('ISSUES FOUND:', 'red')}")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
        
        if self.warnings:
            print(f"\n{colorize('WARNINGS:', 'yellow')}")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        # Recommendations
        print(f"\n{colorize('RECOMMENDATIONS:', 'blue')}")
        if status == 'excellent':
            print("  ✓ System is ready for production deployment!")
        elif status == 'good':
            print("  • System is functional but consider addressing warnings")
        elif status == 'acceptable':
            print("  • System may work but issues should be resolved")
        else:
            print("  • Critical issues must be resolved before deployment")
        
        return self.results
    
    def _compare_versions(self, version1: str, version2: str) -> int:
        """Compare version strings. Returns -1, 0, or 1."""
        try:
            v1_parts = [int(x) for x in version1.split('.')]
            v2_parts = [int(x) for x in version2.split('.')]
            
            # Pad shorter version with zeros
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            if v1_parts < v2_parts:
                return -1
            elif v1_parts > v2_parts:
                return 1
            else:
                return 0
        except:
            return 0  # Can't compare, assume OK
    
    def save_report(self, filename: str):
        """Save validation report to file."""
        report = {
            'timestamp': time.time(),
            'results': self.results,
            'issues': self.issues,
            'warnings': self.warnings
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.print_status(f"Validation report saved to: {filename}", 'success')


def main():
    """Main validation entry point."""
    parser = argparse.ArgumentParser(description="Emergency AI System Validation")
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--report', '-r', type=str, help='Save report to file')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    
    args = parser.parse_args()
    
    validator = SystemValidator(verbose=args.verbose)
    results = validator.run_validation()
    
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    
    if args.report:
        validator.save_report(args.report)
    
    # Exit with appropriate code
    if results['overall']['status'] in ['excellent', 'good']:
        sys.exit(0)
    elif results['overall']['status'] == 'acceptable':
        sys.exit(1)
    else:
        sys.exit(2)


# Alias for backward compatibility
EmergencyAIValidator = SystemValidator


if __name__ == '__main__':
    main()