"""
Enhanced configuration management system for Emergency AI.
Supports YAML/TOML configuration files with CLI override capabilities.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None

from dataclasses import dataclass, field


@dataclass
class ProcessingConfig:
    """Processing-related configuration."""
    parallel_max_workers: int = 4
    enable_batch_processing: bool = False
    audio_batch_size: int = 8
    chunk_duration: float = 30.0
    overlap_duration: float = 15.0
    memory_target_mb: int = 512


@dataclass
class ModelConfig:
    """Model-related configuration."""
    model_path: str = "models/default_model.pt"
    use_onnx: bool = False
    use_onnx_audio: bool = False
    use_onnx_text: bool = False
    onnx_model_dir: str = "models/optimized"
    use_onnx_cuda: str = "auto"
    vosk_model_path: str = "models/vosk-model-medium-en-us"
    yamnet_model_path: str = "models/yamnet"
    wav2vec2_model_path: str = "models/wav2vec2"
    distilroberta_model_path: str = "models/distilroberta"


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 16000
    vad_aggressiveness: int = 2
    silence_threshold: float = 0.01
    normalize_audio: bool = True
    remove_silence: bool = True
    max_audio_length: float = 600.0


@dataclass
class FusionConfig:
    """Fusion engine configuration."""
    enable_advanced_fusion: bool = True
    distress_threshold: float = 0.5
    confidence_threshold: float = 0.3
    emotion_weight: float = 0.4
    keyword_weight: float = 0.3
    sound_event_weight: float = 0.3
    sensitivity: str = "medium"
    sensitivity_adjustments: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "low": {"distress_threshold": 0.7, "confidence_threshold": 0.5},
        "medium": {"distress_threshold": 0.5, "confidence_threshold": 0.3},
        "high": {"distress_threshold": 0.3, "confidence_threshold": 0.2}
    })


@dataclass
class KeywordConfig:
    """Keyword detection configuration."""
    enable_keyword_detection: bool = True
    emergency_keywords: list = field(default_factory=lambda: [
        "help", "emergency", "fire", "accident", "hurt", "injured", 
        "danger", "attack", "threat", "scared", "violence", "robbery", "assault"
    ])
    medical_keywords: list = field(default_factory=lambda: [
        "ambulance", "hospital", "doctor", "medical", "pain", "bleeding",
        "unconscious", "breathing", "heart", "stroke", "overdose"
    ])
    keyword_boost_factor: float = 1.5


@dataclass
class StreamingConfig:
    """Streaming and real-time processing configuration."""
    enable_live_audio: bool = True
    chunk_size_ms: int = 1000
    buffer_size_ms: int = 5000
    min_chunk_duration: float = 3.0
    max_chunk_duration: float = 30.0
    real_time_processing: bool = True
    show_partial_results: bool = True
    microphone: Dict[str, Any] = field(default_factory=lambda: {
        "device_index": None,
        "channels": 1,
        "sample_rate": 16000,
        "frames_per_buffer": 1024
    })


@dataclass
class UIConfig:
    """User interface configuration."""
    streamlit: Dict[str, Any] = field(default_factory=lambda: {
        "page_title": "Emergency AI - Audio Analysis",
        "page_icon": "🚨",
        "layout": "wide",
        "sidebar_state": "expanded",
        "update_interval_ms": 500,
        "max_display_chunks": 10,
        "show_confidence_bars": True,
        "show_waveform": True
    })
    results_display: Dict[str, bool] = field(default_factory=lambda: {
        "show_raw_scores": False,
        "show_debug_info": False,
        "highlight_high_distress": True,
        "color_code_emotions": True
    })


@dataclass
class EmergencyAIConfig:
    """Main configuration class containing all subsections."""
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    keywords: KeywordConfig = field(default_factory=KeywordConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # Additional configs
    logging: Dict[str, Any] = field(default_factory=lambda: {
        "level": "INFO",
        "file": "logs/system.log",
        "max_file_size_mb": 50,
        "backup_count": 3,
        "console_output": True,
        "performance_logging": True,
        "detailed_timing": False,
        "memory_tracking": True
    })
    
    performance: Dict[str, Any] = field(default_factory=lambda: {
        "enable_gpu": True,
        "gpu_device_id": 0,
        "memory_limit_mb": 2048,
        "cache_models": True,
        "preload_models": True,
        "use_mixed_precision": False,
        "optimize_for_latency": True,
        "batch_inference": True,
        "model_quantization": False
    })
    
    development: Dict[str, Any] = field(default_factory=lambda: {
        "debug_mode": False,
        "profile_performance": False,
        "save_intermediate_results": False,
        "enable_test_data": False,
        "mock_models": False,
        "test_audio_path": "WORKING_FILES/audio_samples",
        "generate_synthetic_data": False
    })
    
    security: Dict[str, Any] = field(default_factory=lambda: {
        "sanitize_inputs": True,
        "max_file_size_mb": 100,
        "allowed_file_types": [".wav", ".mp3", ".flac", ".m4a", ".ogg"],
        "api_key_required": False,
        "rate_limiting": False,
        "max_requests_per_minute": 60
    })


class ConfigManager:
    """Enhanced configuration manager with YAML/TOML support and CLI overrides."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.config = EmergencyAIConfig()
        self._cli_overrides = {}
        
        if self.config_path and os.path.exists(self.config_path):
            self.load_config()
    
    def _find_config_file(self) -> Optional[str]:
        """Find configuration file in common locations."""
        search_paths = [
            "config.yaml",
            "config.yml",
            "config.toml",
            "WORKING_FILES/config.yaml",
            "WORKING_FILES/config.yml",
            "WORKING_FILES/config.toml",
            os.path.expanduser("~/.emergency_ai/config.yaml"),
            "/etc/emergency_ai/config.yaml"
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
        return None
    
    def load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML or TOML file."""
        path = config_path or self.config_path
        if not path or not os.path.exists(path):
            print(f"⚠️ Config file not found: {path}, using defaults")
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.endswith('.toml'):
                    if tomllib is None:
                        raise ImportError("TOML support requires 'tomli' package")
                    data = tomllib.load(f)
                else:
                    data = yaml.safe_load(f)
            
            self._update_config_from_dict(data)
            print(f"✅ Configuration loaded from {path}")
            
        except Exception as e:
            print(f"❌ Error loading config from {path}: {e}")
            print("Using default configuration")
    
    def _update_config_from_dict(self, data: Dict[str, Any]) -> None:
        """Update configuration from dictionary data."""
        for section, values in data.items():
            if hasattr(self.config, section):
                section_obj = getattr(self.config, section)
                if hasattr(section_obj, '__dict__'):
                    # Update dataclass fields
                    for key, value in values.items():
                        if hasattr(section_obj, key):
                            setattr(section_obj, key, value)
                else:
                    # Update dictionary fields
                    if isinstance(section_obj, dict):
                        section_obj.update(values)
    
    def setup_cli_parser(self) -> argparse.ArgumentParser:
        """Set up command-line argument parser with config overrides."""
        parser = argparse.ArgumentParser(
            description="Emergency AI - Audio Analysis System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python main.py --config config.yaml
  python main.py --workers 8 --batch-size 16
  python main.py --sensitivity high --debug
  streamlit run app_streamlit.py -- --live-audio --sensitivity low
            """
        )
        
        # Configuration file
        parser.add_argument('--config', '-c', type=str, 
                          help='Path to configuration file (YAML/TOML)')
        
        # Processing overrides
        parser.add_argument('--workers', type=int, 
                          help='Number of parallel workers')
        parser.add_argument('--batch-size', type=int,
                          help='Audio batch size for processing')
        parser.add_argument('--enable-batch', action='store_true',
                          help='Enable batch processing')
        parser.add_argument('--memory-target', type=int,
                          help='Target memory usage in MB')
        
        # Model overrides
        parser.add_argument('--use-onnx', action='store_true',
                          help='Use ONNX optimized models')
        parser.add_argument('--model-path', type=str,
                          help='Path to model directory')
        parser.add_argument('--gpu', '--enable-gpu', action='store_true',
                          help='Enable GPU acceleration')
        parser.add_argument('--cpu-only', action='store_true',
                          help='Force CPU-only processing')
        
        # Fusion engine overrides
        parser.add_argument('--sensitivity', choices=['low', 'medium', 'high'],
                          help='Detection sensitivity level')
        parser.add_argument('--distress-threshold', type=float,
                          help='Distress detection threshold (0-1)')
        parser.add_argument('--confidence-threshold', type=float,
                          help='Confidence threshold (0-1)')
        
        # Audio overrides
        parser.add_argument('--sample-rate', type=int,
                          help='Audio sample rate')
        parser.add_argument('--chunk-duration', type=float,
                          help='Audio chunk duration in seconds')
        
        # Streaming overrides
        parser.add_argument('--live-audio', action='store_true',
                          help='Enable live microphone input')
        parser.add_argument('--no-live-audio', action='store_true',
                          help='Disable live microphone input')
        parser.add_argument('--real-time', action='store_true',
                          help='Enable real-time processing')
        
        # Development overrides
        parser.add_argument('--debug', action='store_true',
                          help='Enable debug mode')
        parser.add_argument('--profile', action='store_true',
                          help='Enable performance profiling')
        parser.add_argument('--verbose', '-v', action='store_true',
                          help='Verbose logging')
        
        return parser
    
    def apply_cli_overrides(self, args: argparse.Namespace) -> None:
        """Apply command-line overrides to configuration."""
        # Load custom config file if specified
        if args.config:
            self.load_config(args.config)
        
        # Processing overrides
        if args.workers:
            self.config.processing.parallel_max_workers = args.workers
        if args.batch_size:
            self.config.processing.audio_batch_size = args.batch_size
        if args.enable_batch:
            self.config.processing.enable_batch_processing = True
        if args.memory_target:
            self.config.processing.memory_target_mb = args.memory_target
        
        # Model overrides
        if args.use_onnx:
            self.config.models.use_onnx = True
            self.config.models.use_onnx_audio = True
            self.config.models.use_onnx_text = True
        if args.model_path:
            self.config.models.model_path = args.model_path
        if args.gpu:
            self.config.performance['enable_gpu'] = True
        if args.cpu_only:
            self.config.performance['enable_gpu'] = False
        
        # Fusion engine overrides
        if args.sensitivity:
            self.config.fusion.sensitivity = args.sensitivity
            # Apply sensitivity adjustments
            if args.sensitivity in self.config.fusion.sensitivity_adjustments:
                adjustments = self.config.fusion.sensitivity_adjustments[args.sensitivity]
                self.config.fusion.distress_threshold = adjustments['distress_threshold']
                self.config.fusion.confidence_threshold = adjustments['confidence_threshold']
        
        if args.distress_threshold is not None:
            self.config.fusion.distress_threshold = args.distress_threshold
        if args.confidence_threshold is not None:
            self.config.fusion.confidence_threshold = args.confidence_threshold
        
        # Audio overrides
        if args.sample_rate:
            self.config.audio.sample_rate = args.sample_rate
        if args.chunk_duration:
            self.config.processing.chunk_duration = args.chunk_duration
        
        # Streaming overrides
        if args.live_audio:
            self.config.streaming.enable_live_audio = True
        if args.no_live_audio:
            self.config.streaming.enable_live_audio = False
        if args.real_time:
            self.config.streaming.real_time_processing = True
        
        # Development overrides
        if args.debug:
            self.config.development['debug_mode'] = True
            self.config.logging['level'] = 'DEBUG'
        if args.profile:
            self.config.development['profile_performance'] = True
        if args.verbose:
            self.config.logging['level'] = 'DEBUG'
            self.config.logging['detailed_timing'] = True
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key."""
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                elif isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        except Exception:
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by dot-notation key."""
        keys = key.split('.')
        obj = self.config
        
        for k in keys[:-1]:
            if hasattr(obj, k):
                obj = getattr(obj, k)
            elif isinstance(obj, dict):
                if k not in obj:
                    obj[k] = {}
                obj = obj[k]
            else:
                return  # Can't set nested value
        
        final_key = keys[-1]
        if hasattr(obj, final_key):
            setattr(obj, final_key, value)
        elif isinstance(obj, dict):
            obj[final_key] = value
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        path = output_path or self.config_path or "config.yaml"
        
        # Convert config to dictionary
        config_dict = self._config_to_dict()
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            print(f"✅ Configuration saved to {path}")
        except Exception as e:
            print(f"❌ Error saving config to {path}: {e}")
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        result = {}
        
        for attr_name in dir(self.config):
            if attr_name.startswith('_'):
                continue
            
            attr_value = getattr(self.config, attr_name)
            
            if hasattr(attr_value, '__dict__'):
                # Convert dataclass to dict
                result[attr_name] = {
                    k: v for k, v in attr_value.__dict__.items()
                    if not k.startswith('_')
                }
            elif isinstance(attr_value, dict):
                result[attr_name] = attr_value.copy()
        
        return result
    
    def print_config(self, section: Optional[str] = None) -> None:
        """Print current configuration."""
        config_dict = self._config_to_dict()
        
        if section:
            if section in config_dict:
                print(f"📋 Configuration - {section}:")
                for key, value in config_dict[section].items():
                    print(f"  {key}: {value}")
            else:
                print(f"❌ Section '{section}' not found")
        else:
            print("📋 Current Configuration:")
            for section_name, section_data in config_dict.items():
                print(f"\n[{section_name}]")
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  {section_data}")
    
    # Backward compatibility methods
    def get_parallel_max_workers(self) -> int:
        return self.config.processing.parallel_max_workers
    
    def get_enable_batch_processing(self) -> bool:
        return self.config.processing.enable_batch_processing
    
    def get_audio_batch_size(self) -> int:
        return self.config.processing.audio_batch_size
    
    def get_model_path(self) -> str:
        return self.config.models.model_path
    
    def get_use_onnx(self) -> bool:
        return self.config.models.use_onnx


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get or create global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config(config_path: Optional[str] = None) -> ConfigManager:
    """Load configuration and return manager instance."""
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config() -> EmergencyAIConfig:
    """Get current configuration object."""
    return get_config_manager().config


# Convenience functions for backward compatibility
def get_parallel_max_workers() -> int:
    return get_config_manager().get_parallel_max_workers()


def get_enable_batch_processing() -> bool:
    return get_config_manager().get_enable_batch_processing()


def get_audio_batch_size() -> int:
    return get_config_manager().get_audio_batch_size()


def get_model_path() -> str:
    return get_config_manager().get_model_path()


def get_use_onnx() -> bool:
    return get_config_manager().get_use_onnx()