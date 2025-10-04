# Emergency AI - Troubleshooting Import Issues

## Quick Fix for Missing Dependencies

If you encounter import errors, follow these steps:

### 1. Automatic Installation (Recommended)
```bash
# Run the dependency installer
python install_dependencies.py

# Or install from requirements.txt
pip install -r requirements.txt

# Or install from pyproject.toml with all features
pip install -e ".[all]"
```

### 2. Manual Installation for Specific Issues

#### Plotly (for visualization)
```bash
pip install plotly>=5.0.0
```

#### Loguru (for structured logging)
```bash
pip install loguru>=0.7.0
```

#### Memory Profiler (for performance analysis)
```bash
pip install memory-profiler>=0.60.0
```

#### Streamlit WebRTC (for real-time audio streaming)
```bash
pip install streamlit-webrtc>=0.44.0
```

#### PyAudio (for audio capture)
```bash
# Windows
pip install pyaudio

# macOS
brew install portaudio
pip install pyaudio

# Ubuntu/Debian
sudo apt-get install python3-pyaudio
pip install pyaudio

# Alternative: Use conda
conda install pyaudio
```

### 3. System-Specific Audio Issues

#### Windows
```powershell
# Install Visual C++ Build Tools if needed
# Then install PyAudio
pip install pyaudio
```

#### macOS
```bash
# Install Homebrew if not available
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install PortAudio
brew install portaudio

# Install PyAudio
pip install pyaudio
```

#### Linux (Ubuntu/Debian)
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev python3-pip
sudo apt-get install libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0
sudo apt-get install ffmpeg

# Install PyAudio
pip install pyaudio
```

### 4. Verification

After installation, verify everything works:

```bash
# Run system validation
python main.py validate

# Or use the validation script
python WORKING_FILES/validate.py

# Quick dependency check
python install_dependencies.py
```

### 5. Common Import Errors and Solutions

| Error | Solution |
|-------|----------|
| `Import "plotly.express" could not be resolved` | `pip install plotly` |
| `Import "loguru" could not be resolved` | `pip install loguru` |
| `Import "memory_profiler" could not be resolved` | `pip install memory-profiler` |
| `Import "streamlit_webrtc" could not be resolved` | `pip install streamlit-webrtc` |
| `Import "pyaudio" could not be resolved` | See PyAudio installation above |

### 6. Development Environment Setup

For development with all dependencies:

```bash
# Clone repository
git clone https://github.com/SM-Pravin/EchoResQ.git
cd EchoResQ

# Create virtual environment
python -m venv emergency-ai-env
source emergency-ai-env/bin/activate  # Linux/macOS
# emergency-ai-env\Scripts\activate   # Windows

# Install all dependencies including development tools
pip install -e ".[all]"

# Verify installation
python main.py validate
```

### 7. Docker Alternative

If you encounter persistent dependency issues, use Docker:

```bash
# Build and run with Docker
docker-compose up -d

# Access web interface at http://localhost:8501
```

### 8. Minimal Installation

For basic functionality without advanced features:

```bash
# Core dependencies only
pip install numpy pandas librosa soundfile streamlit transformers torch

# Then run
python main.py validate --minimal
```

## Support

If you continue to experience issues:

1. Check the GitHub Issues: https://github.com/SM-Pravin/EchoResQ/issues
2. Run the validation script for detailed diagnostics
3. Use Docker for a clean environment
4. Check your Python version (3.8+ required)