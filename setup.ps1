# Emergency AI - Installation and Setup Script
# Windows PowerShell version for easy setup

param(
    [string]$Command = "help",
    [string]$Environment = "dev",
    [string]$Tag = "latest",
    [switch]$NoCache,
    [string]$Profile,
    [switch]$Verbose
)

# Colors for output
$Global:Colors = @{
    Red    = "Red"
    Green  = "Green" 
    Yellow = "Yellow"
    Blue   = "Blue"
    White  = "White"
}

# Logging functions
function Write-Log {
    param([string]$Message, [string]$Color = "Blue")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $Message" -ForegroundColor $Color
}

function Write-Error-Log {
    param([string]$Message)
    Write-Log "[ERROR] $Message" -Color Red
}

function Write-Success {
    param([string]$Message)
    Write-Log "[SUCCESS] $Message" -Color Green
}

function Write-Warning-Log {
    param([string]$Message)
    Write-Log "[WARNING] $Message" -Color Yellow
}

# Help function
function Show-Help {
    Write-Host @"
Emergency AI Setup Script for Windows

Usage: .\setup.ps1 [COMMAND] [OPTIONS]

Commands:
    install             Install Emergency AI and dependencies
    build               Build Docker images (requires Docker)
    run                 Run the application locally
    dev                 Start development environment  
    test                Run test suite
    benchmark          Run performance benchmarks
    clean              Clean up temporary files
    update             Update dependencies
    doctor             Check system requirements
    help               Show this help message

Options:
    -Environment ENV   Environment (dev|prod)
    -Tag TAG          Docker image tag
    -NoCache          Build without cache
    -Profile PROFILE  Docker compose profile
    -Verbose          Verbose output

Examples:
    .\setup.ps1 install
    .\setup.ps1 run
    .\setup.ps1 dev -Profile jupyter
    .\setup.ps1 test -Verbose
    .\setup.ps1 build -Tag v1.0.0

"@
}

# Check if running as administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Check system requirements
function Test-SystemRequirements {
    Write-Log "Checking system requirements..."
    
    $requirements = @{
        "Python 3.8+" = { python --version 2>$null | Select-String "Python 3\.[8-9]|Python 3\.1[0-9]" }
        "pip"         = { pip --version 2>$null }
        "git"         = { git --version 2>$null }
    }
    
    $missing = @()
    
    foreach ($req in $requirements.Keys) {
        if (-not (& $requirements[$req])) {
            $missing += $req
        }
    }
    
    if ($missing.Count -gt 0) {
        Write-Error-Log "Missing requirements: $($missing -join ', ')"
        Write-Host "Please install the missing requirements:" -ForegroundColor Yellow
        Write-Host "  - Python 3.8+: https://www.python.org/downloads/" -ForegroundColor Yellow
        Write-Host "  - Git: https://git-scm.com/download/win" -ForegroundColor Yellow
        return $false
    }
    
    Write-Success "All system requirements met"
    return $true
}

# Install Emergency AI
function Install-EmergencyAI {
    Write-Log "Installing Emergency AI..."
    
    if (-not (Test-SystemRequirements)) {
        return
    }
    
    # Create virtual environment
    Write-Log "Creating virtual environment..."
    if (Test-Path "venv") {
        Write-Warning-Log "Virtual environment already exists"
    }
    else {
        python -m venv venv
    }
    
    # Activate virtual environment
    Write-Log "Activating virtual environment..."
    if (Test-Path "venv\Scripts\Activate.ps1") {
        & "venv\Scripts\Activate.ps1"
    }
    else {
        Write-Error-Log "Failed to activate virtual environment"
        return
    }
    
    # Upgrade pip
    Write-Log "Upgrading pip..."
    python -m pip install --upgrade pip setuptools wheel
    
    # Install dependencies
    Write-Log "Installing dependencies..."
    if (Test-Path "pyproject.toml") {
        pip install -e .[dev, testing] 
    }
    else {
        Write-Error-Log "pyproject.toml not found"
        return
    }
    
        # Download models
        Write-Log "Downloading AI models..."
        New-Item -ItemType Directory -Force -Path "WORKING_FILES\models" | Out-Null
    
    # Create model download script (downloads whisper-medium)
    $downloadScript = @'
import os
from pathlib import Path

MODELDIR = Path('WORKING_FILES') / 'models' / 'whisper-medium'
MODELDIR.mkdir(parents=True, exist_ok=True)

def already_downloaded(p: Path) -> bool:
    # A minimal heuristic: model dir contains at least one model file
    for child in p.iterdir():
        if child.is_file():
            return True
    return False

if already_downloaded(MODELDIR):
    print('Whisper-medium already present at', MODELDIR)
else:
    print('Attempting to download whisper-medium into', MODELDIR)
    try:
        # Prefer huggingface_hub snapshot_download (will place files into the target dir)
        from huggingface_hub import snapshot_download

        print('Using huggingface_hub.snapshot_download to fetch model...')
        snapshot_download(repo_id='openai/whisper-medium', cache_dir=str(MODELDIR), repo_type='model')
        print('Downloaded whisper-medium to', MODELDIR)
    except Exception as e:
        print('huggingface_hub unavailable or download failed:', e)
        print('Falling back to forcing faster-whisper to download into HF cache...')
        try:
            from faster_whisper import WhisperModel

            # Instantiating will download the model into the Hugging Face cache
            print('Instantiating faster-whisper WhisperModel("medium") to trigger download (may use ~2GB)...')
            _ = WhisperModel('medium', device='cpu')
            print('faster-whisper has downloaded the model into the Hugging Face cache.')
            print('You can move or symlink the cache contents into', MODELDIR, 'if you want a local copy.')
        except Exception as e2:
            print('Failed to trigger faster-whisper download:', e2)
            print('Please install huggingface_hub or faster-whisper and re-run this script to fetch the model.')

'@

$downloadScript | Out-File -FilePath "download_models.py" -Encoding UTF8
        python download_models.py
        Remove-Item "download_models.py"

        Write-Host "Note: If you plan to use the Whisper model, install ffmpeg on your system (choco install ffmpeg on Windows)." -ForegroundColor Yellow
    
    Write-Success "Emergency AI installation completed!"
    Write-Host "To run the application: .\setup.ps1 run" -ForegroundColor Green
}

# Run the application
function Start-Application {
    Write-Log "Starting Emergency AI application..."
    
    # Activate virtual environment
    if (Test-Path "venv\Scripts\Activate.ps1") {
        & "venv\Scripts\Activate.ps1"
    }
    
    # Check if models exist
    if (-not (Test-Path "WORKING_FILES\models")) {
        Write-Warning-Log "Models not found. Please run: .\setup.ps1 install"
        return
    }
    
    # Start Streamlit app
    Write-Log "Starting Streamlit interface on http://localhost:8501"
    streamlit run "WORKING_FILES\app_streamlit.py" --server.port 8501 --server.address localhost
}

# Start development environment
function Start-Development {
    Write-Log "Starting development environment..."
    
    # Check for Docker
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        Write-Log "Using Docker for development environment..."
        if ($Profile) {
            docker-compose -f docker-compose.dev.yml --profile $Profile up -d
        }
        else {
            docker-compose -f docker-compose.dev.yml up -d
        }
        
        Write-Success "Development environment started"
        Write-Host "Streamlit UI: http://localhost:8501" -ForegroundColor Green
        if ($Profile -eq "jupyter") {
            Write-Host "Jupyter Lab: http://localhost:8888 (token: emergency123)" -ForegroundColor Green
        }
    }
    else {
        Write-Log "Docker not found, starting local development..."
        Start-Application
    }
}

# Run tests
function Invoke-Tests {
    Write-Log "Running test suite..."
    
    # Activate virtual environment
    if (Test-Path "venv\Scripts\Activate.ps1") {
        & "venv\Scripts\Activate.ps1"
    }
    
    # Run pytest
    $testArgs = @("WORKING_FILES\tests\")
    if ($Verbose) {
        $testArgs += "-v"
    }
    else {
        $testArgs += "-q"
    }
    
    python -m pytest @testArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "All tests passed"
    }
    else {
        Write-Error-Log "Some tests failed"
    }
}

# Run benchmarks
function Invoke-Benchmarks {
    Write-Log "Running performance benchmarks..."
    
    # Activate virtual environment
    if (Test-Path "venv\Scripts\Activate.ps1") {
        & "venv\Scripts\Activate.ps1"
    }
    
    python "WORKING_FILES\benchmarks\performance_profiler.py" --benchmark
    Write-Success "Benchmarks completed"
}

# Build Docker images
function Build-DockerImages {
    Write-Log "Building Docker images..."
    
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error-Log "Docker not found. Please install Docker Desktop for Windows"
        return
    }
    
    $buildArgs = @()
    if ($NoCache) {
        $buildArgs += "--no-cache"
    }
    
    # Build production image
    Write-Log "Building production image..."
    docker build @buildArgs -t "emergency-ai:$Tag" -f Dockerfile .
    
    # Build development image
    Write-Log "Building development image..."
    docker build @buildArgs -t "emergency-ai:dev" -f Dockerfile.dev .
    
    Write-Success "Docker images built successfully"
}

# Clean up temporary files
function Clean-Temporary {
    Write-Log "Cleaning up temporary files..."
    
    $cleanupPaths = @(
        "WORKING_FILES\__pycache__",
        "WORKING_FILES\tmp_chunks",
        "logs\*.log",
        "*.pyc",
        ".pytest_cache"
    )
    
    foreach ($path in $cleanupPaths) {
        if (Test-Path $path) {
            Remove-Item $path -Recurse -Force
            Write-Log "Removed: $path"
        }
    }
    
    Write-Success "Cleanup completed"
}

# Update dependencies
function Update-Dependencies {
    Write-Log "Updating dependencies..."
    
    # Activate virtual environment
    if (Test-Path "venv\Scripts\Activate.ps1") {
        & "venv\Scripts\Activate.ps1"
    }
    
    # Update pip
    python -m pip install --upgrade pip
    
    # Update all packages
    pip install --upgrade -e .[dev, testing]
    
    Write-Success "Dependencies updated"
}

# System doctor - comprehensive health check
function Invoke-SystemDoctor {
    Write-Log "Running system diagnostics..."
    
    # Check Python installation
    Write-Host "`nPython Environment:" -ForegroundColor Blue
    python --version
    pip --version
    
    # Check available memory
    Write-Host "`nSystem Resources:" -ForegroundColor Blue
    $memory = Get-WmiObject -Class Win32_ComputerSystem
    $totalMemory = [math]::Round($memory.TotalPhysicalMemory / 1GB, 2)
    Write-Host "Total RAM: $totalMemory GB"
    
    # Check disk space
    $disk = Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='C:'"
    $freeSpace = [math]::Round($disk.FreeSpace / 1GB, 2)
    Write-Host "Free Disk Space: $freeSpace GB"
    
    # Check if models exist
    Write-Host "`nEmergency AI Status:" -ForegroundColor Blue
    if (Test-Path "WORKING_FILES\models") {
        Write-Host "Models directory: ✓ Found" -ForegroundColor Green
        $modelCount = (Get-ChildItem "WORKING_FILES\models" -Recurse).Count
        Write-Host "Model files: $modelCount files"
    }
    else {
        Write-Host "Models directory: ✗ Not found" -ForegroundColor Red
    }
    
    if (Test-Path "venv") {
        Write-Host "Virtual environment: ✓ Found" -ForegroundColor Green
    }
    else {
        Write-Host "Virtual environment: ✗ Not found" -ForegroundColor Red
    }
    
    # Check Docker (optional)
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        Write-Host "Docker: ✓ Available" -ForegroundColor Green
        docker --version
    }
    else {
        Write-Host "Docker: ✗ Not available" -ForegroundColor Yellow
    }
    
    Write-Success "System diagnostics completed"
}

# Main script logic
switch ($Command.ToLower()) {
    "install" {
        Install-EmergencyAI
    }
    "run" {
        Start-Application
    }
    "dev" {
        Start-Development
    }
    "test" {
        Invoke-Tests
    }
    "benchmark" {
        Invoke-Benchmarks
    }
    "build" {
        Build-DockerImages
    }
    "clean" {
        Clean-Temporary
    }
    "update" {
        Update-Dependencies
    }
    "doctor" {
        Invoke-SystemDoctor
    }
    "help" {
        Show-Help
    }
    default {
        Write-Error-Log "Unknown command: $Command"
        Show-Help
    }
}