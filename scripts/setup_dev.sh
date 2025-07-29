#!/bin/bash
# Development environment setup script for DP-Flash-Attention

set -e

echo "🚀 Setting up DP-Flash-Attention development environment..."

# Check Python version
python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.10"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "❌ Python 3.10+ required. Found: $python_version"
    exit 1
fi
echo "✅ Python version check passed: $python_version"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Not in a virtual environment. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ Virtual environment created and activated"
else
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
fi

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install development dependencies
echo "📦 Installing development dependencies..."
pip install -e ".[dev,test,docs]"

# Set up pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Check CUDA availability
echo "🖥️  Checking CUDA availability..."
if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    echo "✅ PyTorch with CUDA support detected"
    
    # Install CUDA-specific dependencies if available
    echo "📦 Installing CUDA dependencies..."
    pip install ".[cuda]" || echo "⚠️  CUDA dependencies installation failed (optional)"
else
    echo "⚠️  CUDA not available. Some features will be limited."
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data logs outputs checkpoints

# Generate initial configuration files
echo "⚙️  Generating configuration files..."
cat > privacy_config.yaml << EOF
# Privacy configuration for DP-Flash-Attention
privacy:
  default_epsilon: 1.0
  default_delta: 1e-5
  max_grad_norm: 1.0
  
testing:
  num_shadow_models: 10
  membership_inference_threshold: 0.5
  
logging:
  level: INFO
  privacy_audit: true
EOF

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "   1. Run 'make test' to verify installation"
echo "   2. Run 'make test-gpu' to test CUDA functionality (if available)"
echo "   3. Check 'make help' for available commands"
echo "   4. Read DEVELOPMENT.md for detailed setup instructions"