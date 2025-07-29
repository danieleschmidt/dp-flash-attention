#!/bin/bash
# Post-create setup script for development container

set -e

echo "🚀 Setting up DP-Flash-Attention development environment..."

# Ensure we're in the right directory
cd /workspaces/dp-flash-attention

# Activate virtual environment
source .venv/bin/activate

# Install development dependencies
echo "📦 Installing development dependencies..."
pip install -e ".[dev,test,docs]"

# Try to install CUDA dependencies if available
echo "🖥️  Installing CUDA dependencies..."
pip install ".[cuda]" || echo "⚠️  CUDA dependencies installation failed (may not be available in container)"

# Set up pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install || echo "⚠️  Pre-commit setup failed (will be available after first git clone)"

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data logs outputs checkpoints
mkdir -p notebooks examples

# Set up Jupyter kernel
echo "📓 Setting up Jupyter kernel..."
python -m ipykernel install --user --name dp-flash-attention --display-name "DP-Flash-Attention"

# Generate privacy configuration
echo "⚙️  Generating privacy configuration..."
cat > privacy_config.yaml << EOF
# Privacy configuration for DP-Flash-Attention development
privacy:
  default_epsilon: 1.0
  default_delta: 1e-5
  max_grad_norm: 1.0
  
development:
  enable_debug_logging: true
  validate_privacy_bounds: true
  
testing:
  num_shadow_models: 5  # Reduced for dev container
  membership_inference_threshold: 0.5
  
logging:
  level: DEBUG
  privacy_audit: true
EOF

# Set up environment variables
echo "🌍 Setting up environment variables..."
cat > .env << EOF
# Development environment variables
CUDA_VISIBLE_DEVICES=0
PYTHONPATH=/workspaces/dp-flash-attention/src
DP_FLASH_ATTENTION_DEV=true
TOKENIZERS_PARALLELISM=false
EOF

# Create example notebook
echo "📓 Creating example notebook..."
cat > notebooks/getting_started.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# DP-Flash-Attention Getting Started\n",
    "\n",
    "This notebook demonstrates the basic usage of DP-Flash-Attention."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import sys\n",
    "\n",
    "print(f\"Python version: {sys.version}\")\n",
    "print(f\"PyTorch version: {torch.__version__}\")\n",
    "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"CUDA version: {torch.version.cuda}\")\n",
    "    print(f\"GPU count: {torch.cuda.device_count()}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test basic import\n",
    "try:\n",
    "    import dp_flash_attention\n",
    "    print(f\"DP-Flash-Attention version: {dp_flash_attention.__version__}\")\n",
    "    print(\"✅ Import successful!\")\n",
    "except ImportError as e:\n",
    "    print(f\"❌ Import failed: {e}\")\n",
    "    print(\"Run 'pip install -e .' from the repository root to install in development mode\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Basic Usage Example\n",
    "\n",
    "Here's how to use DP-Flash-Attention in your code:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Example code will go here once the library is implemented\n",
    "print(\"Ready for development! 🚀\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "DP-Flash-Attention",
   "language": "python",
   "name": "dp-flash-attention"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Verify installation
echo "🧪 Verifying installation..."
python -c "import sys; print('Python:', sys.version_info[:2])" || echo "❌ Python check failed"
python -c "import torch; print('PyTorch version:', torch.__version__)" || echo "❌ PyTorch check failed"

# Show CUDA info if available
python -c "
import torch
if torch.cuda.is_available():
    print('✅ CUDA is available')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('⚠️  CUDA not available - CPU-only mode')
" || echo "❌ CUDA check failed"

echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "   1. Run 'make test' to verify the setup"
echo "   2. Start Jupyter with 'jupyter lab --ip=0.0.0.0 --port=8888'"
echo "   3. Open notebooks/getting_started.ipynb to begin"
echo "   4. Check 'make help' for available development commands"
echo ""
echo "🔧 Development tools available:"
echo "   - Python 3.10 with virtual environment"
echo "   - PyTorch with CUDA support (if available)"
echo "   - Code formatting: black, ruff"
echo "   - Type checking: mypy"
echo "   - Testing: pytest"
echo "   - Documentation: sphinx"
echo "   - Jupyter notebooks"
echo ""
echo "Happy coding! 🚀"