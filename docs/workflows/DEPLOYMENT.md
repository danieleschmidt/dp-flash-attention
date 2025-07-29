# GitHub Actions Deployment Guide

This document provides comprehensive guidance for implementing GitHub Actions workflows in the DP-Flash-Attention repository.

## 🎯 Overview

Since Terry cannot create actual GitHub Actions workflow files, this document serves as a complete implementation guide for repository maintainers to set up CI/CD workflows.

## 📋 Required Workflows

### 1. Continuous Integration (CI) Workflow

**File**: `.github/workflows/ci.yml`

```yaml
name: Continuous Integration

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

env:
  PYTHON_VERSION: "3.10"
  CUDA_VERSION: "12.1"

jobs:
  lint:
    name: Code Quality
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Run linting
        run: |
          make lint
      - name: Check formatting
        run: |
          black --check src/ tests/
          ruff check src/ tests/

  test-cpu:
    name: CPU Tests
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install -e ".[test]"
      - name: Run CPU tests
        run: |
          pytest tests/unit/ -v --cov=dp_flash_attention

  test-gpu:
    name: GPU Tests
    runs-on: [self-hosted, gpu]
    if: github.event_name == 'push' || github.event.pull_request.draft == false
    steps:
      - uses: actions/checkout@v4
      - name: Setup CUDA environment
        run: |
          export CUDA_HOME=/usr/local/cuda
          export PATH=$PATH:$CUDA_HOME/bin
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install CUDA dependencies
        run: |
          pip install -e ".[cuda,test]"
      - name: Run GPU tests
        run: |
          pytest tests/ -m gpu -v
      - name: Run privacy tests
        run: |
          pytest tests/privacy/ -v

  build:
    name: Build Package
    runs-on: ubuntu-latest
    needs: [lint, test-cpu]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install build dependencies
        run: |
          pip install build twine
      - name: Build package
        run: |
          python -m build
      - name: Check package
        run: |
          twine check dist/*
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/
```

### 2. Security Workflow

**File**: `.github/workflows/security.yml`

```yaml
name: Security Scanning

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday

jobs:
  security-scan:
    name: Security Analysis
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      actions: read
      contents: read
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Bandit Security Scan
        run: |
          pip install bandit[toml]
          bandit -r src/ -f json -o bandit-report.json
        continue-on-error: true
      
      - name: Run Safety Check
        run: |
          pip install safety
          safety check --json --output safety-report.json
        continue-on-error: true
      
      - name: Upload Security Reports
        uses: actions/upload-artifact@v4
        with:
          name: security-reports
          path: |
            bandit-report.json
            safety-report.json

  codeql:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      actions: read
      contents: read
    
    strategy:
      matrix:
        language: [python]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Initialize CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: ${{ matrix.language }}
      
      - name: Autobuild
        uses: github/codeql-action/autobuild@v3
      
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v3

  dependency-review:
    name: Dependency Review
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/dependency-review-action@v4
        with:
          fail-on-severity: high
```

### 3. Release Workflow

**File**: `.github/workflows/release.yml`

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    name: Create Release
    runs-on: ubuntu-latest
    permissions:
      contents: write
      id-token: write
    
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      
      - name: Install build dependencies
        run: |
          pip install build twine
      
      - name: Build package
        run: |
          python -m build
      
      - name: Create GitHub Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref }}
          draft: false
          prerelease: ${{ contains(github.ref, 'alpha') || contains(github.ref, 'beta') }}
      
      - name: Publish to PyPI
        if: startsWith(github.ref, 'refs/tags/v')
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

## 🔧 Required Repository Secrets

Add these secrets in GitHub repository settings:

### PyPI Deployment
- `PYPI_API_TOKEN`: PyPI API token for package publishing
- `TEST_PYPI_API_TOKEN`: TestPyPI API token for testing

### Security Scanning
- `SECURITY_EMAIL`: Email for security notifications
- `SLACK_WEBHOOK`: Slack webhook for security alerts (optional)

## 🏗️ Self-Hosted Runner Setup

For GPU testing, set up self-hosted runners:

### Requirements
- NVIDIA GPU with CUDA 12.1+
- Docker with NVIDIA Container Runtime
- GitHub Actions Runner software

### Setup Script
```bash
#!/bin/bash
# Setup self-hosted runner for GPU testing

# Install NVIDIA drivers and CUDA toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Register as self-hosted runner
# Follow GitHub's instructions for your repository
```

## 📊 Monitoring and Alerts

### Workflow Status Monitoring
Set up notifications for:
- Failed CI builds
- Security scan findings
- Dependency vulnerabilities
- Release failures

### Performance Monitoring
Track metrics such as:
- Test execution times
- Build durations
- Package size changes
- Security scan results

## 🔄 Workflow Maintenance

### Regular Updates
- Update action versions monthly
- Review and update Python versions
- Update CUDA versions as needed
- Refresh security scanning tools

### Performance Optimization
- Cache dependencies where possible
- Parallelize independent jobs
- Use matrix strategies for testing
- Optimize Docker images

## 🚀 Implementation Steps

1. **Create Workflow Files**
   ```bash
   mkdir -p .github/workflows
   # Create the YAML files using the templates above
   ```

2. **Configure Repository Settings**
   ```bash
   # Enable GitHub Actions
   # Set up branch protection rules
   # Configure required status checks
   ```

3. **Add Repository Secrets**
   ```bash
   # Go to Settings > Secrets and variables > Actions
   # Add all required secrets listed above
   ```

4. **Test Workflows**
   ```bash
   # Create a test PR to verify all workflows
   # Monitor workflow runs in Actions tab
   # Fix any configuration issues
   ```

5. **Enable Branch Protection**
   ```bash
   # Require status checks to pass
   # Require up-to-date branches
   # Require review from CODEOWNERS
   ```

## 🔐 Security Best Practices

### Workflow Security
- Use specific action versions (not @main)
- Limit workflow permissions
- Use environment protection rules
- Review third-party actions regularly

### Secret Management
- Use GitHub secrets for sensitive data
- Rotate secrets regularly
- Use environment-specific secrets
- Audit secret access logs

## 📈 Success Metrics

Monitor these KPIs:
- CI/CD pipeline success rate (target: >95%)
- Average build time (target: <10 minutes)
- Security scan coverage (target: 100%)
- Time to deployment (target: <30 minutes)

## 🤝 Team Workflow

### Developer Workflow
1. Create feature branch
2. Make changes and commit
3. Push branch (triggers CI)
4. Create pull request
5. Address any CI failures
6. Get code review approval
7. Merge when all checks pass

### Release Workflow
1. Update version in pyproject.toml
2. Update CHANGELOG.md
3. Create and push version tag
4. Release workflow automatically publishes
5. Create GitHub release with notes

This comprehensive workflow setup ensures high-quality, secure, and automated development processes for the DP-Flash-Attention project.