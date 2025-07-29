# GitHub Workflows for DP-Flash-Attention

This directory contains documentation and templates for GitHub Actions workflows that should be implemented for the DP-Flash-Attention repository.

## Required Workflows

### 1. Continuous Integration (`ci.yml`)

**Purpose**: Run tests, linting, and quality checks on every push and PR.

**Location**: `.github/workflows/ci.yml`

**Key Features**:
- Multi-Python version testing (3.10, 3.11, 3.12)
- GPU and CPU testing environments  
- Privacy-specific test validation
- Code quality enforcement (black, ruff, mypy)
- Coverage reporting with codecov integration

**Triggers**:
- Push to main branch
- Pull requests to main branch
- Manual workflow dispatch

### 2. Security Scanning (`security.yml`)

**Purpose**: Automated security vulnerability scanning and compliance checks.

**Location**: `.github/workflows/security.yml`

**Key Features**:
- Dependency vulnerability scanning (safety, pip-audit)
- Static analysis security testing (bandit, semgrep)
- Container security scanning for CUDA environments
- License compliance verification
- Privacy-specific security validations

**Schedule**: Daily at 3 AM UTC

### 3. Performance Benchmarking (`benchmark.yml`)

**Purpose**: Automated performance regression testing on GPU hardware.

**Location**: `.github/workflows/benchmark.yml`

**Key Features**:  
- GPU-accelerated benchmark execution
- Performance regression detection
- Memory usage validation
- Privacy overhead measurement
- Results publishing to GitHub Pages

**Triggers**:
- Weekly schedule
- Release tags
- Manual dispatch for performance testing

### 4. Release Automation (`release.yml`)

**Purpose**: Automated package building, testing, and PyPI publishing.

**Location**: `.github/workflows/release.yml`

**Key Features**:
- Semantic version validation
- Multi-platform wheel building (Linux, Windows, macOS)
- CUDA-enabled wheel compilation
- Automated changelog generation
- PyPI publishing with attestations
- GitHub release creation

**Triggers**:
- Push to version tags (v*.*.*)

## Implementation Guide

### Step 1: Create Workflow Files

Copy the templates from `docs/workflows/templates/` to `.github/workflows/`:

```bash
mkdir -p .github/workflows
cp docs/workflows/templates/*.yml .github/workflows/
```

### Step 2: Configure Secrets

Add the following repository secrets:

- `CODECOV_TOKEN`: For coverage reporting
- `PYPI_API_TOKEN`: For package publishing  
- `SECURITY_EMAIL`: For vulnerability notifications

### Step 3: Configure GPU Runners

For CUDA-dependent tests and benchmarks:

1. Set up self-hosted runners with GPU hardware
2. Label runners with `gpu` for appropriate job targeting
3. Configure CUDA toolkit and drivers on runners

### Step 4: Branch Protection Rules

Enable branch protection for `main` with:

- Require status checks to pass
- Require up-to-date branches  
- Require review from code owners
- Include administrators in restrictions

## Workflow Dependencies

### External Actions Used

- `actions/checkout@v4`: Repository checkout
- `actions/setup-python@v4`: Python environment setup
- `codecov/codecov-action@v3`: Coverage reporting
- `github/codeql-action@v2`: Security analysis
- `actions/upload-artifact@v3`: Artifact handling

### Custom Actions

Consider creating custom actions for:

- CUDA environment setup
- Privacy test validation
- Performance benchmark execution
- Multi-GPU test orchestration

## Testing Workflows

### Local Testing

Use `act` to test workflows locally:

```bash
# Install act
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Test CI workflow
act -W .github/workflows/ci.yml

# Test with specific event
act push -W .github/workflows/ci.yml
```

### Staging Environment

1. Create a fork or staging repository
2. Enable workflows in the staging environment
3. Test all workflows with sample changes
4. Validate GPU/CUDA functionality if available

## Monitoring and Alerting

### Workflow Monitoring

- Set up workflow failure notifications via Slack/email
- Monitor workflow run times for performance degradation
- Track success rates across different triggers

### Performance Alerts

- Alert on benchmark regression > 10%
- Monitor memory usage increases
- Track privacy guarantee validation failures

## Maintenance

### Regular Updates

- Update action versions quarterly
- Review and update Python versions annually
- Refresh CUDA toolkit versions with releases
- Update security scanning tools monthly

### Documentation

- Keep workflow documentation synchronized
- Update runbook for common failure scenarios
- Document GPU hardware requirements
- Maintain troubleshooting guides

## Compliance Considerations

### Privacy Requirements

- Ensure no sensitive data in workflow logs
- Validate privacy parameters in automated tests
- Maintain audit trails for privacy-critical changes

### Security Requirements

- Use minimal necessary permissions
- Pin action versions for security
- Regularly audit workflow permissions
- Implement secrets rotation schedule

## See Also

- [Templates Directory](templates/): Actual workflow YAML files
- [Security Policy](../SECURITY.md): Security practices and reporting
- [Contributing Guide](../CONTRIBUTING.md): Development workflow integration