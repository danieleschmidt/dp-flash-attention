# Manual Setup Required

Due to GitHub App permission limitations, the following setup tasks require manual intervention by repository maintainers:

## GitHub Actions Workflows

The repository includes comprehensive CI/CD workflow templates in `docs/workflows/templates/`, but they must be manually created in `.github/workflows/` due to permission restrictions.

### Required Workflows

1. **CI Pipeline** (`ci.yml`)
   - Location: Copy from `docs/workflows/templates/ci.yml` to `.github/workflows/ci.yml`
   - Purpose: PR validation, testing, security scanning
   - Required secrets: None (uses GitHub token)

2. **Security Scanning** (`security.yml`)
   - Location: Copy from `docs/workflows/templates/security.yml` to `.github/workflows/security.yml`
   - Purpose: Dependency scanning, SAST, privacy audit
   - Required secrets: `SNYK_TOKEN` (optional, for enhanced scanning)

3. **Deployment** (when ready for releases)
   - Location: Template in `docs/workflows/DEPLOYMENT.md`
   - Purpose: Automated PyPI releases, Docker builds
   - Required secrets: `PYPI_API_TOKEN`, `DOCKER_USERNAME`, `DOCKER_PASSWORD`

## Repository Settings

### Branch Protection Rules
Configure the following branch protection rules for `main`:

```json
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "ci/tests",
      "ci/security-scan",
      "ci/privacy-tests"
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 2,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
```

### Repository Configuration
1. Enable vulnerability alerts
2. Enable dependency graph
3. Enable Dependabot security updates
4. Configure topics: `differential-privacy`, `flash-attention`, `privacy-preserving-ml`, `cuda`, `pytorch`

## External Integrations

### Code Quality
- **Codecov**: Configure token in repository secrets as `CODECOV_TOKEN`
- **SonarCloud**: Set up project and add `SONAR_TOKEN` to secrets

### Security
- **Snyk**: Add `SNYK_TOKEN` for enhanced vulnerability scanning
- **FOSSA**: Configure for license compliance (optional)

### Monitoring
- **Sentry**: Add `SENTRY_DSN` for error tracking (optional)
- **DataDog**: Configure for performance monitoring (optional)

## Privacy-Specific Setup

### Privacy Audit Infrastructure
1. Set up automated privacy testing infrastructure
2. Configure membership inference attack testing
3. Establish formal verification pipeline for DP guarantees

### Compliance Requirements
1. Configure SBOM generation workflows
2. Set up privacy impact assessment procedures
3. Establish audit trail for privacy parameter changes

## Manual Verification Checklist

After setting up workflows, verify:

- [ ] CI pipeline runs successfully on PR creation
- [ ] Security scans complete without critical vulnerabilities
- [ ] Privacy tests pass with expected DP guarantees
- [ ] Documentation builds and deploys correctly
- [ ] Branch protection rules are enforced
- [ ] All required secrets are configured
- [ ] External integrations are working

## Support

For questions about setup requirements:
- Create an issue with the `setup-help` label
- Review workflow templates in `docs/workflows/`
- Consult the comprehensive documentation in `docs/`