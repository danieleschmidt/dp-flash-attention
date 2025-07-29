# Pull Request

## 📋 Description

<!-- Provide a clear and concise description of your changes -->

## 🔗 Related Issues

<!-- Link to any related issues -->
- Fixes #(issue number)
- Closes #(issue number)
- Related to #(issue number)

## 🧪 Type of Change

<!-- Mark the relevant option with an [x] -->

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that causes existing functionality to change)
- [ ] 📚 Documentation update
- [ ] 🔧 Code refactoring (no functional changes)
- [ ] ⚡ Performance improvement
- [ ] 🧪 Test additions or improvements
- [ ] 🔒 Security enhancement

## 🔒 Privacy Impact Assessment

<!-- REQUIRED for all code changes affecting DP mechanisms -->

### Privacy Analysis
- [ ] ✅ No privacy impact - changes don't affect DP guarantees
- [ ] 🔍 Privacy impact assessed - see analysis below
- [ ] ⚠️ Privacy review needed - uncertain about impact

### Privacy Checklist (if applicable)
- [ ] Theoretical privacy guarantees maintained or improved
- [ ] Privacy parameters validation updated
- [ ] Sensitivity analysis reviewed and updated
- [ ] Composition bounds verified
- [ ] Privacy tests added/updated
- [ ] Documentation updated with privacy implications

### Privacy Analysis Details
<!-- If privacy impact exists, provide detailed analysis -->
```
Privacy guarantee before: (ε, δ) = (?, ?)
Privacy guarantee after:  (ε, δ) = (?, ?)
Rationale: ...
Verification method: ...
```

## 🚀 Performance Impact

<!-- Assess performance implications -->

### Performance Analysis
- [ ] ✅ No performance impact expected
- [ ] 📈 Performance improvement - see benchmarks below  
- [ ] 📉 Performance regression - justified by benefits
- [ ] 🤔 Performance impact unclear - benchmarks needed

### Benchmark Results (if applicable)
<!-- Include before/after performance measurements -->
```
Test Case: [configuration]
Before: X.X ms ± Y.Y ms
After:  X.X ms ± Y.Y ms
Change: ±Z.Z% (improvement/regression)
```

## 🧪 Testing

### Test Coverage
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated  
- [ ] Privacy-specific tests added/updated
- [ ] Performance benchmarks added/updated
- [ ] GPU tests added/updated (if applicable)
- [ ] All existing tests pass

### Test Strategy
<!-- Describe your testing approach -->
- Test cases cover: ...
- Edge cases considered: ...
- Error conditions tested: ...

## 📖 Documentation

- [ ] Code comments added/updated
- [ ] API documentation updated
- [ ] README updated (if needed)
- [ ] Changelog entry added
- [ ] Tutorial/example updated (if needed)

## ✅ Checklist

### Code Quality
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Code is well-commented
- [ ] No debug code or TODO comments left behind
- [ ] Import statements cleaned up

### Security
- [ ] No hardcoded secrets or API keys
- [ ] Input validation added where needed
- [ ] No security vulnerabilities introduced
- [ ] Privacy parameters properly validated

### Compatibility
- [ ] Changes are backwards compatible (or breaking changes documented)
- [ ] Works with supported Python versions (3.10+)
- [ ] CUDA compatibility maintained (if applicable)
- [ ] Dependencies updated appropriately

## 🔍 Reviewer Focus Areas

<!-- Help reviewers by highlighting specific areas that need attention -->

### Please Pay Special Attention To:
- [ ] Privacy guarantee correctness
- [ ] Performance impact on critical paths
- [ ] CUDA kernel implementation
- [ ] API design and usability
- [ ] Error handling and edge cases
- [ ] Memory usage patterns
- [ ] Thread safety (if concurrent)

### Specific Questions for Reviewers:
1. Question about approach/design choice?
2. Concerns about specific implementation?
3. Alternative suggestions welcome?

## 🚨 Breaking Changes

<!-- If this introduces breaking changes, describe them -->

### API Changes
- [ ] No breaking changes
- [ ] Breaking changes documented below

### Migration Guide (if breaking changes)
```python
# Before
old_api_usage()

# After  
new_api_usage()
```

## 📸 Screenshots/Examples

<!-- Include screenshots for UI changes or code examples for API changes -->

### Code Example
```python
# Example usage of new feature
from dp_flash_attention import NewFeature

feature = NewFeature(params)
result = feature.process(data)
```

## 🔗 Additional Context

<!-- Any additional information that reviewers should know -->

### Dependencies
- New dependencies added: ...
- Version requirements changed: ...

### Future Work
- Follow-up issues to create: ...
- Known limitations: ...
- Potential improvements: ...

---

## 📝 Merge Requirements

<!-- Automated checks that must pass -->

### Required Checks
- [ ] All CI checks pass
- [ ] Code coverage maintained/improved
- [ ] Security scans pass
- [ ] Privacy tests pass
- [ ] Performance benchmarks acceptable
- [ ] Documentation builds successfully

### Review Requirements
- [ ] Code review from maintainer
- [ ] Privacy review (if privacy-impacting)
- [ ] Performance review (if performance-impacting)
- [ ] Security review (if security-impacting)

---

**Note**: Privacy-related changes require additional review time and may need theoretical validation before merging.