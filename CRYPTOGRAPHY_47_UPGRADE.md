# Cryptography 47.0.0 Upgrade

## Overview

This document details the upgrade of the `cryptography` library from version 46.0.7 to 47.0.0, which includes important security updates and performance improvements.

## Changes Made

### 1. Dependency Version Update

**File:** `framework/pyproject.toml`

Updated the cryptography dependency:
```toml
# Before:
"cryptography>=46.0.5,<47.0.0",

# After:
"cryptography>=47.0.0,<48.0.0",
```

### 2. Verification Completed

#### 2.1 CI/CD Environment Compatibility

✅ **OpenSSL 3.0+ Requirement Met**
- GitHub Actions: Uses `ubuntu-22.04` which includes OpenSSL 3.0
- Docker Base Images:
  - Ubuntu base: `ubuntu:24.04` (OpenSSL 3.0)
  - Alpine base: `alpine:3.22` (OpenSSL 3.0)
  - CUDA base: Ubuntu 24.04 (OpenSSL 3.0)
- All environments meet the OpenSSL 3.0+ requirement for cryptography 47.0.0

#### 2.2 Exception Handling Review

✅ **UnsupportedAlgorithm Exception Already Handled**

Verified exception handling patterns in key files:
- `framework/py/flwr/cli/app_cmd/review.py` (line 142): `except (OSError, ValueError, UnsupportedAlgorithm)`
- `framework/py/flwr/cli/supernode/register.py` (line 136): `except (ValueError, UnsupportedAlgorithm)`
- `framework/py/flwr/supernode/cli/flower_supernode.py` (line 289): `except (ValueError, UnsupportedAlgorithm)`

No code changes needed for exception handling.

#### 2.3 Elliptic Curve Usage Review

✅ **Only NIST-Approved Curves Used**

Codebase uses:
- SECP384R1 (NIST-approved)
- No usage of deprecated SECT curves, Brainpool curves, or other unsupported curves

Key files:
- `framework/py/flwr/supercore/primitives/asymmetric.py`: Uses `ec.SECP384R1()`
- `framework/py/flwr/cli/supernode/register.py`: Validates NIST curves

#### 2.4 TLS/Secure Aggregation Components

✅ **Verified Key Cryptography Functions**

Functions using cryptography library:
- `load_pem_private_key()` - Used in asymmetric key loading
- `load_der_private_key()` - Compatible with cryptography 47.0.0
- `InvalidSignature` exception - Already imported and handled
- Secure aggregation module at `framework/py/flwr/common/secure_aggregation/`

## Testing Performed

### Manual Verification

- ✅ Searched codebase for cryptography function usage
- ✅ Verified exception handling patterns
- ✅ Confirmed NIST curve usage (no deprecated curves)
- ✅ Validated CI environment OpenSSL versions
- ✅ Reviewed Docker base image versions

### Automated Tests (CI Pipeline)

The following test suites will be run in CI:

1. **Framework Python Tests**
   ```bash
   cd framework
   python -m pytest py/flwr/
   ```

2. **TLS/Authentication Tests**
   - SuperNode private key loading and validation
   - Public key verification against NIST curves
   - SSH key handling with updated cryptography

3. **Secure Aggregation Tests**
   ```bash
   python -m pytest py/flwr/common/secure_aggregation/
   ```

4. **gRPC TLS Handshake Tests**
   - SuperLink <-> SuperNode connections
   - Client <-> Server encrypted connections

## Lock File Update

### Background

Due to environment permission constraints, the `framework/uv.lock` file requires manual regeneration. This can be done using either:

#### Option 1: Using uv (Recommended)
```bash
cd framework
uv lock --upgrade-package cryptography
```

#### Option 2: Using poetry
```bash
cd framework
poetry update cryptography
```

### Expected Changes

The `uv.lock` file will be updated with:
- Cryptography 47.0.0 (released 2026-02-10)
- Potentially updated versions of dependencies:
  - `cffi` (if needed)
  - `typing-extensions` (if needed for Python < 3.11)

### Validation

After lock file update, verify:
```bash
cd framework
uv sync --python=3.10.19 --locked --all-extras --all-groups
```

## Compatibility Notes

### Breaking Changes in Cryptography 47.0.0

1. **OpenSSL Requirement**: Dropped support for OpenSSL 1.1.x
   - ✅ Our infrastructure uses OpenSSL 3.0+

2. **Platform Changes**:
   - Dropped: 32-bit Windows wheels
   - Dropped: macOS wheels for Intel (10.9+)
   - Added: Universal macOS 2 wheels
   - Impact: Edge devices using 32-bit Windows not affected (not in active benchmarks)

3. **Rust MSRV**: Increased to 1.83.0 for source builds
   - ✅ Not directly relevant to Python packaging

### New Features

1. Performance improvements in elliptic curve operations
2. Enhanced security for cryptographic operations
3. Better error messages for invalid key formats

## Rollback Plan

If issues arise, downgrade using:

```toml
# framework/pyproject.toml
"cryptography>=46.0.5,<47.0.0",
```

Then regenerate lock file:
```bash
cd framework
uv lock --upgrade-package cryptography
# or
poetry update cryptography
```

## Related Issues

- Reference: Cryptography 47.0.0 release notes
- Security updates: [cryptography GitHub](https://github.com/pyca/cryptography)
- OpenSSL 3.0 documentation: [OpenSSL Project](https://www.openssl.org/docs/)

## Sign-Off Checklist

- [x] Dependency version updated in `framework/pyproject.toml`
- [x] CI environment compatibility verified (OpenSSL 3.0+)
- [x] Exception handling patterns reviewed
- [x] Elliptic curves usage validated (NIST curves only)
- [x] Docker base images verified
- [x] Code changes not required (exception handling already in place)
- [ ] Lock file regenerated (pending environment fix)
- [ ] Full test suite passed (pending CI)
- [ ] TLS handshake validation (pending CI)
- [ ] Secure aggregation tests passed (pending CI)

## Next Steps

1. Merge this PR to trigger CI tests
2. Monitor CI pipeline for test results
3. Regenerate lock file if CI passes (or manually as part of CI workflow)
4. Tag release when all tests pass
