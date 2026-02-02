# PyEarthMesh Release Guide

## Overview
This guide explains how to release pyearthmesh to PyPI using the automated GitHub Actions workflow.

## Prerequisites

### 1. PyPI Account and API Token
- Create an account on [PyPI](https://pypi.org/) if you don't have one
- Generate an API token at https://pypi.org/manage/account/token/
- The token should have upload permissions for the pyearthmesh package

### 2. GitHub Secrets
Configure the following secrets in your GitHub repository settings (Settings → Secrets and variables → Actions):

- `PYEARTHMESH_PYPI`: Your PyPI API token for production releases
- `PYPI_TEST` (optional): Your TestPyPI API token for pre-releases

To add secrets:
1. Go to your repository on GitHub
2. Click Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Add the secret name and value

### 3. Local Testing (Optional)
Before creating a release, you can test the build locally:

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Check the package
twine check dist/*
```

## Release Process

### Step 1: Update Version Number
Update the version in pyproject.toml:
```toml
version = "0.1.0"  # Change to your new version
```

Also update it in pyearthmesh/__init__.py:
```python
__version__ = "0.1.0"  # Must match pyproject.toml
```

### Step 2: Update CHANGELOG
Create or update a CHANGELOG.md file documenting changes in the new version.

### Step 3: Commit Changes
```bash
git add pyproject.toml pyearthmesh/__init__.py CHANGELOG.md
git commit -m "Bump version to X.Y.Z"
git push origin main
```

### Step 4: Create a GitHub Release
1. Go to your repository on GitHub
2. Click "Releases" → "Draft a new release"
3. Click "Choose a tag" and create a new tag (e.g., `v0.1.0`)
   - **Important**: The tag must start with 'v' (e.g., v0.1.0, v1.2.3)
   - The version after 'v' must match the version in pyproject.toml
4. Set the release title (e.g., "Release v0.1.0")
5. Add release notes describing changes
6. Check "Set as a pre-release" if this is a beta/alpha version (will publish to TestPyPI)
7. Click "Publish release"

### Step 5: Monitor the Release
The GitHub Actions workflow will automatically:
1. Build the package
2. Verify the package metadata
3. Check that the tag version matches the package version
4. Publish to TestPyPI (if pre-release) or PyPI (if full release)

Monitor progress at: https://github.com/YOUR_USERNAME/pyearthmesh/actions

## Release Types

### Full Release (Production)
- Uncheck "Set as a pre-release" when creating the GitHub release
- Package will be published to PyPI
- Users can install with: `pip install pyearthmesh`

### Pre-release (Testing)
- Check "Set as a pre-release" when creating the GitHub release
- Package will be published to TestPyPI
- Users can install with: `pip install --index-url https://test.pypi.org/simple/ pyearthmesh`

## Manual Release (Alternative)

If you prefer to release manually without GitHub Actions:

```bash
# Install required tools
pip install build twine

# Build the distribution packages
python -m build

# Check the packages
twine check dist/*

# Upload to TestPyPI (testing)
twine upload --repository testpypi dist/*

# Upload to PyPI (production)
twine upload dist/*
```

## Troubleshooting

### Version Mismatch Error
If the workflow fails with a version mismatch error:
- Ensure the version in pyproject.toml matches the git tag (without the 'v' prefix)
- Example: Tag `v0.1.0` should match version `0.1.0` in pyproject.toml

### Authentication Error
If upload fails with authentication error:
- Verify the `PYEARTHMESH_PYPI` secret is correctly set in GitHub
- Ensure the API token has not expired
- Check that the token has upload permissions

### Package Already Exists
PyPI does not allow overwriting existing versions:
- You must increment the version number for each release
- Once a version is published, it cannot be removed or replaced

## Post-Release

After a successful release:

1. Verify the package on PyPI: https://pypi.org/project/pyearthmesh/
2. Test installation: `pip install pyea
3. Update documentation if 


## Version

Follow [Semantic Versioning](https://semver.org/):
- MAJOR version for incompatible API changes
- 


Example: `1.2.3

## Files Involved in Release

- `
- `pyearthmesh/__init__.py` - Package version
- `MANIFEST.in` - Files 
- `.github/workflows/release.yml` - Automated release workflow
- 
- `LICENSE` - Package license
