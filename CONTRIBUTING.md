# Contributing to gpkit-core

Thank you for your interest in contributing to gpkit-core! This guide will help you set up your development environment and get started with contributing.

## Development Environment Setup

### Prerequisites

- Python 3.11 or higher
- Git
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Setting Up

1. **Clone the repository**
   ```bash
   git clone https://github.com/beautifulmachines/gpkit-core.git
   cd gpkit-core
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Install git hooks** (recommended — runs formatting and linting on each commit)
   ```bash
   uv run pre-commit install
   ```

4. **Verify**
   ```bash
   make test
   ```

## Development Workflow

### Code Style

We use [ruff](https://docs.astral.sh/ruff/) for code formatting, import sorting, and
linting.

To format your code:
```bash
make format
```

To run all code quality checks:
```bash
make lint
```

### Running Tests

To run the full test suite:
```bash
make test
```

To run specific tests:
```bash
uv run pytest gpkit/tests/test_specific_file.py
```

### Making Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

3. Run tests and code quality checks:
   ```bash
   make lint
   make test
   ```

4. Push your changes:
   ```bash
   git push origin feature/your-feature-name
   ```

### Pull Request Process

1. Create a pull request from your branch to `main`
2. Ensure all tests pass
3. Update documentation if necessary
4. Wait for review and address any feedback

## Releasing

Versions are derived automatically from git tags (via `hatch-vcs`) — there is no
`__version__` to hand-edit and no version-bump PR to remember. `gpkit.__version__`
reflects whatever tag is checked out:

- On a commit exactly at tag `v0.4.0`, with no local changes: `0.4.0`.
- On any other commit: `0.4.1.devN+g<hash>`, where `N` is commits since the last tag.
- With uncommitted local changes: the version always carries a dev/dirty suffix, even
  if `HEAD` is itself tagged — so a clean `X.Y.Z` version only ever comes from a
  committed, exactly-tagged commit.

To cut a release:

1. Make sure `main` is green.
2. Create a GitHub Release with a new tag, e.g.:
   ```bash
   gh release create v0.4.0 --generate-notes
   ```
   (or via the GitHub UI: Releases → Draft a new release → tag `vX.Y.Z` → Generate
   release notes → Publish.)
3. That's it — publishing the release triggers `.github/workflows/publish.yml`,
   which builds the package (version read straight from the new tag) and uploads it
   to PyPI.

Use plain `vMAJOR.MINOR.PATCH` tags. Bump PATCH for fixes, MINOR for backwards-compatible
additions, MAJOR for breaking changes.

## Documentation

- Code should be documented using docstrings (Google style)
- Update relevant documentation when making changes
- Build and test documentation locally:
  ```bash
  cd docs && make html
  ```

## Common Issues and Solutions

### Test Failures
- Ensure all dependencies are installed: `uv sync`
- Check if you have the required solvers installed
- Run tests with verbose output:
  ```bash
  uv run pytest -v
  ```

## Getting Help

- Open an issue on GitHub

## License

By contributing to gpkit-core, you agree that your contributions will be licensed under the project's MIT License.
