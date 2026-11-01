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
2. Run:
   ```bash
   make release V=X.Y.Z
   ```
   This checks that your working directory is clean and up to date with
   `origin/main`, then creates a GitHub Release tagged `vX.Y.Z` (equivalent to
   `gh release create vX.Y.Z --generate-notes`, or via the GitHub UI: Releases →
   Draft a new release → tag `vX.Y.Z` → Generate release notes → Publish).
3. That's it — publishing the release triggers `.github/workflows/publish.yml`,
   which builds the package (version read straight from the new tag) and uploads it
   to PyPI.

Use plain `vMAJOR.MINOR.PATCH` tags. While gpkit-core is pre-1.0, bump MINOR for breaking
changes and additions, and PATCH for fixes. Breaking changes don't get a deprecation period.

### Releases that change output

gassolar's CI installs gpkit-core from PyPI and fails on snapshot drift, so a change to
reported output reaches it through a release, not a merge. The order is:

1. Merge to gpkit-core.
2. Cut the release.
3. Regenerate snapshots in gassolar and commit them:
   ```bash
   uv run pytest tests/test_catalog.py::test_catalog_snapshots
   ```

`make check-downstream` shows whether a change affects output in sibling repos, and leaves the
regenerated snapshots there to commit. gassolar's CI is red between steps 2 and 3, so have that
commit ready before releasing. The nightly Downstream workflow goes red at step 1, which is
expected. gpkit-core's own snapshots passing doesn't show that other repos' output is unchanged.

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
