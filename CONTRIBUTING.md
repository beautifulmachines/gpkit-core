# Contributing to gpkit-core

Thank you for your interest in contributing to gpkit-core! This guide will help you set up your development environment and get started with contributing.

## Development Environment Setup

### Prerequisites

- Python 3.8 or higher
- Git
- A virtual environment manager (recommended: `venv` or `conda`)

### Setting Up the Development Environment

1. **Clone the Repository**
   ```bash
   git clone https://github.com/beautifulmachines/gpkit-core.git
   cd gpkit-core
   ```

2. **Create and Activate a Virtual Environment**

   Using `venv`:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate  # On Windows
   ```

   Using `conda`:
   ```bash
   conda create -n gpkit-core python=3.11
   conda activate gpkit-core
   ```

3. **Install Development Dependencies**

   Install the package in editable mode with development dependencies:
   ```bash
   make install-dev
   ```

   This will install:
   - Core dependencies
   - Code quality tools (black, isort, pylint, flake8)
   - Testing tools (pytest)

4. **Verify Installation**
   ```bash
   python -c "import gpkit; print(gpkit.__version__)"
   ```

### Git Hooks (recommended)

```bash
uv sync
uv run pre-commit install
```

This runs formatting and linting automatically on each commit.
You can also run checks manually: `make format && make lint`

## Development Workflow

### Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **pylint** for code analysis
- **flake8** for style checking

To format your code:
```bash
make format
```

To run all code quality checks:
```bash
make lint
```

### Running Tests

We use pytest for testing. To run the test suite:
```bash
make test
```

To run specific tests:
```bash
pytest gpkit/tests/test_specific_file.py
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

## Documentation

- Code should be documented using docstrings (Google style)
- Update relevant documentation when making changes
- Build and test documentation locally:
  ```bash
  cd docs
  make html
  ```

## Common Issues and Solutions

### Build Issues
- If you encounter build issues, try:
  ```bash
  make clean
  make install-dev
  ```

### Test Failures
- Ensure all dependencies are installed
- Check if you have the required solvers installed
- Run tests with verbose output:
  ```bash
  pytest -v
  ```

## Getting Help

- Check the [documentation](https://gpkit.readthedocs.io/)
- Open an issue on GitHub
- Join our [discussion forum](https://github.com/beautifulmachines/gpkit-core/discussions)

## License

By contributing to GPkit-Core, you agree that your contributions will be licensed under the project's MIT License. 