.PHONY: clean check-clean test test-unittest test-pytest coverage lint pylint format

# Code quality
lint:
	uv run flake8 gpkit docs

pylint:
	uv run pylint --rcfile=.pylintrc gpkit/
	uv run pylint --rcfile=.pylintrc.docs docs/

# Code formatting
format:
	uv run isort gpkit docs
	uv run black gpkit docs

# Testing
test: test-unittest test-pytest  # Run both test runners

test-unittest:  # Run tests using the original test runner
	uv run python -c "import gpkit.tests; gpkit.tests.run()"

test-pytest:  # Run tests with pytest
	uv run pytest gpkit/tests -v

coverage:  # Run tests with coverage reporting
	uv run pytest gpkit/tests --cov=gpkit --cov-report=term-missing

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

check-clean:
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "Found uncommitted changes:"; \
		git status --porcelain; \
		exit 1; \
	else \
		echo "Working directory is clean."; \
	fi


# Help
help:
	@echo "Available commands:"
	@echo "  lint              Run fast lint checks"
	@echo "  pylint            Run pylint (slow)"
	@echo "  format            Format code with isort and black"
	@echo "  test              Run both unittest and pytest"
	@echo "  test-unittest     Run tests using the original test runner"
	@echo "  test-pytest       Run tests with pytest"
	@echo "  coverage     Run tests with coverage reporting"
	@echo "  clean             Clean build artifacts"
	@echo "  check-clean       Check no uncommitted changes"
	@echo "  help              Show this help message"
