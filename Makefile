.PHONY: clean check-clean test coverage lint format examples release

# Code quality
lint:
	uv run ruff check .

# Code formatting
format:
	uv run ruff format .
	uv run ruff check --select I --fix .

# Testing
examples:  # Regenerate catalog example output files for docs
	uv run python scripts/regen_examples.py

test:  # Run all tests
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

# Releasing
release: check-clean  # Cut a release: make release VERSION=v0.4.0
	@if [ -z "$(VERSION)" ]; then \
		echo "Usage: make release VERSION=vX.Y.Z"; \
		exit 1; \
	fi
	@echo "$(VERSION)" | grep -qE '^v[0-9]+\.[0-9]+\.[0-9]+$$' || { \
		echo "VERSION must look like vX.Y.Z (got '$(VERSION)')"; \
		exit 1; \
	}
	@git fetch origin main --quiet
	@if [ "$$(git rev-parse HEAD)" != "$$(git rev-parse origin/main)" ]; then \
		echo "HEAD is not up to date with origin/main. Pull or push first."; \
		exit 1; \
	fi
	gh release create $(VERSION) --generate-notes


# Help
help:
	@echo "Available commands:"
	@echo "  lint              Run lint checks"
	@echo "  format            Format code with ruff"
	@echo "  examples          Regenerate catalog example output files"
	@echo "  test              Run all tests"
	@echo "  coverage          Run tests with coverage reporting"
	@echo "  clean             Clean build artifacts"
	@echo "  check-clean       Check no uncommitted changes"
	@echo "  release           Cut a release (make release VERSION=vX.Y.Z)"
	@echo "  help              Show this help message"
