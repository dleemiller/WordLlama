.PHONY: help install install-dev build clean test test-cov lint format pre-commit-install pre-commit-run tag release all

help:
	@echo "WordLlama Development Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install            - Install package and dependencies"
	@echo "  install-dev        - Install package with dev dependencies"
	@echo "  build              - Build Cython extensions"
	@echo "  clean              - Clean build artifacts"
	@echo "  test               - Run tests"
	@echo "  test-cov           - Run tests with coverage"
	@echo "  lint               - Run ruff linter"
	@echo "  format             - Format code with ruff"
	@echo "  pre-commit-install - Install pre-commit hooks"
	@echo "  pre-commit-run     - Run pre-commit on all files"
	@echo "  tag VERSION=X.Y.Z  - Create and push a new release tag"
	@echo "  release VERSION=X.Y.Z - Create tag and GitHub release"
	@echo "  all                - Clean, build, lint, format, and test"

install:
	uv sync

install-dev:
	uv sync --all-extras

build:
	uv run python setup.py build_ext --inplace

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf src/wordllama/**/*.so src/wordllama/**/*.c src/wordllama/**/*.cpp
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

test:
	uv run pytest

test-cov:
	uv run pytest --cov=wordllama --cov-report=html --cov-report=term-missing

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

pre-commit-install:
	uv run pre-commit install

pre-commit-run:
	uv run pre-commit run --all-files

tag:
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: VERSION is required. Usage: make tag VERSION=0.4.0"; \
		exit 1; \
	fi
	@echo "Creating and pushing tag v$(VERSION)..."
	git tag -a v$(VERSION) -m "Release version $(VERSION)"
	git push origin v$(VERSION)
	@echo "- Tag v$(VERSION) created and pushed"
	@echo "- GitHub Actions will build and publish the release"

release:
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: VERSION is required. Usage: make release VERSION=0.4.1"; \
		exit 1; \
	fi
	@if ! command -v gh &> /dev/null; then \
		echo "Error: GitHub CLI (gh) is not installed."; \
		echo "Install: https://cli.github.com/"; \
		exit 1; \
	fi
	@echo "Creating release v$(VERSION)..."
	git tag -a v$(VERSION) -m "Release version $(VERSION)"
	git push origin v$(VERSION)
	gh release create v$(VERSION) --title "Release v$(VERSION)" --generate-notes
	@echo "- Tag v$(VERSION) created and pushed"
	@echo "- GitHub release created with auto-generated notes"
	@echo "- GitHub Actions will build and publish to PyPI"

all: clean build lint format test
	@echo "- All tasks completed successfully"
