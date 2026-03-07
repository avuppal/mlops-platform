# Contributing to mlops-platform

Thank you for your interest in contributing! This document explains the process.

## Development Setup

```bash
git clone https://github.com/avuppal/mlops-platform.git
cd mlops-platform
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-cov
```

## Running Tests

```bash
pytest -v --tb=short          # run all tests
pytest tests/test_drift.py -v  # run a specific module
pytest --cov=src --cov-report=term-missing  # with coverage
```

All PRs must pass `pytest` with **zero failures**.

## Code Style

- Follow PEP 8 (max line length: 99)
- Type-annotate all public functions and methods
- Document public APIs with NumPy-style docstrings
- No new external dependencies without discussion in an issue first

## Submitting a PR

1. Fork the repo and create a feature branch: `git checkout -b feat/my-feature`
2. Write tests for any new behaviour
3. Ensure `pytest` passes locally
4. Open a PR with a clear description of *what* and *why*

## Reporting Issues

Please include:
- Python version (`python --version`)
- Reproduction steps
- Expected vs. actual behaviour
