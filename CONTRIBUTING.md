# Contributing to HFT Simulator

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## How to Contribute

### Reporting Bugs

1. Check existing [Issues](https://github.com/Xyerophyte/hft-simulator/issues) to avoid duplicates
2. Create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (Python version, OS)

### Suggesting Features

1. Open an issue with the `enhancement` label
2. Describe the feature and its use case
3. Discuss implementation approach if possible

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes following code style guidelines
4. Write/update tests as needed
5. Commit with clear messages: `git commit -m "Add feature: description"`
6. Push to your fork: `git push origin feature/your-feature`
7. Open a Pull Request

## Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Write docstrings for public functions/classes
- Keep functions focused and under 50 lines
- Use meaningful variable names

## Testing

```bash
# Run tests before submitting
python -m pytest tests/ -v

# Check coverage
python -m pytest tests/ --cov=src
```

## Questions?

Open an issue or start a discussion. We're happy to help!
