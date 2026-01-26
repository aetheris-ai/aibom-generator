# Contributing to OWASP AIBOM Generator

Thank you for your interest in contributing to the OWASP AIBOM Generator! This project is part of the [OWASP GenAI Security Project](https://genai.owasp.org) and welcomes contributions from the community.

## Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/aibom-generator.git
   cd aibom-generator
   ```
3. **Set up your environment**:

   ```bash
   # Using Docker (recommended)
   docker build -t aibom .
   docker run -p 7860:7860 aibom

   # Or local Python setup
   cd HF_files/aibom-generator
   pip install -r requirements.txt
   ```

## Development Workflow

### Branch Naming

Use descriptive branch names following this pattern:

```
type/issue-number-description
```

**Types:**

- `feat` - New features
- `fix` - Bug fixes
- `docs` - Documentation changes
- `refactor` - Code refactoring
- `test` - Adding or updating tests

**Examples:**

- `feat/17-schema-validation`
- `fix/13-purl-encoding`
- `docs/contributing-guide`

### Pull Request Process

1. Create a branch from `main`
2. Make your changes with clear, focused commits
3. Push to your fork and open a PR
4. Link your PR to any related issues
5. Respond to review feedback

### Commit Messages

Use conventional commits:

```
type(scope): description

[optional body]
```

**Examples:**

- `feat(validation): add CycloneDX 1.6 schema validation`
- `fix(generator): correct PURL encoding for model IDs`

## Code Standards

### Python Style

- **Python 3.8+** compatibility required
- Follow existing patterns in the codebase
- Use type hints for function signatures

### Import Organization

```python
# Standard library
import json
import logging

# Third-party
import requests
from huggingface_hub import HfApi

# Local imports
from .utils import calculate_completeness_score
```

### Output Conventions

- **User-facing output**: Use `print()` with emoji indicators

  ```python
  print("✅ AIBOM generated successfully")
  print("⚠️ Warning: Missing license information")
  print("❌ Error: Model not found")
  ```

- **Programmatic logging**: Use `logger` with lazy formatting
  ```python
  logger.info("Processing model: %s", model_id)
  logger.warning("Schema validation found %d issues", count)
  ```

## Project Architecture

```
HF_files/aibom-generator/
├── src/aibom-generator/
│   ├── generator.py          # Core AIBOM generation
│   ├── enhanced_extractor.py # Metadata extraction
│   ├── field_registry.json   # Registry-driven field definitions
│   ├── api.py                # FastAPI endpoints
│   ├── cli.py                # Command-line interface
│   ├── utils.py              # Completeness scoring
│   └── validation.py         # CycloneDX schema validation
└── requirements.txt
```

### Key Concepts

- **Registry-driven design**: Field definitions come from `field_registry.json`
- **CycloneDX 1.6 compliance**: All generated AIBOMs must validate against the schema
- **Completeness scoring**: Helps users understand AIBOM quality

## Building and Running

### Docker (Recommended)

```bash
docker build -t aibom .
docker run -p 7860:7860 aibom
```

### Local Development

```bash
cd HF_files/aibom-generator
pip install -r requirements.txt

# Run API server
python -m uvicorn src.aibom_generator.api:app --reload --port 7860

# Or use CLI
python -m src.aibom_generator.cli --model_id "microsoft/DialoGPT-medium"
```

## Areas Welcoming Contributions

We especially welcome contributions in these areas:

- **Unit test coverage** - Help us build a comprehensive test suite
- **SPDX 3.0 export** - Add support for SPDX AI Profile format
- **Model-specific extractors** - Better metadata extraction for specific model types
- **Documentation** - Improve guides, examples, and API docs
- **UI/UX enhancements** - Improve the web interface
- **Performance optimization** - Faster extraction and generation

## Reporting Issues

Before creating an issue:

1. **Search existing issues** to avoid duplicates
2. Use a **clear, descriptive title**
3. For bugs, include:
   - Steps to reproduce
   - Expected vs actual behavior
   - Model ID if applicable
   - Error messages

**Issue tracker:** [GitHub Issues](https://github.com/GenAI-Security-Project/aibom-generator/issues)

## License

This project is licensed under the [Apache License 2.0](LICENSE). By contributing, you agree that your contributions will be licensed under the same license.

## Community

- **OWASP GenAI Security Project**: [genai.owasp.org](https://genai.owasp.org)
- **AIBOM Initiative**: [genai.owasp.org/ai-sbom-initiative](https://genai.owasp.org/ai-sbom-initiative)
- **Slack**: `#team-genai-aibom` on [owasp.slack.com](https://owasp.slack.com)

---

Thank you for helping make AI transparency practical!
