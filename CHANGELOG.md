# Changelog

## 0.2.0 (2026-06-04)

### Security & CI Improvements

- Security tests: header injection, API key validation, HTTPS enforcement, DoS caps
- Max retries capped at 5 globally to prevent abuse
- Header injection protection (`\r`, `\n`, `\x00`) in API key
- Flake8, black, isort, mypy linters fixed and passing
- Python version updated for mypy compatibility (3.8 → 3.10)
- Split CI/CD: `ci.yml` (test+lint+security) and `publish.yml` (PyPI publish)
- `tests/` tracked in git for CI compatibility
- Stable test suite: 134 unit tests, 0 failures
- Coverage threshold 98.5% enforced in `pyproject.toml`
- Branch protection rules for `main` and `before-develop`
- `CODECOV_TOKEN` required for coverage upload

## 0.1.0 (2026-06-01)

### Initial Release

- Chat Completions (standard + streaming SSE)
- Embeddings (single + batch)
- Image Generation (text-to-image)
- Audio (speech-to-text + text-to-speech)
- Video stub (future-ready, NotImplementedError)
- Full error hierarchy (8 custom errors)
- Structured JSON logging with alerts
- Usage statistics per session
- Automatic retry with exponential backoff
- OpenAI drop-in replacement syntax
- HTTPS enforced, API key masked everywhere
- Multi-architecture Docker support
