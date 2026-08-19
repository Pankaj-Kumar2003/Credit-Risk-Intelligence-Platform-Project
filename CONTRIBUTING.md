# Contributing

## Setup

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python scripts/generate_mock_data.py   # or drop a real dataset into data/raw/
    python main.py --stage all

## Before opening a PR

- Run `pytest -v` — all tests must pass.
- Run `python main.py --stage all` end-to-end without errors.
- Keep config-driven: no hardcoded paths or column names in `src/`.
- Add/update tests for any new feature or pipeline stage.
- Follow PEP 8; type-hint public functions; docstring every module.

## Branching

- `main` is always deployable.
- Feature branches: `feature/<short-description>`.
- Bugfix branches: `fix/<short-description>`.
