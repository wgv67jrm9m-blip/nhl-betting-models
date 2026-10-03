# NHL Betting Models — Shared Foundation

Shared production-oriented infrastructure for transparent, reproducible pregame NHL
betting models.

This repository does not contain live sportsbook scraping, API credentials, live
odds, or unsupported betting recommendations. Test sportsbook records are synthetic.

## Scope

The shared foundation provides:

- American and decimal odds conversion.
- Raw implied probability calculation.
- Model probability to fair-price conversion.
- Proportional two-way and multiway no-vig normalization.
- Extension interfaces for future Shin and power no-vig methods.
- Strict like-for-like sportsbook market matching.
- Expected-value and diagnostic edge calculations.
- Configurable manual-review eligibility thresholds.
- Bet grading.
- DuckDB persistence schemas.
- Brier score.
- Binary log loss.
- Calibration bins.
- ROI and yield.
- Win rate.
- Like-for-like closing-line value.
- Grouped performance summaries.
- Expanding walk-forward validation.
- Forecast-time feature leakage validation.
- DuckDB schema verification.

## Repository principles

### Point-in-time integrity

Model features must be known at forecast time. Feature timestamps newer than the
model-run timestamp are rejected.

### Time-series validation

Random train/test splitting is intentionally absent. Model validation uses
walk-forward chronological splits.

### Market comparability

Sportsbook offers are comparable only when all of the following match:

- canonical event ID
- canonical subject ID
- market key
- side
- numerical line
- settlement-rule ID

A player SOG Over 3.5 offer is therefore not comparable with Under 3.5 or Over 4.5.

### No-vig pricing

No-vig calculations require a complete market snapshot. The proportional method
preserves both raw implied probabilities and the market overround.

Shin and power methods are represented by explicit extension interfaces but are not
implemented until they can be separately validated.

### Candidate eligibility

Positive EV does not automatically imply that a wager should be placed.

The shared evaluation layer supports configurable EV, edge, probability, and data
quality thresholds. Eligibility means only that a candidate passed the configured
gates for downstream/manual review.

### Raw and historical data

Raw source records, odds snapshots, feature snapshots, model predictions, and
historical model runs should be treated as immutable records. New information should
produce new records rather than rewriting historical state.

## macOS local setup

Open Terminal and change into the repository:

```bash
cd /path/to/nhl-betting-models
```

Confirm the repository:

```bash
pwd
ls -la
```

You should see:

```text
.gitignore
README.md
pyproject.toml
scripts
src
tests
```

### Confirm Python

The project requires Python 3.12 or newer and targets Python 3.12.

```bash
python3 --version
```

Preferred result:

```text
Python 3.12.x
```

If `python3.12` is installed explicitly:

```bash
python3.12 --version
```

### Create the virtual environment

Preferred:

```bash
python3.12 -m venv .venv
```

If `python3` already resolves to Python 3.12 or newer:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Verify the active interpreter:

```bash
which python
python --version
```

The interpreter path should end with:

```text
nhl-betting-models/.venv/bin/python
```

### Upgrade pip

```bash
python -m pip install --upgrade pip
```

Verify:

```bash
python -m pip --version
```

The reported pip location should be inside `.venv`.

### Install the package and development dependencies

```bash
python -m pip install -e ".[dev]"
```

Verify important packages:

```bash
python -m pip show nhl-betting-models duckdb polars numpy scipy pydantic pytest ruff mypy
```

Run an import smoke test:

```bash
python -c "import duckdb, polars, numpy, scipy, pydantic, nhl_betting_models; print('imports OK')"
```

Expected:

```text
imports OK
```

## Validation

### pytest

```bash
python -m pytest
```

Pass criterion:

- exit status 0
- no `FAILED` or `ERROR` results
- all collected tests pass

The exact test count can change as the repository grows; the authoritative criterion
is that every collected test passes.

### Ruff

```bash
python -m ruff check .
```

Expected:

```text
All checks passed!
```

### mypy

```bash
python -m mypy src
```

Pass criterion:

```text
Success: no issues found ...
```

### Initialize DuckDB

```bash
python -m nhl_betting_models.common.database ./nhl_models.duckdb
```

The command should exit successfully and create:

```text
nhl_models.duckdb
```

Run the initializer a second time to verify idempotent DDL:

```bash
python -m nhl_betting_models.common.database ./nhl_models.duckdb
```

The second invocation must also exit successfully.

### Verify the DuckDB schema

```bash
python scripts/verify_duckdb_schema.py ./nhl_models.duckdb
```

Expected final output:

```text
DuckDB schema verification passed: 9 expected tables present.
```

The verifier exits nonzero if a required table or required column is missing.

The generated `nhl_models.duckdb` file is local runtime state and is intentionally
excluded by `.gitignore`.

## Full local validation sequence

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m mypy src
python -m nhl_betting_models.common.database ./nhl_models.duckdb
python -m nhl_betting_models.common.database ./nhl_models.duckdb
python scripts/verify_duckdb_schema.py ./nhl_models.duckdb
```

Every command should exit with status 0.

## macOS troubleshooting

### `python3` not found

If Homebrew is installed:

```bash
brew --version
brew install python@3.12
python3.12 --version
```

If `brew` is also unavailable, install Homebrew using its official installation
instructions, then install `python@3.12`.

### Python is older than 3.12

Install Python 3.12:

```bash
brew install python@3.12
```

Delete any virtual environment created with the older interpreter:

```bash
rm -rf .venv
```

Recreate it:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version
```

Do not reuse a virtual environment created by an older Python interpreter.

### `pip install -e ".[dev]"` fails

Confirm the repository root and environment:

```bash
pwd
ls pyproject.toml
which python
python --version
python -m pip --version
```

Upgrade packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Retry:

```bash
python -m pip install -e ".[dev]"
```

If necessary, retry without cached packages:

```bash
python -m pip install --no-cache-dir -e ".[dev]"
```

### DuckDB or Polars installation fails

Check Mac architecture:

```bash
uname -m
```

Apple Silicon normally reports:

```text
arm64
```

Intel Macs normally report:

```text
x86_64
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install each dependency separately to expose the specific failure:

```bash
python -m pip install --no-cache-dir duckdb
python -m pip install --no-cache-dir polars
```

Verify:

```bash
python -c "import duckdb; print(duckdb.__version__)"
python -c "import polars; print(polars.__version__)"
```

Then retry:

```bash
python -m pip install -e ".[dev]"
```

### `pytest` not found after activation

Check the active interpreter:

```bash
which python
python -m pip show pytest
```

If pytest is missing:

```bash
python -m pip install -e ".[dev]"
```

Use module invocation:

```bash
python -m pytest
```

This avoids ambiguity between a system `pytest` executable and the virtual
environment.