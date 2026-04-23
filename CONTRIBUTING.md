# Contributing

Contributions are welcome! Here's how to get started.

## Development setup

```bash
# Clone the repo
git clone https://github.com/arielNacamulli/pitedgar.git
cd pitedgar

# Install dependencies (requires Poetry)
poetry install

# Run the test suite
pytest
```

## Running tests

```bash
# All tests
pytest

# Single file
pytest tests/test_parser.py

# Single test
pytest tests/test_parser.py::test_deduplication

# With verbose output
pytest -v
```

## Making changes

1. Fork the repository and create a branch from `main`.
2. Make your changes and add tests for any new behaviour.
3. Ensure all tests pass: `pytest`.
4. Open a pull request with a clear description of the change and why.

## Code style

- Standard Python — no formatter is enforced yet, but keep style consistent with surrounding code.
- Type hints are encouraged for public API functions.
- Docstrings on public classes and functions are appreciated.

## Adversarial regression tests

The `tests/test_adversarial.py` file is the umbrella for cross-cutting
regression tests surfaced during adversarial review (see issue #38). When
fixing an adversarial-review finding on a dedicated branch, add the primary
regression test in the most natural `test_<module>.py` file and add one
smoke test to `test_adversarial.py` covering the cross-cutting invariant.

Run the adversarial subset with:

```bash
pytest -m adversarial
```

See [tests/README.md](tests/README.md) for the full test layout and guidance
on adding new regression tests.

## Reporting bugs

Open an issue and include:
- Python version and OS
- Full traceback
- Minimal reproduction steps

## License

By contributing you agree that your contributions will be licensed under the [MIT License](LICENSE).
