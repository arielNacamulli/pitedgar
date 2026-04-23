# Test layout

- `test_<module>.py` — unit tests for `pitedgar/<module>.py`.
- `test_integration.py` — cross-module pipeline tests.
- `test_adversarial.py` — regression hub for adversarial-review findings (issue #38 and the per-fix branches #12–#37).

## Running subsets

    # Only adversarial regressions
    pytest -m adversarial

    # Everything except adversarial (fast feedback loop)
    pytest -m 'not adversarial'

## Adding a regression test

When you fix an adversarial-review finding on a dedicated branch (e.g.
`claude/v0.4/fix-<N>`), add the primary regression test in the most natural
`test_<module>.py` file. Then add ONE smoke test to `test_adversarial.py` that
exercises the cross-cutting invariant — so a future refactor can't regress
every finding at once without `pytest -m adversarial` going red.
