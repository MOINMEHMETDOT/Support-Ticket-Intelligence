"""Evaluation suite.

Separate from tests/ on purpose. tests/ proves the code does what it says and
runs offline in under two seconds; evals/ measures how well the *system plus the
model* answers real questions, costs API calls, and is expected to score well
rather than score perfectly.

Conflating the two produces a test suite that is slow, flaky and quietly
tolerant of regressions.
"""
