"""Ingestion, derived columns and the read-only execution path."""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.data.loader import QueryExecutionError
from app.data.schema import UNRESOLVED_STATUSES


def test_loads_all_rows(store):
    assert len(store.frame) == 500


def test_age_column_is_derived_against_reference_date(store):
    assert "age_hrs" in store.frame.columns
    assert (store.frame["age_hrs"] >= 0).all()
    # The newest ticket defines "now", so its age is zero.
    assert store.frame["age_hrs"].min() == pytest.approx(0.0, abs=0.01)


def test_nulls_mean_unresolved_not_missing(store):
    """resolution_time and rating are NULL exactly for unresolved tickets."""
    df = store.frame
    unresolved = df["status"].isin(UNRESOLVED_STATUSES)
    assert df.loc[unresolved, "resolution_time_hrs"].isna().all()
    assert df.loc[~unresolved, "resolution_time_hrs"].notna().all()
    assert df.loc[unresolved, "customer_rating"].isna().all()
    assert df.loc[~unresolved, "customer_rating"].notna().all()


def test_select_executes(store):
    result = store.execute_readonly("SELECT COUNT(*) AS n FROM tickets")
    assert result.rows[0][0] == 500
    assert result.columns == ["n"]


def test_writes_are_blocked_at_the_connection(store):
    """Even if the string guard were bypassed, the connection must refuse writes."""
    with pytest.raises(QueryExecutionError):
        store.execute_readonly("DELETE FROM tickets")
    assert store.execute_readonly("SELECT COUNT(*) FROM tickets").rows[0][0] == 500


def test_result_truncation_flag(store):
    """A row cap is applied and reported. The store fixture is session-scoped, so
    the narrowed setting is restored rather than leaking into later tests."""
    original = store._settings
    store._settings = replace(original, sql_max_rows=10)
    try:
        result = store.execute_readonly("SELECT ticket_id FROM tickets")
    finally:
        store._settings = original

    assert result.row_count == 10
    assert result.truncated is True
    # And the cap is genuinely restored.
    assert store.execute_readonly("SELECT ticket_id FROM tickets").row_count == 200


def test_profile_reports_dataset_shape(store):
    profile = store.profile()
    assert profile["rows"] == 500
    assert profile["agents"] == 12
    assert sum(profile["status_counts"].values()) == 500
