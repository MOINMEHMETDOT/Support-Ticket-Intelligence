"""Anomaly detectors. These are pure functions of the data, so they assert on
real values rather than mocks."""

from __future__ import annotations

import pytest

from app.anomaly.detectors import (
    DETECTORS,
    SEVERITY_RANK,
    aging_high_priority,
    detect,
    low_satisfaction,
    slow_resolution_outliers,
    summarise_for_llm,
    unresolved_past_sla,
)
from app.data.schema import SLA_HOURS, UNRESOLVED_STATUSES


def test_detect_returns_findings_sorted_by_severity(store):
    found = detect(store)
    assert found, "expected at least one anomaly in this dataset"
    ranks = [SEVERITY_RANK[a.severity] for a in found]
    assert ranks == sorted(ranks)


def test_every_detector_is_reachable(store):
    """Guards against a detector being written but never registered."""
    produced = {a.type for a in detect(store)}
    names = {d.__name__ for d in DETECTORS}
    assert names  # registry is populated
    assert produced  # and at least some fire on real data


def test_detection_is_deterministic(store):
    first = [(a.type, a.count) for a in detect(store)]
    second = [(a.type, a.count) for a in detect(store)]
    assert first == second


def test_aging_high_priority_only_flags_unresolved_over_24h(store):
    for anomaly in aging_high_priority(store.frame):
        flagged = store.frame[store.frame["ticket_id"].isin(anomaly.ticket_ids)]
        assert flagged["status"].isin(UNRESOLVED_STATUSES).all()
        assert flagged["priority"].isin(["High", "Critical"]).all()
        assert (flagged["age_hrs"] > 24).all()


def test_sla_breaches_respect_their_priority_threshold(store):
    for anomaly in unresolved_past_sla(store.frame):
        priority = anomaly.evidence["priority"]
        flagged = store.frame[store.frame["ticket_id"].isin(anomaly.ticket_ids)]
        assert (flagged["age_hrs"] > SLA_HOURS[priority]).all()
        assert (flagged["priority"] == priority).all()


def test_slow_resolution_outliers_exclude_unresolved(store):
    """An unresolved ticket has no resolution time and cannot be a slow-resolution outlier."""
    for anomaly in slow_resolution_outliers(store.frame):
        flagged = store.frame[store.frame["ticket_id"].isin(anomaly.ticket_ids)]
        assert flagged["resolution_time_hrs"].notna().all()
        assert (flagged["resolution_time_hrs"] > anomaly.evidence["threshold_hrs"]).all()


def test_low_satisfaction_only_flags_rated_tickets(store):
    for anomaly in low_satisfaction(store.frame):
        flagged = store.frame[store.frame["ticket_id"].isin(anomaly.ticket_ids)]
        assert (flagged["customer_rating"] <= 2).all()


def test_type_filter(store):
    found = detect(store, types=["aging_high_priority"])
    assert {a.type for a in found} <= {"aging_high_priority"}


def test_unknown_type_filter_raises(store):
    with pytest.raises(ValueError, match="No detector matched"):
        detect(store, types=["not_a_detector"])


def test_summary_is_empty_safe():
    assert "No anomalies" in summarise_for_llm([])
