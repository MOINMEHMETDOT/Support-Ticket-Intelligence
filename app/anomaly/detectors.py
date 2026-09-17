"""Anomaly detection over the ticket dataset.

Deliberately deterministic — statistics and rules, no LLM. An operations team
acts on these flags, and a flag that changes between two runs over identical
data is not actionable. The LLM's only role (see summarise_for_llm) is to
describe findings it did not produce.

Two families of detector:
  * Rule-based, from stated SLA policy (schema.SLA_HOURS).
  * Statistical, from the data's own distribution (IQR fences, z-scores),
    computed *within priority group* so a slow Low-priority ticket isn't
    flagged merely because Critical tickets are fast.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import pandas as pd

from app.data.loader import TicketStore
from app.data.schema import SLA_HOURS, UNRESOLVED_STATUSES

# Ordering for report sorting and UI colour.
SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}

# Below this many rated tickets, an agent's mean rating is noise.
MIN_AGENT_SAMPLE = 15


@dataclass
class Anomaly:
    type: str
    severity: str
    title: str
    description: str
    ticket_ids: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.ticket_ids)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["count"] = self.count
        # Keep API payloads bounded; the full list is available via NL query.
        data["ticket_ids"] = self.ticket_ids[:50]
        data["ticket_ids_truncated"] = self.count > 50
        return data


# --- detectors ----------------------------------------------------------


def unresolved_past_sla(df: pd.DataFrame) -> list[Anomaly]:
    """Open or Escalated tickets that have blown their priority's SLA target."""
    out: list[Anomaly] = []
    unresolved = df[df["status"].isin(UNRESOLVED_STATUSES)]

    for priority, target in SLA_HOURS.items():
        breached = unresolved[
            (unresolved["priority"] == priority) & (unresolved["age_hrs"] > target)
        ]
        if breached.empty:
            continue
        worst = breached.loc[breached["age_hrs"].idxmax()]
        out.append(
            Anomaly(
                type="unresolved_past_sla",
                severity={"Critical": "critical", "High": "high"}.get(priority, "medium"),
                title=f"{len(breached)} unresolved {priority} ticket(s) past the {target:g}h SLA",
                description=(
                    f"{len(breached)} {priority}-priority ticket(s) are still Open or "
                    f"Escalated beyond the {target:g}h target. The oldest, "
                    f"{worst['ticket_id']}, has been waiting {worst['age_hrs']:.1f}h "
                    f"({worst['age_hrs'] / target:.1f}x the target) with {worst['agent_id']}."
                ),
                ticket_ids=breached["ticket_id"].tolist(),
                evidence={
                    "priority": priority,
                    "sla_hours": target,
                    "breached": int(len(breached)),
                    "max_age_hrs": round(float(breached["age_hrs"].max()), 2),
                    "median_age_hrs": round(float(breached["age_hrs"].median()), 2),
                },
            )
        )
    return out


def aging_high_priority(df: pd.DataFrame) -> list[Anomaly]:
    """The brief's worked example: unresolved High/Critical tickets over 24h old."""
    stale = df[
        df["status"].isin(UNRESOLVED_STATUSES)
        & df["priority"].isin(["High", "Critical"])
        & (df["age_hrs"] > 24)
    ]
    if stale.empty:
        return []
    return [
        Anomaly(
            type="aging_high_priority",
            severity="critical",
            title=f"{len(stale)} high-priority ticket(s) unresolved for over 24 hours",
            description=(
                f"{len(stale)} High or Critical ticket(s) have been unresolved for more "
                f"than 24 hours, the oldest for {stale['age_hrs'].max():.1f}h. "
                f"{int((stale['status'] == 'Escalated').sum())} of them are already escalated."
            ),
            ticket_ids=stale["ticket_id"].tolist(),
            evidence={
                "threshold_hours": 24,
                "count": int(len(stale)),
                "by_priority": stale["priority"].value_counts().to_dict(),
                "by_status": stale["status"].value_counts().to_dict(),
                "max_age_hrs": round(float(stale["age_hrs"].max()), 2),
            },
        )
    ]


def slow_resolution_outliers(df: pd.DataFrame) -> list[Anomaly]:
    """Resolved tickets beyond the IQR upper fence for their own priority band."""
    out: list[Anomaly] = []
    resolved = df[df["resolution_time_hrs"].notna()]

    for priority in SLA_HOURS:
        band = resolved[resolved["priority"] == priority]
        if len(band) < 10:  # too few points for a meaningful fence
            continue
        q1, q3 = band["resolution_time_hrs"].quantile([0.25, 0.75])
        fence = q3 + 1.5 * (q3 - q1)
        outliers = band[band["resolution_time_hrs"] > fence]
        if outliers.empty:
            continue
        out.append(
            Anomaly(
                type="slow_resolution_outlier",
                severity="high" if priority in ("Critical", "High") else "medium",
                title=f"{len(outliers)} {priority} ticket(s) took abnormally long to resolve",
                description=(
                    f"Typical {priority} resolution is {band['resolution_time_hrs'].median():.1f}h "
                    f"(median). These {len(outliers)} ticket(s) exceeded the statistical "
                    f"outlier threshold of {fence:.1f}h, peaking at "
                    f"{outliers['resolution_time_hrs'].max():.1f}h."
                ),
                ticket_ids=outliers["ticket_id"].tolist(),
                evidence={
                    "priority": priority,
                    "method": "IQR upper fence (Q3 + 1.5 x IQR), within priority band",
                    "median_hrs": round(float(band["resolution_time_hrs"].median()), 2),
                    "threshold_hrs": round(float(fence), 2),
                    "max_hrs": round(float(outliers["resolution_time_hrs"].max()), 2),
                },
            )
        )
    return out


def fast_response_slow_resolution(df: pd.DataFrame) -> list[Anomaly]:
    """Answered quickly, then stalled — the signature of a dropped handoff."""
    resolved = df[df["resolution_time_hrs"].notna()]
    if len(resolved) < 20:
        return []

    quick_reply = resolved["response_time_hrs"].quantile(0.25)
    slow_fix = resolved["resolution_time_hrs"].quantile(0.90)
    stalled = resolved[
        (resolved["response_time_hrs"] <= quick_reply)
        & (resolved["resolution_time_hrs"] >= slow_fix)
    ]
    if stalled.empty:
        return []
    return [
        Anomaly(
            type="fast_response_slow_resolution",
            severity="medium",
            title=f"{len(stalled)} ticket(s) were answered fast but resolved very slowly",
            description=(
                f"{len(stalled)} ticket(s) got a first response within {quick_reply:.1f}h "
                f"(fastest quartile) yet took at least {slow_fix:.1f}h to close (slowest "
                "decile). Quick acknowledgement followed by a long silence usually means "
                "the ticket was picked up and then stalled rather than being genuinely hard."
            ),
            ticket_ids=stalled["ticket_id"].tolist(),
            evidence={
                "response_threshold_hrs": round(float(quick_reply), 2),
                "resolution_threshold_hrs": round(float(slow_fix), 2),
                "count": int(len(stalled)),
                "by_category": stalled["category"].value_counts().to_dict(),
            },
        )
    ]


def low_satisfaction(df: pd.DataFrame) -> list[Anomaly]:
    """Resolved tickets the customer scored 1 or 2."""
    poor = df[df["customer_rating"].notna() & (df["customer_rating"] <= 2)]
    if poor.empty:
        return []
    rated = int(df["customer_rating"].notna().sum())
    share = len(poor) / rated * 100
    return [
        Anomaly(
            type="low_satisfaction",
            severity="high" if share > 20 else "medium",
            title=f"{len(poor)} ticket(s) rated 1-2 out of 5",
            description=(
                f"{len(poor)} of {rated} rated tickets ({share:.1f}%) scored 1 or 2. "
                f"They concentrate in {poor['category'].value_counts().idxmax()} "
                f"({poor['category'].value_counts().max()} tickets) and their median "
                f"resolution time is {poor['resolution_time_hrs'].median():.1f}h versus "
                f"{df['resolution_time_hrs'].median():.1f}h overall."
            ),
            ticket_ids=poor["ticket_id"].tolist(),
            evidence={
                "count": int(len(poor)),
                "share_of_rated_pct": round(share, 1),
                "by_category": poor["category"].value_counts().to_dict(),
                "median_resolution_hrs": round(float(poor["resolution_time_hrs"].median()), 2),
            },
        )
    ]


def agent_rating_outliers(df: pd.DataFrame) -> list[Anomaly]:
    """Agents whose mean rating sits well below their peers'."""
    rated = df[df["customer_rating"].notna()]
    per_agent = rated.groupby("agent_id")["customer_rating"].agg(["mean", "count"])
    eligible = per_agent[per_agent["count"] >= MIN_AGENT_SAMPLE]
    if len(eligible) < 3:
        return []

    mean, std = eligible["mean"].mean(), eligible["mean"].std()
    if not std or pd.isna(std):
        return []

    laggards = eligible[eligible["mean"] < mean - 1.5 * std]
    if laggards.empty:
        return []

    detail = ", ".join(
        f"{agent} ({row['mean']:.2f} over {int(row['count'])} tickets)"
        for agent, row in laggards.iterrows()
    )
    return [
        Anomaly(
            type="agent_rating_outlier",
            severity="medium",
            title=f"{len(laggards)} agent(s) rated well below their peers",
            description=(
                f"Across agents with at least {MIN_AGENT_SAMPLE} rated tickets, the mean "
                f"rating is {mean:.2f} (sd {std:.2f}). {detail} fall more than 1.5 standard "
                "deviations below that. This measures ratings, not effort - check ticket mix "
                "before drawing conclusions."
            ),
            ticket_ids=rated[rated["agent_id"].isin(laggards.index)]["ticket_id"].tolist(),
            evidence={
                "method": "z-score across agent means, min 15 rated tickets",
                "peer_mean": round(float(mean), 2),
                "peer_sd": round(float(std), 2),
                "agents": {
                    agent: {"mean_rating": round(float(r["mean"]), 2), "rated": int(r["count"])}
                    for agent, r in laggards.iterrows()
                },
            },
        )
    ]


def escalation_hotspots(df: pd.DataFrame) -> list[Anomaly]:
    """Issue types escalating at more than twice the baseline rate."""
    baseline = (df["status"] == "Escalated").mean()
    if baseline == 0:
        return []

    grouped = df.groupby("issue_summary").agg(
        total=("ticket_id", "count"),
        escalated=("status", lambda s: int((s == "Escalated").sum())),
    )
    grouped = grouped[grouped["total"] >= 10]
    grouped["rate"] = grouped["escalated"] / grouped["total"]
    hotspots = grouped[grouped["rate"] > baseline * 2].sort_values("rate", ascending=False)
    if hotspots.empty:
        return []

    top = hotspots.index[0]
    return [
        Anomaly(
            type="escalation_hotspot",
            severity="high",
            title=f"{len(hotspots)} issue type(s) escalate at over twice the normal rate",
            description=(
                f"The dataset-wide escalation rate is {baseline * 100:.1f}%. "
                f"'{top}' escalates {hotspots.loc[top, 'rate'] * 100:.1f}% of the time "
                f"({int(hotspots.loc[top, 'escalated'])} of {int(hotspots.loc[top, 'total'])}), "
                "which points at a product or documentation gap rather than agent performance."
            ),
            ticket_ids=df[df["issue_summary"].isin(hotspots.index)]["ticket_id"].tolist(),
            evidence={
                "baseline_rate_pct": round(float(baseline * 100), 1),
                "issues": {
                    issue: {
                        "escalation_rate_pct": round(float(r["rate"] * 100), 1),
                        "escalated": int(r["escalated"]),
                        "total": int(r["total"]),
                    }
                    for issue, r in hotspots.iterrows()
                },
            },
        )
    ]


DETECTORS: tuple[Callable[[pd.DataFrame], list[Anomaly]], ...] = (
    aging_high_priority,
    unresolved_past_sla,
    slow_resolution_outliers,
    fast_response_slow_resolution,
    escalation_hotspots,
    low_satisfaction,
    agent_rating_outliers,
)


# --- orchestration ------------------------------------------------------


def detect(store: TicketStore, types: list[str] | None = None) -> list[Anomaly]:
    """Run every detector (or the named subset), most severe first."""
    selected = DETECTORS
    if types:
        wanted = set(types)
        selected = tuple(d for d in DETECTORS if d.__name__ in wanted)
        if not selected:
            known = ", ".join(d.__name__ for d in DETECTORS)
            raise ValueError(f"No detector matched {types}. Available: {known}")

    found: list[Anomaly] = []
    for detector in selected:
        found.extend(detector(store.frame))

    found.sort(key=lambda a: (SEVERITY_RANK.get(a.severity, 9), -a.count))
    return found


def summarise_for_llm(anomalies: list[Anomaly]) -> str:
    """Compact text handed to the LLM when a narrative summary is requested."""
    if not anomalies:
        return "No anomalies were detected."
    return "\n".join(
        f"- [{a.severity.upper()}] {a.title}: {a.description}" for a in anomalies
    )
