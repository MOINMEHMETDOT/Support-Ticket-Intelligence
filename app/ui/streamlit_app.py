"""Streamlit UI.

Talks to the REST API over HTTP rather than importing the engine directly, so
the UI exercises exactly the same surface an external consumer would. If the
API is down the UI says so and shows the command to start it, instead of
failing with a stack trace.
"""

from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 120.0

SEVERITY_STYLE = {
    "critical": ("🔴", "#d62728"),
    "high": ("🟠", "#ff7f0e"),
    "medium": ("🟡", "#bcbd22"),
    "low": ("🔵", "#1f77b4"),
}

st.set_page_config(page_title="Support Ticket Intelligence", page_icon="🎫", layout="wide")


def api_get(path: str, **params) -> dict | None:
    try:
        response = httpx.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT)
    except httpx.HTTPError as exc:
        st.error(
            f"Cannot reach the API at {API_URL} ({exc}).\n\n"
            "Start it with: `uvicorn app.api.main:app --reload`"
        )
        return None
    if response.status_code >= 400:
        st.error(_detail(response))
        return None
    return response.json()


def api_post(path: str, payload: dict) -> dict | None:
    try:
        response = httpx.post(f"{API_URL}{path}", json=payload, timeout=TIMEOUT)
    except httpx.HTTPError as exc:
        st.error(f"Cannot reach the API at {API_URL} ({exc}).")
        return None
    if response.status_code >= 400:
        st.error(_detail(response))
        return None
    return response.json()


def api_post_quiet(path: str, payload: dict) -> tuple[bool, str, dict | None]:
    """POST without rendering errors itself, so the caller can place them.

    st.error() always draws in the main pane; the sidebar config panel needs its
    failures shown next to the form that caused them.
    """
    try:
        response = httpx.post(f"{API_URL}{path}", json=payload, timeout=TIMEOUT)
    except httpx.HTTPError as exc:
        return False, f"Cannot reach the API at {API_URL}: {exc}", None
    if response.status_code >= 400:
        return False, _detail(response), None
    return True, "", response.json()


def _detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("detail", response.text))
    except ValueError:
        return response.text


# Provider → (where to get a key, link, default model). Ollama needs no key.
PROVIDER_HELP = {
    "gemini": ("Google AI Studio", "https://aistudio.google.com/apikey", "gemini-3.1-flash-lite"),
    "groq": ("Groq Console", "https://console.groq.com/keys", "llama-3.3-70b-versatile"),
}
PROVIDERS = ["gemini", "groq", "ollama"]


def _llm_config_panel(*, current_provider: str, current_model: str, llm_ok: bool) -> None:
    """Sidebar form for supplying an API key without editing .env and restarting.

    The key goes straight to POST /config/llm, which verifies it with a real
    round-trip before accepting it. It is never stored on disk by this app.
    """
    title = "🔑 LLM connected" if llm_ok else "🔑 Connect an LLM — needed to ask questions"

    with st.sidebar.expander(title, expanded=not llm_ok):
        if message := st.session_state.pop("cfg_message", None):
            st.success(message)

        index = PROVIDERS.index(current_provider) if current_provider in PROVIDERS else 0
        provider = st.selectbox("Provider", PROVIDERS, index=index, key="cfg_provider")

        api_key = None
        if provider in PROVIDER_HELP:
            site, url, fallback_model = PROVIDER_HELP[provider]
            api_key = st.text_input(
                "API key",
                type="password",
                key=f"cfg_key_{provider}",
                placeholder="paste your key",
            )
            st.caption(f"Free key → [{site}]({url})")
        else:
            fallback_model = "qwen2.5:7b"
            st.caption("Runs locally, no key needed. Requires `ollama serve`.")

        # Prefill with what's actually configured when we're not switching provider.
        default_model = current_model if provider == current_provider else fallback_model
        model = st.text_input("Model", value=default_model, key=f"cfg_model_{provider}")

        if st.button("Connect", type="primary", key="cfg_submit", use_container_width=True):
            if provider in PROVIDER_HELP and not (api_key or "").strip():
                st.error("Enter an API key first.")
            else:
                with st.spinner("Verifying the key with a live call…"):
                    ok, error, body = api_post_quiet(
                        "/config/llm",
                        {"provider": provider, "api_key": api_key, "model": model or None},
                    )
                if ok and body:
                    st.session_state["cfg_message"] = (
                        f"Connected to {body['provider']} / {body['model']}."
                    )
                    st.rerun()
                else:
                    st.error(error)

        st.caption(
            "Held in memory for this session only — never written to disk or logged. "
            "To persist it, put it in `.env` and restart."
        )


# --- sidebar ------------------------------------------------------------

st.sidebar.title("🎫 Ticket Intelligence")
health = api_get("/health")

if health:
    # Must be a statement, not a ternary expression: Streamlit's "magic" renders
    # the value of any bare top-level expression, which would dump the returned
    # DeltaGenerator's repr into the page.
    if health["status"] == "ok":
        st.sidebar.success("All systems operational")
    else:
        st.sidebar.warning("Running degraded")

    st.sidebar.caption(f"**LLM** · {health['llm_provider']} / `{health['llm_model']}`")
    for name, component in health["components"].items():
        # Truncate: the full guidance for a missing key lives in the panel below,
        # where it is actionable, rather than filling the sidebar.
        detail = component["detail"]
        if len(detail) > 60:
            detail = detail[:57].rstrip() + "…"
        st.sidebar.write(f"{'✅' if component['ok'] else '❌'} **{name}** — {detail}")

    _llm_config_panel(
        current_provider=health["llm_provider"],
        current_model=health["llm_model"],
        llm_ok=health["components"].get("llm", {}).get("ok", False),
    )

    profile = health["dataset"]
    st.sidebar.divider()
    st.sidebar.caption("**Dataset**")
    st.sidebar.write(f"{profile['rows']} tickets · {profile['agents']} agents")
    st.sidebar.write(f"{profile['date_range']['start']} → {profile['date_range']['end']}")
    st.sidebar.info(
        f"Reference date: **{profile['as_of']}**\n\n"
        "The data is historical, so *this week* and *this month* are measured "
        "from here, not from today."
    )
    st.sidebar.caption("**Status**")
    st.sidebar.bar_chart(pd.Series(profile["status_counts"]), horizontal=True)

st.title("Support Ticket Intelligence")

ask_tab, anomaly_tab = st.tabs(["Ask a question", "Anomalies"])


# --- query tab ----------------------------------------------------------

with ask_tab:
    st.caption(
        "Questions are translated to SQL by the LLM and executed by SQLite. "
        "Every number below was computed by the database — expand **Query** to verify it."
    )

    if "question" not in st.session_state:
        st.session_state.question = ""

    examples = (api_get("/examples") or {}).get("questions", [])
    if examples:
        st.write("**Try one:**")
        for row_start in range(0, len(examples), 3):
            for col, example in zip(st.columns(3), examples[row_start : row_start + 3]):
                if col.button(example, key=example, use_container_width=True):
                    st.session_state.question = example

    question = st.text_input(
        "Your question",
        value=st.session_state.question,
        placeholder="e.g. Which category has the worst average resolution time?",
    )

    if st.button("Ask", type="primary") or (question and question != st.session_state.get("last")):
        st.session_state.last = question
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Translating to SQL and running it…"):
                result = api_post("/query", {"question": question, "include_rows": True})

            if result:
                st.success(result["answer"])

                meta = st.columns(3)
                meta[0].metric("Rows returned", result["row_count"])
                meta[1].metric("Model confidence", result["confidence"].title())
                meta[2].metric("Auto-repaired", "Yes" if result["repaired"] else "No")

                for warning in result["warnings"]:
                    st.warning(warning)

                with st.expander("Query — click to verify the answer"):
                    st.code(result["sql"], language="sql")
                    if result["explanation"]:
                        st.caption(f"**Why this query:** {result['explanation']}")

                if result["rows"]:
                    st.dataframe(pd.DataFrame(result["rows"]), use_container_width=True)
                else:
                    st.info("The query matched no rows.")


# --- anomaly tab --------------------------------------------------------

with anomaly_tab:
    st.caption(
        "Detection is deterministic — SLA rules plus IQR and z-score outlier tests, "
        "no LLM. The same data always produces the same flags."
    )

    controls = st.columns([2, 2, 3])
    severity = controls[0].selectbox("Severity", ["all", "critical", "high", "medium", "low"])
    summarize = controls[1].toggle("LLM summary", value=False, help="Narrate the findings.")

    params: dict = {"summarize": summarize}
    if severity != "all":
        params["severity"] = severity

    with st.spinner("Scanning…"):
        report = api_get("/anomalies", **params)

    if report:
        top = st.columns(4)
        top[0].metric("Anomalies", report["total"])
        top[1].metric("Tickets flagged", report["tickets_flagged"])
        top[2].metric("Critical", report["severity_counts"].get("critical", 0))
        top[3].metric("High", report["severity_counts"].get("high", 0))

        if report.get("summary"):
            st.info(report["summary"])

        if not report["anomalies"]:
            st.success("No anomalies matched this filter.")

        for anomaly in report["anomalies"]:
            icon, colour = SEVERITY_STYLE.get(anomaly["severity"], ("⚪", "#777"))
            with st.container(border=True):
                st.markdown(
                    f"{icon} **{anomaly['title']}**  \n"
                    f"<span style='color:{colour};font-size:0.8em'>"
                    f"{anomaly['severity'].upper()} · {anomaly['type']} · "
                    f"{anomaly['count']} ticket(s)</span>",
                    unsafe_allow_html=True,
                )
                st.write(anomaly["description"])
                with st.expander("Evidence and affected tickets"):
                    st.json(anomaly["evidence"])
                    ids = ", ".join(anomaly["ticket_ids"])
                    if anomaly["ticket_ids_truncated"]:
                        ids += f" … (showing 50 of {anomaly['count']})"
                    st.caption(ids)
