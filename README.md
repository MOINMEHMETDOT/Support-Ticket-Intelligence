# Support Ticket Intelligence

An LLM-powered system over a 500-row customer support ticket dataset. It ingests the CSV, answers natural-language questions about it, flags anomalies, and exposes both a REST API and a Streamlit UI.

Built for the DOTMappers AI Engineer assessment.

---

## The one design decision that drives everything else

**The LLM never computes a number.**

Every question in the brief is an aggregate over structured columns — *how many*, *what's the average*, *which agent is lowest*. There are two obvious ways to answer them:

| Approach | Why not |
| --- | --- |
| Embed the rows, retrieve the relevant ones, ask the LLM to answer | Retrieval returns *some* rows, not *all* matching rows. "How many tickets are open?" becomes a guess from a sample. Wrong, and confidently so. |
| Stuff all 500 rows into the context and ask | Works at 500 rows, breaks at 50,000, and LLMs miscount long lists even when the data fits. |

So the LLM is used for the two things it is reliably good at, and nothing in between:

```
question ──▶ LLM ──▶ SQL ──▶ guard ──▶ SQLite ──▶ rows ──▶ LLM ──▶ answer
            translate                  compute            phrase
```

It translates English to SQL, SQLite computes the answer, and it phrases the result. **Every figure in every answer was produced by the database**, and the API returns the SQL alongside the answer so any claim can be checked. This is also why the system scales past this dataset unchanged — 500 rows or 5 million, the prompt is the same size.

Anomaly detection doesn't use the LLM at all. See [below](#3-anomaly-detection).

---

## Setup

Requires Python 3.10+.

```bash
git clone <your-repo-url> && cd <repo>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### Supplying an API key

Two ways, both fine:

- **From the UI** — start the app and use the **🔑 Connect an LLM** panel in the sidebar. Pick a provider, paste the key, hit Connect. It's verified with a live call before being accepted, held in process memory only, and never written to disk or logged. Lasts until restart. *Nothing else needs configuring to run this — you can skip straight to [Run](#run).*
- **From `.env`** — persists across restarts. Details per provider below.

Then pick a provider in `.env`. **Any one of these three works** — the system is provider-agnostic:

<details open>
<summary><b>Option A — Ollama (fully local, no API key, zero cost)</b></summary>

Recommended if you're evaluating this and don't want to sign up for anything.

```bash
brew install ollama          # or: curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen2.5:7b
```

```dotenv
LLM_PROVIDER=ollama
```
</details>

<details>
<summary><b>Option B — Groq free tier</b></summary>

Free key from [console.groq.com/keys](https://console.groq.com/keys).

```dotenv
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
```
</details>

<details>
<summary><b>Option C — Google Gemini free tier</b></summary>

Free key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
```

Model ids change between releases. Confirm which ones your key can use:

```bash
python scripts/list_models.py
```
</details>

### Run

```bash
./run.sh
```

API at **http://localhost:8000** (interactive docs at `/docs`), UI at **http://localhost:8501**.

Or run them separately: `./run.sh api` / `./run.sh ui`, or `uvicorn app.api.main:app --reload`.

### Verify

```bash
pytest                        # 63 tests, no network required
python scripts/smoke_test.py  # runs the brief's sample questions against the live LLM
```

---

## Architecture

```
app/
├── config.py            Environment-driven settings, resolved once
├── data/
│   ├── schema.py        Column definitions + SLA policy — single source of truth
│   └── loader.py        CSV → pandas → SQLite; read-only query execution
├── llm/
│   ├── base.py          LLMProvider ABC — the only interface the app talks to
│   ├── gemini.py  groq.py  ollama.py
│   └── factory.py       Provider selection
├── query/
│   ├── prompts.py       Schema-derived prompts + few-shot examples
│   ├── guards.py        Validation of LLM-generated SQL
│   └── engine.py        The pipeline, incl. self-repair on failed SQL
├── anomaly/detectors.py Seven deterministic detectors
├── api/main.py          FastAPI
└── ui/streamlit_app.py  Streamlit, talks to the API over HTTP
```

### Why these pieces

**SQLite over a vector store or pandas-eval.** Exact aggregates, a query language the LLM already knows well, zero infrastructure, and a natural security boundary (read-only connections). `pandas.eval` on model output would be arbitrary code execution.

**`schema.py` as the single source of truth.** The SQL prompt is *generated* from the same column definitions the detectors use. Add a column and the model learns about it automatically — the model's picture of the data cannot drift from the code's.

**Provider behind an ABC.** Swapping Gemini for Ollama is a one-line `.env` change — or a dropdown in the sidebar, since `POST /config/llm` rebuilds the provider in place. This matters practically: the API key in the author's `.env` isn't in the repo, so without runtime configuration an evaluator would have to edit a dotfile before seeing the system work at all. The key is validated with a real round-trip before being accepted, kept in memory, never logged, and redacted out of any error message — provider SDKs have a habit of echoing the key back inside a request URL. A failed reconfigure leaves the previous working provider untouched.

**UI calls the API over HTTP** rather than importing the engine. The UI exercises the exact surface any other consumer would, so a broken endpoint fails visibly instead of being bypassed.

**Startup degrades rather than crashes.** The dataset must load, but a missing LLM key only disables `/query`. `/health` reports it precisely and `/anomalies` keeps working.

---

## The three problems worth talking about

### 1. The dataset is historical, but the questions are relative

The data runs **2024-01-01 to 2024-03-30**. The brief's sample questions say *"this month"* and *"this week"*. Using `datetime.now()` would make every one of them return zero rows.

So the system has an explicit **reference date**, defaulting to the newest `created_at` in the data (`2024-03-30 18:06`) and overridable via `AS_OF_DATE` in `.env`. It's injected into the prompt as an instruction, used to derive the `age_hrs` column, and shown in the UI sidebar so nobody misreads an answer.

This is the difference between a demo and a system: the alternative is silently wrong answers to reasonable questions.

### 2. NULL means "not resolved", not "missing"

`resolution_time_hrs` and `customer_rating` are NULL for **exactly** the 173 Open and Escalated tickets, and never for a Resolved one. That's not dirty data — it's the schema encoding "this hasn't finished yet".

It's also the single easiest thing to get wrong. `AVG(customer_rating)` silently describes resolved tickets only. The prompt states the rule explicitly, the few-shot examples demonstrate it, and the model is instructed to return a `COUNT` next to every average so answers can state their own sample size — *"3.74 across the 104 rated Technical tickets"*, not a bare *"3.74"*. A test asserts the NULL pattern holds, so if a future CSV breaks the assumption, the suite says so.

### 3. LLM-generated SQL is untrusted input

Three independent layers, because any one can be reasoned around and all three are cheap:

1. **String validation** (`query/guards.py`) — single statement, must start with `SELECT`/`WITH`, keyword denylist scanned with string literals blanked out (so a ticket titled *"Request for account delete"* doesn't trip it), only the `tickets` table, `LIMIT` injected if absent.
2. **Read-only connection** — `file:tickets.db?mode=ro`.
3. **sqlite3 authorizer callback** — denies every action except `SELECT`, `READ` and `FUNCTION` at the C level.

`tests/test_guards.py` throws nine attack strings at layer 1; `tests/test_data.py` verifies layer 3 independently by asserting a `DELETE` fails and the row count is unchanged.

---

## Prompt design

Both prompts are built in `query/prompts.py` from the schema module.

**SQL generation** supplies the generated schema with per-column semantics, the enumerated legal values for `category`/`priority`/`status` (case-sensitive — a hallucinated `'critical'` returns an empty set, not an error), the reference-date rule, the NULL semantics, and the SLA table.

Output is constrained to JSON:

```json
{"sql": "...", "explanation": "...", "confidence": "high|medium|low"}
```

with an explicit escape hatch — `{"error": "..."}` — so an out-of-scope question ("what's the customer's phone number?") produces an honest refusal instead of a plausible query against a column that doesn't exist. That maps to HTTP 422.

**Six few-shot examples**, each chosen for a failure mode observed while testing:

| Example | Failure it prevents |
| --- | --- |
| Average rating by category | Averaging across unresolved tickets |
| Lowest-rated agent | Ranking without a tie-break — AGT-08 and AGT-11 both sit at 3.48 |
| Critical not resolved in 12h | Reading this as one condition when it's two (slow-but-resolved **and** still-open) |
| "This month" | Anchoring to the real-world clock instead of the reference date |
| Phone number | Inventing a column |

**Self-repair.** If generated SQL fails to execute, the engine feeds the broken query *and the database's error message* back for one retry. Most failures are a misremembered column name, which the model fixes reliably when shown the actual error. Retries are capped at one — a second failure is a real problem and should surface, not spin. The response flags `repaired: true` so this is never invisible.

**Graceful degradation.** If the narration call fails but the SQL already succeeded, the correct numbers are returned with a note, rather than a valid result being thrown away over a formatting step.

---

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Component status incl. a real LLM round-trip, plus dataset profile |
| `POST` | `/config/llm` | Set provider/key/model at runtime; validates before accepting |
| `POST` | `/query` | Natural-language question → answer + the SQL that produced it |
| `GET` | `/anomalies` | Detected anomalies; `?severity=`, `?types=`, `?summarize=` |
| `GET` | `/examples` | Sample questions (seeds the UI) |
| `GET` | `/docs` | Interactive OpenAPI docs |

```bash
curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "Which agent has the lowest average customer rating?"}'
```

```json
{
  "answer": "AGT-11 has the lowest average customer rating at 3.48 across 29 rated tickets, tied with AGT-08 (3.48 over 25 tickets).",
  "sql": "SELECT agent_id, ROUND(AVG(customer_rating), 2) AS avg_rating, COUNT(customer_rating) AS rated_tickets FROM tickets WHERE customer_rating IS NOT NULL GROUP BY agent_id ORDER BY avg_rating ASC, rated_tickets DESC LIMIT 5",
  "explanation": "Ranks agents by mean rating over rated tickets; returns the top 5 so the margin over the runner-up is visible.",
  "confidence": "high",
  "row_count": 5,
  "repaired": false,
  "rows": [{"agent_id": "AGT-11", "avg_rating": 3.48, "rated_tickets": 29}, "..."],
  "warnings": []
}
```

Error handling: `400` malformed request · `422` unanswerable question or rejected SQL · `503` LLM unavailable (with the reason and a pointer to `/anomalies`, which doesn't need one).

---

## Example queries

Verified against the dataset. The SQL and figures are the system's actual output; the prose wording comes from the model and will vary slightly between runs and providers.

| Question | Answer |
| --- | --- |
| How many tickets are currently open? | **111** |
| Which agent resolved the most tickets this month? | **AGT-01, 16 resolved** in March 2024 (AGT-12 and AGT-09 tie at 13) |
| What is the average customer rating for Technical tickets? | **3.74**, across the 104 rated Technical tickets |
| Which agent has the lowest average customer rating? | **AGT-11 and AGT-08, both 3.48** — AGT-11 over 29 tickets, AGT-08 over 25 |
| Show me all Critical tickets not resolved within 12 hours | **34** — 3 resolved slower than 12h, 31 still unresolved |
| Which category has the worst average resolution time? | **Technical, 20.59h** (General 20.28h, Billing 16.33h) |
| How many Critical tickets are unresolved? | **31** of 55 |
| What is the customer's email for TKT-001? | *Declined* — no contact columns exist. HTTP 422 |

Regenerate this table live with `python scripts/smoke_test.py`.

---

## 3. Anomaly detection

**Deterministic by design — no LLM.** An operations team acts on these flags. A flag that changes between two runs over identical data isn't actionable, and a model that invents an anomaly is worse than no detector. `test_detection_is_deterministic` enforces this.

The LLM's only role is optional: `GET /anomalies?summarize=true` asks it to *narrate* findings it did not produce.

Seven detectors, in two families:

**Rule-based**, from the stated SLA policy in `schema.py` (Critical 4h, High 12h, Medium 24h, Low 72h — our assumption, not supplied with the dataset, and cited in every finding):
- `aging_high_priority` — the brief's worked example: unresolved High/Critical over 24h
- `unresolved_past_sla` — unresolved tickets past their own priority's target

**Statistical**, from the data's own distribution:
- `slow_resolution_outlier` — beyond the IQR upper fence (Q3 + 1.5×IQR), **computed within each priority band** so a slow Low ticket isn't flagged merely because Critical tickets are fast
- `fast_response_slow_resolution` — answered in the fastest quartile, closed in the slowest decile: the signature of a dropped handoff rather than a hard problem
- `escalation_hotspot` — issue types escalating at over 2× the 12.4% baseline
- `low_satisfaction` — resolved tickets rated 1–2
- `agent_rating_outlier` — agents >1.5 SD below the peer mean, minimum 15 rated tickets so a small sample can't defame anyone

Every finding carries severity, affected ticket IDs, and an `evidence` object stating the method and thresholds — so a lead can disagree with the threshold rather than the verdict.

### Actual output on this dataset

**13 anomalies across 286 tickets** — 2 critical, 4 high, 7 medium:

```
[CRITICAL] 80 high-priority ticket(s) unresolved for over 24 hours
    ... the oldest for 2118.6h. 40 of them are already escalated.

[CRITICAL] 31 unresolved Critical ticket(s) past the 4h SLA
    The oldest, TKT-361, has been waiting 1922.2h (480.5x the target) with AGT-04.

[HIGH] 2 issue type(s) escalate at over twice the normal rate
    Baseline is 12.4%. 'Data sync not working' escalates 33.3% of the time (5 of 15),
    which points at a product or documentation gap rather than agent performance.

[HIGH] 7 High ticket(s) took abnormally long to resolve
    Typical High resolution is 6.9h (median). These 7 exceeded the outlier
    threshold of 19.4h, peaking at 119.7h.

[MEDIUM] 2 agent(s) rated well below their peers
    Peer mean 3.75 (sd 0.18). AGT-08 (3.48/25) and AGT-11 (3.48/29) fall more than
    1.5 SD below. This measures ratings, not effort — check ticket mix first.

[MEDIUM] 3 ticket(s) were answered fast but resolved very slowly
    Responded within 1.6h yet took 43.2h+ to close.
```

---

## Known limitations

**The SLA detector is blunt on this dataset.** It flags 169 of 173 unresolved tickets, because the data spans three months and no unresolved ticket is recent — the youngest is already hundreds of hours old. The detector is *correct*; it just isn't *discriminating* here. On a live feed, where most open tickets are hours old, it would be. If this dataset were the real steady state, I'd switch the primary signal to age percentile within priority band rather than an absolute threshold. `aging_high_priority` and the IQR detectors are the ones that actually separate signal from noise here.

**Two LLM calls per question** (translate, then phrase) — roughly 2–5s end to end. The second call is skippable for API consumers that only want rows; I'd add `?narrate=false` before putting this behind a UI that people use all day.

**No conversational memory.** Every question is independent, so "what about Billing?" after a Technical question won't resolve. Deliberate for a first version — carrying context needs a resolution step for pronouns and ellipsis that's a meaningful chunk of work to do correctly, and doing it badly produces answers to questions nobody asked.

**The reference date is a workaround for a historical dataset.** Correct here, but on a live feed `AS_OF_DATE` should be unset and the code should use real time. The default (newest row) does the right thing in both cases.

**Whole-CSV load at startup.** Fine at 500 rows and at a few million; beyond that the ingestion becomes a batch job into a real database. Nothing else in the design changes — the query path is already SQL.

**Read-only by construction.** No ticket updates, no write path. The guard layers assume this; adding writes means a genuine authorization model, not a longer denylist.

**Rating comparisons between agents are not performance reviews.** `agent_rating_outlier` deliberately says so in its own description. Without controlling for ticket mix — AGT-08 may simply get the hard Critical ones — the number is a prompt to investigate, not a conclusion.

---

## Testing

63 tests, no network access required (a scripted fake stands in for the LLM):

| File | Covers |
| --- | --- |
| `test_guards.py` | 9 injection/unsafe-SQL strings, fence stripping, LIMIT injection, CTEs, literals containing keywords |
| `test_data.py` | Row count, derived `age_hrs`, the NULL⇔unresolved invariant, truncation, writes blocked at the connection |
| `test_anomalies.py` | Each detector's invariants, severity ordering, determinism, filter validation |
| `test_engine.py` | Happy path, self-repair (incl. asserting the error reaches the retry prompt), unanswerable questions, unparseable output, JSON-in-prose, narration failure, input validation |
| `test_api.py` | Every endpoint, error codes, that `/anomalies` works with no LLM configured, and that runtime key config never leaks the key into an error or disables a working provider on failure |

`scripts/smoke_test.py` covers what the suite deliberately can't: a real round trip against the live model, including an out-of-scope question and a prompt-injection attempt.

---

## What I'd do next

1. **Query result caching** keyed on the normalized question — the same six questions get asked constantly, and each currently costs two LLM calls.
2. **Evaluation set** — ~50 question/expected-SQL pairs, run against each provider, so "did that prompt change help?" becomes a measurement instead of an opinion. This is the piece I'd want first before touching prompts again.
3. **Trend detection over time** — the current detectors are a snapshot. "Escalation rate in Technical doubled week over week" needs a time-series comparison the dataset's three months can just about support.
4. **Streaming responses** so the UI shows the SQL while the narration is still generating.
