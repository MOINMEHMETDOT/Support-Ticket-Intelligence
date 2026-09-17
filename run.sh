#!/usr/bin/env bash
# Start the API and the UI together. This is the single command in the README.
#
#   ./run.sh            API on :8000, UI on :8501
#   ./run.sh api        API only
#   ./run.sh ui         UI only (expects the API already running)

set -euo pipefail
cd "$(dirname "$0")"

API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"
MODE="${1:-all}"

if [[ ! -f .env ]]; then
  echo "No .env found — copying .env.example. Add your API key before querying."
  cp .env.example .env
fi

start_api() {
  echo "API   → http://localhost:${API_PORT}  (docs at /docs)"
  uvicorn app.api.main:app --host 0.0.0.0 --port "${API_PORT}"
}

start_ui() {
  echo "UI    → http://localhost:${UI_PORT}"
  API_URL="http://localhost:${API_PORT}" \
    streamlit run app/ui/streamlit_app.py \
    --server.port "${UI_PORT}" --server.headless true
}

case "${MODE}" in
  api) start_api ;;
  ui)  start_ui ;;
  all)
    start_api &
    API_PID=$!
    trap 'kill ${API_PID} 2>/dev/null || true' EXIT INT TERM
    # Give uvicorn a moment so the UI's first /health call succeeds.
    sleep 2
    start_ui
    ;;
  *)
    echo "Usage: ./run.sh [all|api|ui]" >&2
    exit 1
    ;;
esac
