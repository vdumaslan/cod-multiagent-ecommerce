#!/usr/bin/env bash
# Start copilot-v2 FastAPI with specialist caches loaded from BigQuery at startup.
#
# Prerequisite (one-time per machine, interactive — opens a browser):
#   gcloud auth application-default login
#
# Your Google account must have BigQuery access to the team's dataset.
# Tables must already exist: pricing_cache, sentiment_cache, inventory_cache
# (rows for snapshot_id used by the app, default 38710839ca6e1009).
#
# Local FAISS index is still required under:
#   copilot-v2/artifacts/indexes/38710839ca6e1009/dense/intfloat_e5-large-v2/
#
# Usage (from anywhere):
#   export GCP_PROJECT_ID="your-teammates-project-id"
#   bash copilot-v2/scripts/run_api_bigquery_adc.sh
#
# Optional overrides:
#   BIGQUERY_DATASET (default copilot_v2)
#   BIGQUERY_LOCATION (default US)
#   COPILOT_BIGQUERY_FALLBACK_TO_LOCAL (default 1)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Prefer repo-root venv so google-cloud-bigquery is available (system python is often PEP-668 / no BQ lib).
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
VENV_PY="${REPO_ROOT}/.venv-copilot-v2/bin/python3"
if [[ -x "$VENV_PY" ]]; then
  PYTHON="$VENV_PY"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python3" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python3"
elif [[ -n "${COPILOT_PYTHON:-}" ]]; then
  PYTHON="$COPILOT_PYTHON"
else
  PYTHON="python3"
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "Install Google Cloud SDK and ensure 'gcloud' is on PATH." >&2
  exit 1
fi

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID to your GCP project id (same as teammates). Example: export GCP_PROJECT_ID=my-project-123}"

export COPILOT_ARTIFACTS_ROOT="${COPILOT_ARTIFACTS_ROOT:-$ROOT/artifacts}"
export PYTHONPATH="${PYTHONPATH:-$ROOT}"
export COPILOT_DATA_BACKEND="${COPILOT_DATA_BACKEND:-bigquery}"
export BIGQUERY_DATASET="${BIGQUERY_DATASET:-copilot_v2}"
export BIGQUERY_LOCATION="${BIGQUERY_LOCATION:-US}"
export COPILOT_BIGQUERY_FALLBACK_TO_LOCAL="${COPILOT_BIGQUERY_FALLBACK_TO_LOCAL:-1}"

echo "Starting API with BigQuery backend (ADC)..."
echo "  Python: $PYTHON"
echo "  GCP_PROJECT_ID=$GCP_PROJECT_ID"
echo "  BIGQUERY_DATASET=$BIGQUERY_DATASET  BIGQUERY_LOCATION=$BIGQUERY_LOCATION"
echo "  COPILOT_BIGQUERY_FALLBACK_TO_LOCAL=$COPILOT_BIGQUERY_FALLBACK_TO_LOCAL"
echo "Check /health for pricing_cache_source / sentiment / inventory (expect 'bigquery')."
exec "$PYTHON" -m uvicorn app.api.app:app --host 0.0.0.0 --port 8000 --reload
