#!/usr/bin/env bash
# Refresh the dashboard data from the run logs and deploy dashboard/ to Cloudflare.
#
#   bash scripts/deploy_dashboard.sh             # export, then deploy
#   bash scripts/deploy_dashboard.sh --no-deploy # export only (serve dashboard/public to check it)
#
# The export refuses to write if it finds a .env value, a token, a local path or an
# email in the data, so a deploy never publishes them. Wrangler must be logged in
# (npx --yes wrangler@4 whoami).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -x "$ROOT/.venv/Scripts/python.exe" ]; then
  PYTHON="$ROOT/.venv/Scripts/python.exe"
elif [ -x "$ROOT/.venv/bin/python" ]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python}"
fi

"$PYTHON" "$ROOT/scripts/export_dashboard.py"

if [ "${1:-}" = "--no-deploy" ]; then
  echo "exported only; serve it with: $PYTHON -m http.server 8765 --directory dashboard/public"
  exit 0
fi

cd "$ROOT/dashboard"
npx --yes wrangler@4 deploy
