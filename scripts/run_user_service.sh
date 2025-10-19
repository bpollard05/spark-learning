#!/usr/bin/env bash
set -euo pipefail

# Always run from repo service root so `services.*` is importable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR%/scripts}"
cd "$REPO_ROOT"

# Activate local venv if present
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

# Allow override but provide sensible defaults
export DATABASE_URL="${DATABASE_URL:-postgresql://user:password@localhost:5432/learning_companion}"
PORT="${PORT:-8011}"
HOST="${HOST:-0.0.0.0}"

exec uvicorn services.user_service.main:app --host "$HOST" --port "$PORT" --log-level info


