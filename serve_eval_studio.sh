#!/usr/bin/env bash
# Build and serve LycheeMAS Eval Studio.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND="$ROOT/apps/eval/web"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8010}"
MODE="serve"
PYTHON_BIN="${LYCHEE_EVAL_PYTHON:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    build) MODE="build"; shift ;;
    serve) MODE="serve"; shift ;;
    -H|--host) HOST="$2"; shift 2 ;;
    -p|--port) PORT="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: ./serve_eval_studio.sh [build|serve] [-H host] [-p port]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$ROOT"

if [[ "$MODE" == "build" || ! -f "$FRONTEND/dist/index.html" ]]; then
  "$ROOT/scripts/build_eval_web.sh"
fi

if [[ "$MODE" == "build" ]]; then
  echo "[studio] frontend ready at $FRONTEND/dist"
  exit 0
fi

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo '[studio] Python was not found; set LYCHEE_EVAL_PYTHON explicitly.' >&2
    exit 2
  fi
fi

if ! "$PYTHON_BIN" -c 'import fastapi, uvicorn' >/dev/null 2>&1; then
  echo '[studio] missing Python dependencies; install with: uv pip install -e ".[studio]"' >&2
  exit 2
fi

echo "[studio] http://$HOST:$PORT"
exec "$PYTHON_BIN" apps/eval/server/main.py --host "$HOST" --port "$PORT"
