#!/usr/bin/env bash
# Build and serve LycheeMAS Eval Studio.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND="$ROOT/apps/eval_studio/frontend"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8010}"
MODE="serve"
PYTHON_BIN="${LYCHEE_STUDIO_PYTHON:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo '[studio] Python was not found; set LYCHEE_STUDIO_PYTHON explicitly.' >&2
    exit 2
  fi
fi

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
if ! "$PYTHON_BIN" -c 'import fastapi, uvicorn' >/dev/null 2>&1; then
  echo '[studio] installing Python dependencies: .[studio]'
  "$PYTHON_BIN" -m pip install -e '.[studio]'
fi

# Keep the server process on the same Node contract used by frontend builds so
# Environment checks never mistake the host's legacy /usr/bin/node for the
# project runtime. Serving an already-built dist remains possible without Node.
# shellcheck source=scripts/node_runtime.sh
source "$ROOT/scripts/node_runtime.sh"
NODE_READY=true
if ! lychee_activate_node_runtime; then
  NODE_READY=false
fi

if [[ "$MODE" == "build" || ! -f "$FRONTEND/dist/index.html" ]]; then
  if [[ "$NODE_READY" != "true" ]]; then
    exit 2
  fi
  "$ROOT/scripts/build_eval_studio_frontend.sh"
fi

if [[ "$MODE" == "build" ]]; then
  echo "[studio] frontend ready at $FRONTEND/dist"
  exit 0
fi

echo "[studio] http://$HOST:$PORT"
exec "$PYTHON_BIN" scripts/serve_eval_studio.py --host "$HOST" --port "$PORT"
