#!/usr/bin/env bash
# Build Eval Studio with the project-resolved Node.js runtime.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND="$ROOT/apps/eval_studio/frontend"

# shellcheck source=scripts/node_runtime.sh
source "$ROOT/scripts/node_runtime.sh"
lychee_activate_node_runtime

echo "[studio] installing locked frontend dependencies"
(cd "$FRONTEND" && npm ci)
echo "[studio] building React frontend"
(cd "$FRONTEND" && npm run build:app)
echo "[studio] frontend ready at $FRONTEND/dist"
