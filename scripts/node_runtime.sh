#!/usr/bin/env bash
# Resolve a Node.js runtime compatible with the Eval Studio frontend toolchain.

lychee_node_version_ok() {
  local node_bin="$1"
  local npm_bin="$(dirname "$node_bin")/npm"
  [[ -x "$node_bin" && -x "$npm_bin" ]] && "$node_bin" -e '
    const [major, minor] = process.versions.node.split(".").map(Number);
    process.exit(
      (major === 20 && minor >= 19) || (major === 22 && minor >= 12) || major > 22 ? 0 : 1
    );
  ' >/dev/null 2>&1
}

lychee_activate_node_runtime() {
  local node_bin=""
  local candidate=""

  if [[ -n "${LYCHEE_NODE_HOME:-}" ]]; then
    candidate="${LYCHEE_NODE_HOME%/}/bin/node"
    if ! lychee_node_version_ok "$candidate"; then
      echo "[node] LYCHEE_NODE_HOME does not contain Node.js ^20.19 or >=22.12: $candidate" >&2
      return 2
    fi
    node_bin="$candidate"
  elif command -v node >/dev/null 2>&1 && lychee_node_version_ok "$(command -v node)"; then
    node_bin="$(command -v node)"
  else
    for candidate in "$HOME"/.local/node-v*/bin/node; do
      if lychee_node_version_ok "$candidate"; then
        node_bin="$candidate"
        break
      fi
    done
  fi

  if [[ -z "$node_bin" ]]; then
    echo "[node] Eval Studio requires Node.js ^20.19 or >=22.12." >&2
    echo "[node] Install a compatible runtime or set LYCHEE_NODE_HOME=/path/to/node." >&2
    return 2
  fi

  export PATH="$(dirname "$node_bin"):$PATH"
  export LYCHEE_NODE_BIN="$node_bin"
  echo "[node] using $node_bin ($(node --version), npm $(npm --version))"
}
