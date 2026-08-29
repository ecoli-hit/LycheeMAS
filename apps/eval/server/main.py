#!/usr/bin/env python3
"""Start the LycheeMAS Eval HTTP server and optional built Web client."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve LycheeMAS Eval Studio")
    parser.add_argument("--host", default=os.environ.get("LYCHEE_STUDIO_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("LYCHEE_STUDIO_PORT", "8010"))
    )
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:
        message = 'Eval Studio dependencies missing; install with: pip install -e ".[studio]"'
        raise SystemExit(message) from exc

    uvicorn.run(
        "apps.eval.server.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(ROOT / "src"), str(ROOT / "apps/eval/server")]
        if args.reload
        else None,
    )


if __name__ == "__main__":
    main()
