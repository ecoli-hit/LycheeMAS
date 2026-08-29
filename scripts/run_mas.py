"""Execute one frozen LycheeMAS benchmark Run.

This module is the process composition root only: it parses CLI arguments,
installs signal handling, optionally starts the proxy relay, and delegates the
Run lifecycle to :mod:`lychee_mas.eval.runner.coordinator`.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from contextlib import nullcontext

# Keep ``python scripts/run_mas.py`` usable without an editable installation.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
for _pkg in ("autogen-core", "autogen-agentchat", "autogen-ext"):
    _pkg_src = os.path.join(_ROOT, "src", "autogen", "python", "packages", _pkg, "src")
    if os.path.isdir(_pkg_src) and _pkg_src not in sys.path:
        sys.path.append(_pkg_src)

from lychee_mas.eval.runner import configuration as _runner_config  # noqa: E402
from lychee_mas.eval.runner.cli import build_run_parser  # noqa: E402
from lychee_mas.eval.runner.coordinator import run_one  # noqa: E402
from lychee_mas.eval.runner.team_inspection import print_team_structure  # noqa: E402

_as_bool = _runner_config.as_bool
_get = _runner_config.get_config_value
_load_yaml = _runner_config.load_yaml


def _apply_benchmark_contract(cfg: dict, args: argparse.Namespace) -> dict:
    try:
        return _runner_config.apply_benchmark_contract(cfg, args, repo_root=_ROOT)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


async def _run_with_termination_cleanup(cfg: dict, args: argparse.Namespace) -> int:
    """Cancel the active Run on SIGTERM so runtime cleanup can finish."""

    loop = asyncio.get_running_loop()
    run_task = asyncio.create_task(run_one(cfg, args), name="lychee-run")
    received_signal: int | None = None

    def request_shutdown(signum: int) -> None:
        nonlocal received_signal
        if received_signal is not None:
            return
        received_signal = signum
        print(
            f"[control] received {signal.Signals(signum).name}; "
            "cancelling active work and cleaning runtime resources",
            flush=True,
        )
        run_task.cancel()

    installed: list[int] = []
    for signum in (signal.SIGTERM,):
        try:
            loop.add_signal_handler(signum, request_shutdown, signum)
        except (NotImplementedError, RuntimeError):
            continue
        installed.append(signum)
    try:
        await run_task
    except asyncio.CancelledError:
        if received_signal is None:
            raise
        return 128 + received_signal
    finally:
        for signum in installed:
            loop.remove_signal_handler(signum)
    return 0


def main() -> None:
    parser = build_run_parser()
    args = parser.parse_args()
    if args.list_team_structure:
        print_team_structure(team_spec_path=args.team_spec)
        return
    if not args.config:
        parser.error("--config is required unless --list-team-structure is used")

    cfg = _apply_benchmark_contract(_load_yaml(args.config), args)
    relay_context = nullcontext()
    relay_upstream = args.proxy_relay_upstream_url or _get(cfg, "network.proxy_relay_upstream_url")
    relay_requested = bool(
        args.container_proxy_url
        or (
            _as_bool(_get(cfg, "network.targets.code_executor", False), default=False)
            and _get(cfg, "network.container_proxy_url")
        )
    )
    if relay_upstream and relay_requested:
        from lychee_mas.runtime.adapters.infrastructure.proxy import ProxyRelay

        relay_context = ProxyRelay(
            relay_upstream,
            listen_host=args.proxy_relay_listen_host
            or _get(cfg, "network.docker_bridge_host", "172.17.0.1"),
            listen_port=args.proxy_relay_port
            or int(_get(cfg, "network.container_proxy_port", 17897)),
        )
    with relay_context:
        exit_code = asyncio.run(_run_with_termination_cleanup(cfg, args))
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
