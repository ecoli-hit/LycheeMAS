"""LycheeMAS Eval domain, application, and evaluation packages.

`benchmarks` owns benchmark plugins and official scoring semantics;
`application` exposes transport-independent use cases and read models;
`interfaces` contains transport adapters; `runner` executes Runs and Trials;
`evaluation` owns projections, evidence, metrics, reports, and studies. Model,
API access, pricing, deployment, team, experiment, and scheduling state live in
their matching domain packages; there is no shared ``control_plane`` bucket.

Importing this package registers benchmark plugins without reading datasets or
importing heavy scoring dependencies.
"""
from __future__ import annotations

from . import benchmarks  # noqa: F401  触发 benchmark/<task> 注册

__all__ = ["benchmarks"]
