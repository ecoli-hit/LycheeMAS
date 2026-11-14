#!/usr/bin/env python3
"""Paired evaluation of AgenTracer-style attribution predictions.

Compares a baseline prediction file against one or more adapter prediction
files on the *same* records, with per-record pairing and an exact McNemar test.

Prediction JSONL schema (one object per line):
    {"task_id": str, "source_benchmark": str,
     "prediction": {"agent": str, "step": int, ...} | null,
     "truncated": bool, "excluded": bool}

Records JSONL schema: objects carrying task_id, source_benchmark,
mistake_agent, mistake_step.

Usage:
    python agentracer_paired_eval.py \
        --records validation_canonical.jsonl \
        --baseline baseline.predictions.jsonl \
        --adapter trained.predictions.jsonl \
        --extra s5=s5.predictions.jsonl s10=s10.predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from math import comb
from pathlib import Path

FIELDS = [
    ("parse_success", "parse_ok"),
    ("agent_exact", "agent_ok"),
    ("step_exact", "step_ok"),
    ("joint_exact", "joint_ok"),
]


def load_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def score_predictions(records: list[dict], pred_path: Path) -> tuple[dict, set]:
    """Return {(task_id, subset): row} plus the set of keys actually predicted."""
    rows = [json.loads(l) for l in pred_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    index = {(str(r.get("task_id", "")), r.get("source_benchmark", "")): r for r in rows}
    out, done = {}, set()
    for rec in records:
        key = (str(rec["task_id"]), rec.get("source_benchmark", ""))
        row = index.get(key, {})
        pred = row.get("prediction")
        pred_step = pred.get("step") if isinstance(pred, dict) else None
        try:
            pred_step = int(pred_step) if pred_step is not None else None
        except (TypeError, ValueError):
            pred_step = None
        gold_step = rec.get("mistake_step")
        try:
            gold_step = int(gold_step) if gold_step is not None else None
        except (TypeError, ValueError):
            gold_step = None
        parse_ok = pred_step is not None
        agent_ok = parse_ok and str(pred.get("agent", "")) == str(rec.get("mistake_agent", ""))
        step_ok = parse_ok and pred_step == gold_step
        out[key] = {
            "parse_ok": parse_ok,
            "agent_ok": bool(agent_ok),
            "step_ok": bool(step_ok),
            "joint_ok": bool(agent_ok and step_ok),
            "truncated": bool(row.get("truncated")),
        }
        if row:
            done.add(key)
    return out, done


def rate(m: dict, keys: list, field: str) -> float:
    return sum(1 for k in keys if m[k][field]) / len(keys)


def mcnemar(a: dict, b: dict, keys: list, field: str) -> tuple[int, int, float]:
    only_a = sum(1 for k in keys if a[k][field] and not b[k][field])
    only_b = sum(1 for k in keys if b[k][field] and not a[k][field])
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    p = min(1.0, sum(comb(n, i) for i in range(min(only_a, only_b) + 1)) * 2 / 2 ** n)
    return only_a, only_b, p


def bootstrap_ci(a: dict, b: dict, keys: list, field: str, samples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        draw = [keys[rng.randrange(len(keys))] for _ in keys]
        deltas.append(rate(b, draw, field) - rate(a, draw, field))
    deltas.sort()
    return deltas[int(0.025 * (samples - 1))] * 100, deltas[int(0.975 * (samples - 1))] * 100


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="name=path pairs")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261111)
    args = ap.parse_args()

    records = load_records(Path(args.records))
    base, base_done = score_predictions(records, Path(args.baseline))
    adapter, adapter_done = score_predictions(records, Path(args.adapter))

    keys = sorted(base_done & adapter_done)
    print(f"records={len(records)}  baseline predicted={len(base_done)}  "
          f"adapter predicted={len(adapter_done)}  paired={len(keys)}")

    print("\n=== main (paired) ===")
    print(f"{'metric':16s}{'baseline':>10s}{'adapter':>10s}{'delta':>10s}{'95% CI':>20s}")
    for name, f in FIELDS:
        b, t = rate(base, keys, f), rate(adapter, keys, f)
        lo, hi = bootstrap_ci(base, adapter, keys, f, args.bootstrap, args.seed)
        print(f"{name:16s}{b*100:9.1f}%{t*100:9.1f}%{(t-b)*100:+9.1f}pp"
              f"   [{lo:+.2f}, {hi:+.2f}]")
    for name, f in [("step", "step_ok"), ("joint", "joint_ok")]:
        oa, ob, p = mcnemar(base, adapter, keys, f)
        print(f"  {name:5s} paired: baseline-only={oa} adapter-only={ob}  McNemar p={p:.4g}")

    print("\n=== per subset (step_exact) ===")
    for sub in sorted({k[1] for k in keys}):
        ks = [k for k in keys if k[1] == sub]
        b, t = rate(base, ks, "step_ok"), rate(adapter, ks, "step_ok")
        oa, ob, p = mcnemar(base, adapter, ks, "step_ok")
        print(f"  {sub:22s} n={len(ks):3d}  {b*100:5.1f}% -> {t*100:5.1f}%  "
              f"({(t-b)*100:+5.1f}pp)  p={p:.3g}")

    if args.extra:
        print("\n=== extra checkpoints (paired against baseline) ===")
        print(f"{'policy':12s}{'step':>8s}{'joint':>8s}{'':>16s}{'n':>6s}")
        for item in args.extra:
            name, _, path = item.partition("=")
            if not path:
                continue
            m, done = score_predictions(records, Path(path))
            ks = sorted(base_done & done)
            if not ks:
                print(f"  {name}: no overlap")
                continue
            print(f"{name:12s}{rate(m, ks, 'step_ok')*100:7.1f}%{rate(m, ks, 'joint_ok')*100:7.1f}%"
                  f"{'':>16s}{len(ks):6d}")


if __name__ == "__main__":
    main()
