#!/usr/bin/env python3
"""Dose-response over training checkpoints, evaluated on one fixed record set.

For each checkpoint it reports step/joint accuracy against the shared baseline,
the exact McNemar p-value, pairwise comparisons between consecutive checkpoints,
and the pooled effect (mean over checkpoints) with a bootstrap CI.

Usage:
    python agentracer_checkpoint_curve.py \
        --records validation_canonical.jsonl \
        --baseline baseline.predictions.jsonl \
        --checkpoints s5=s5.jsonl s10=s10.jsonl s15=s15.jsonl s20=s20.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from math import comb
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def score(records: list[dict], pred_path: Path) -> tuple[dict, set]:
    index = {(str(r.get("task_id", "")), r.get("source_benchmark", "")): r
             for r in load_rows(pred_path)}
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
        out[key] = {"parse_ok": parse_ok, "agent_ok": bool(agent_ok),
                    "step_ok": bool(step_ok), "joint_ok": bool(agent_ok and step_ok)}
        if row:
            done.add(key)
    return out, done


def rate(m: dict, keys: list, field: str) -> float:
    return sum(1 for k in keys if m[k][field]) / len(keys)


def mcnemar(a: dict, b: dict, keys: list, field: str) -> tuple[int, int, float]:
    oa = sum(1 for k in keys if a[k][field] and not b[k][field])
    ob = sum(1 for k in keys if b[k][field] and not a[k][field])
    n = oa + ob
    p = 1.0 if n == 0 else min(1.0, sum(comb(n, i) for i in range(min(oa, ob) + 1)) * 2 / 2 ** n)
    return oa, ob, p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20261112)
    args = ap.parse_args()

    records = [json.loads(l) for l in Path(args.records).read_text(encoding="utf-8").splitlines() if l.strip()]
    base, bdone = score(records, Path(args.baseline))
    pol, dones = {}, {"base": bdone}
    names = []
    for item in args.checkpoints:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"bad --checkpoints entry: {item}")
        m, d = score(records, Path(path))
        pol[name] = m
        dones[name] = d
        names.append(name)

    pool = sorted(set.intersection(*dones.values()))
    print(f"records={len(records)}  common={len(pool)}")

    print(f"\n{'policy':10s}{'step':>8s}{'joint':>8s}{'delta(step)':>13s}{'p':>9s}")
    print(f"{'base':10s}{rate(base,pool,'step_ok')*100:7.1f}%{rate(base,pool,'joint_ok')*100:7.1f}%"
          f"{'':>13s}{'':>9s}")
    b_step = rate(base, pool, "step_ok")
    for n in names:
        st = rate(pol[n], pool, "step_ok")
        jt = rate(pol[n], pool, "joint_ok")
        oa, ob, p = mcnemar(base, pol[n], pool, "step_ok")
        print(f"{n:10s}{st*100:7.1f}%{jt*100:7.1f}%{(st-b_step)*100:+12.1f}pp{p:9.3f}")

    print("\nconsecutive checkpoints:")
    for x, y in zip(names, names[1:]):
        d = (rate(pol[y], pool, "step_ok") - rate(pol[x], pool, "step_ok")) * 100
        oa, ob, p = mcnemar(pol[x], pol[y], pool, "step_ok")
        print(f"  {x} -> {y}: {d:+5.1f}pp  ({oa} vs {ob})  p={p:.3f}")

    print("\npooled effect (mean over checkpoints vs baseline):")
    rng = random.Random(args.seed)
    for label, f in [("step_exact", "step_ok"), ("joint_exact", "joint_ok")]:
        diffs = []
        for _ in range(args.bootstrap):
            draw = [pool[rng.randrange(len(pool))] for _ in pool]
            b = sum(1 for k in draw if base[k][f]) / len(draw)
            t = sum(sum(1 for n in names if pol[n][k][f]) / len(names) for k in draw) / len(draw)
            diffs.append(t - b)
        diffs.sort()
        point = sum(sum(1 for n in names if pol[n][k][f]) / len(names) for k in pool) / len(pool)
        point -= rate(base, pool, f)
        lo = diffs[int(0.025 * (args.bootstrap - 1))] * 100
        hi = diffs[int(0.975 * (args.bootstrap - 1))] * 100
        print(f"  {label:12s} {point*100:+.2f}pp  95% CI [{lo:+.2f}, {hi:+.2f}]")

    vals = [rate(pol[n], pool, "step_ok") * 100 for n in names]
    print(f"\nstep over checkpoints: {[round(v,1) for v in vals]}  spread={max(vals)-min(vals):.1f}pp")


if __name__ == "__main__":
    main()
