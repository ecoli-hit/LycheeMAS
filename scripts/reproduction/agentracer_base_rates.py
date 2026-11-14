#!/usr/bin/env python3
"""Gold-label distribution and constant-predictor baselines for attribution data.

Reports, per subset: the share of the modal mistake_step, the number of distinct
mistake_agents, and the accuracy a constant predictor (always answering the
modal agent/step) would obtain. Use this before quoting any step_exact number:
if the modal step already covers a large share, the raw metric is dominated by
the base rate.

Usage:
    python agentracer_base_rates.py --records validation_canonical.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.records).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"rows={len(rows)}")

    by_subset: dict[str, list[dict]] = {}
    for r in rows:
        by_subset.setdefault(r.get("source_benchmark", "unknown"), []).append(r)

    total_hit = 0
    for sub in sorted(by_subset):
        rs = by_subset[sub]
        steps = [int(r["mistake_step"]) for r in rs if r.get("mistake_step") is not None]
        agents = Counter(str(r.get("mistake_agent", "")) for r in rs)
        nmsgs = [len(r.get("messages", [])) for r in rs]
        step_cnt = Counter(steps)
        modal_step, modal_step_n = step_cnt.most_common(1)[0]
        modal_agent, modal_agent_n = agents.most_common(1)[0]

        hit = sum(
            1
            for r in rs
            if str(r.get("mistake_agent", "")) == modal_agent
            and int(r["mistake_step"]) == modal_step
        )
        total_hit += hit

        print(f"\n=== {sub}  n={len(rs)} ===")
        print(f"  distinct_agents={len(agents)}  top_agent={modal_agent!r} "
              f"share={modal_agent_n / len(rs) * 100:.1f}%")
        print(f"  mistake_step: min={min(steps)} max={max(steps)} "
              f"median={statistics.median(steps)} mean={statistics.fmean(steps):.2f}")
        print(f"  modal_step={modal_step} share={modal_step_n / len(steps) * 100:.1f}%  "
              f"<- constant predictor accuracy on step")
        print(f"  messages: median={statistics.median(nmsgs)} max={max(nmsgs)}")
        print(f"  constant predictor (modal agent + modal step): "
              f"{hit}/{len(rs)} = {hit / len(rs) * 100:.1f}%")

    print(f"\n=== overall constant predictor: {total_hit}/{len(rows)} = "
          f"{total_hit / len(rows) * 100:.1f}% ===")
    print("note: the modal value is fitted on this same set, so this is an "
          "optimistic lower bound on trivial performance. Fit on the training "
          "split for a clean baseline.")


if __name__ == "__main__":
    main()
