"""评测训练好的 C2C 投影器（Phase4 PoC）—— AIME 上「角色条件融合 vs 不融合」对照。

对每道 AIME：
  - source = 分析者前缀 + 问题 → 冻结模型 KV cache。
  - target = 求解者前缀 + 问题。
  - 「融合」：求解者带着分析者的 KV（经训练好的 projector，在共享问题后缀上融合）生成。
  - 「不融合」：求解者单独生成（同 prompt，无注入）—— apples-to-apples 对照。
两者用同一抽取+评分，比较准确率，验证可学习 C2C latent 是否优于无注入（且不退化）。

跑法：CUDA_VISIBLE_DEVICES=0 python scripts/eval_c2c_aime.py \
        --ckpt runs/c2c/qwen3-4b_gsm8k/final --n 30
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)  # 复用 train 脚本里的角色前缀，保证 train/eval 一致


def extract_int(text: str) -> str:
    """从生成里抽最终整数答案：优先 '#### N'，其次 \\boxed{...}，再退最后一个整数。"""
    m = re.findall(r"####\s*(-?[0-9][0-9,]*)", text)
    if m:
        return m[-1].replace(",", "")
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return re.sub(r"[^0-9\-]", "", m[-1]) or m[-1].strip()
    m = re.findall(r"-?[0-9][0-9,]*", text)
    return m[-1].replace(",", "") if m else ""


def main() -> None:
    ap = argparse.ArgumentParser(description="AIME 上评测 C2C 融合 vs 不融合")
    ap.add_argument("--model-path", default=os.environ.get("LYCHEE_HF_MODEL",
                    "/data/mxy/Models/Qwen/Qwen3-4B"))
    ap.add_argument("--ckpt", required=True, help="projectors.pt 所在目录")
    ap.add_argument("--n", type=int, default=0, help="AIME 题数；0=全量(30)")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--gate", choices=["soft", "hard"], default="soft",
                    help="soft=σ(logit) 施加所学融合（退火未推开 logit 时更忠实）；hard=C2C 原阈值")
    ap.add_argument("--out", default=os.path.join("runs", "c2c", "eval_aime"))
    args = ap.parse_args()

    import torch
    from lychee_mas.eval.benchmarks import load as load_task
    from lychee_mas.eval.metrics import score
    from lychee_mas.layers.memory.channels.c2c_projector import build_projector_stack
    from lychee_mas.runtime.backends.hf_backend import HFBackend, common_suffix_len
    from train_c2c_projector import ANALYST_SYS, SOLVER_SYS

    be = HFBackend(args.model_path, device="cuda:0", dtype=torch.bfloat16, enable_thinking=False)

    # ---- 载入训练好的 projector ----
    blob = torch.load(os.path.join(args.ckpt, "projectors.pt"), map_location="cuda:0")
    b = blob["build"]
    projectors = build_projector_stack(b["num_hidden_layers"], b["head_dim"], b["num_kv_heads"],
                                       hidden_dim=b["hidden_dim"],
                                       intermediate_dim=b["intermediate_dim"],
                                       num_layers=b["num_layers"], dtype=torch.float32,
                                       zero_init=False).to("cuda:0")
    for proj, sd in zip(projectors, blob["state_dicts"]):
        proj.load_state_dict(sd)
    projectors.eval()
    for proj in projectors:
        proj.hard_gate = (args.gate == "hard")
    print(f"[eval] loaded {len(projectors)} projectors from {args.ckpt}; gate={args.gate}",
          flush=True)

    data = load_task("aime_2024", n=(args.n or None))
    samples, fused_ok, plain_ok = [], 0, 0
    for i, it in enumerate(data):
        prob, gold = it["question"], it["gold"]
        analyst = [{"role": "system", "content": ANALYST_SYS}, {"role": "user", "content": prob}]
        solver = [{"role": "system", "content": SOLVER_SYS}, {"role": "user", "content": prob}]
        src_ids = be._chat_ids(analyst)[0].tolist()
        tgt_ids = be._chat_ids(solver)[0].tolist()
        ln = common_suffix_len(src_ids, tgt_ids)
        src_span, tgt_span = (len(src_ids) - ln, ln), (len(tgt_ids) - ln, ln)
        _, src_layers = be.encode_kv_cache(analyst)

        g_fused = be.generate_chat_with_cache_fusion(solver, src_layers, projectors,
                                                     max_new_tokens=args.max_new_tokens,
                                                     src_span=src_span, tgt_span=tgt_span)
        g_plain = be.generate_chat(solver, max_new_tokens=args.max_new_tokens)
        a_fused, a_plain = extract_int(g_fused.text), extract_int(g_plain.text)
        c_fused = score("aime", a_fused, gold)
        c_plain = score("aime", a_plain, gold)
        fused_ok += c_fused
        plain_ok += c_plain
        samples.append({"q": prob[:200], "gold": gold,
                        "fused_ans": a_fused, "fused_correct": c_fused,
                        "plain_ans": a_plain, "plain_correct": c_plain})
        print(f"  [{i+1}/{len(data)}] fused={c_fused:.0f}({a_fused!r}) "
              f"plain={c_plain:.0f}({a_plain!r}) gold={gold!r}", flush=True)

    n = len(data)
    res = {"n": n, "fused_acc": round(fused_ok / n, 4), "plain_acc": round(plain_ok / n, 4),
           "ckpt": args.ckpt, "max_new_tokens": args.max_new_tokens}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump({"metrics": res, "samples": samples}, f, ensure_ascii=False, indent=2)
    print(f"\n[result] fused_acc={res['fused_acc']} vs plain_acc={res['plain_acc']} (n={n}) "
          f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
