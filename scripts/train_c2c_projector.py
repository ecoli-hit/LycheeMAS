"""训练 C2C 投影器（latent 通道，Phase3 PoC）—— 角色条件融合 + 答案监督。

设定（CDM 单模型，同 Qwen3-4B）：
  - source = 「分析者」角色前缀 + 问题  → 冻结模型前向得逐层 KV cache。
  - target = 「求解者」角色前缀 + 问题 + 答案（teacher forcing）。
  - 两者共享「问题+生成提示」后缀 token（角色前缀不同、不融合）；在该共享跨度上，
    用逐层 C2CProjector 把 source 的 KV 融合进 target 的 KV（make_fusion_cache 的 span 模式）。
  - 损失 = 只在答案 token 上的 CLM 交叉熵。冻结 base，只训 projector（C2C 原则）。

目标：让「拿到分析者 KV 的求解者」答得更好，把免训练 latent 的退化换成可学习融合。

跑法（需 [all] + GPU；数据走本地 gsm8k parquet）：
  CUDA_VISIBLE_DEVICES=0 python scripts/train_c2c_projector.py \
      --model-path /data/mxy/Models/Qwen/Qwen3-4B --n 2000 --steps 400 \
      --out runs/c2c/qwen3-4b_gsm8k

torch/transformers 惰性导入；本脚本被 import 不触发重依赖（黄金法则 2）。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

# 角色前缀：内容不同（⇒ 同一问题的 KV 不同，融合非平凡），长度无所谓（按公共后缀对齐问题段）
ANALYST_SYS = ("You are a math analyst. Carefully identify the quantities, relations, and the "
               "solution strategy for the problem. Think about what computations are needed.")
SOLVER_SYS = ("You are a math solver. Solve the problem step by step and end with the final "
              "numeric answer after '#### '.")


def load_gsm8k(n: int):
    import pandas as pd
    f = glob.glob(os.path.join(_ROOT, "..", "..", "Data", "raw", "gsm8k", "main",
                               "train-*.parquet"))
    f = f or glob.glob("/data/mxy/Project/CDM/Data/raw/gsm8k/main/train-*.parquet")
    df = pd.read_parquet(f[0])
    if n:
        df = df.iloc[:n]
    return list(zip(df["question"].tolist(), df["answer"].tolist()))


def main() -> None:
    ap = argparse.ArgumentParser(description="训练 C2C 投影器（角色条件融合 PoC）")
    ap.add_argument("--model-path", default=os.environ.get("LYCHEE_HF_MODEL",
                    "/data/mxy/Models/Qwen/Qwen3-4B"))
    ap.add_argument("--n", type=int, default=2000, help="gsm8k 训练子集大小")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--proj-hidden", type=int, default=512)
    ap.add_argument("--proj-intermediate", type=int, default=512)
    ap.add_argument("--proj-layers", type=int, default=3)
    ap.add_argument("--max-prompt-len", type=int, default=512)
    ap.add_argument("--max-ans-len", type=int, default=320)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=0, help=">0 则每该步数存一次中间 ckpt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join("runs", "c2c", "qwen3-4b_gsm8k"))
    args = ap.parse_args()

    import torch
    from lychee_mas.layers.memory.channels.c2c_projector import build_projector_stack
    from lychee_mas.runtime.backends.hf_backend import (
        HFBackend,
        common_suffix_len,
        make_fusion_cache,
    )
    from torch.optim import AdamW

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # ---- 冻结的 base（eval + requires_grad False）----
    be = HFBackend(args.model_path, device="cuda:0", dtype=torch.bfloat16, enable_thinking=False)
    model, tok = be.model, be.tok
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    nL, nKV, hd = be.kv_dims()
    print(f"[train] base frozen; kv_dims layers={nL} kv_heads={nKV} head_dim={hd}", flush=True)

    # ---- 可训练 projector 栈（fp32，zero_init=恒等起步）----
    projectors = build_projector_stack(nL, hd, nKV, hidden_dim=args.proj_hidden,
                                       intermediate_dim=args.proj_intermediate,
                                       num_layers=args.proj_layers,
                                       dtype=torch.float32, zero_init=True).to("cuda:0")
    projectors.train()
    for p in projectors.parameters():
        p.requires_grad_(True)
    for proj in projectors:
        proj.anneal_steps = max(1, args.steps // 2)  # 前半程退火门控温度
    nparam = sum(p.numel() for p in projectors.parameters())
    print(f"[train] projector params: {nparam/1e6:.1f}M ({nL} layers)", flush=True)

    opt = AdamW([p for p in projectors.parameters()], lr=args.lr, weight_decay=0.01)
    data = load_gsm8k(args.n)
    print(f"[train] gsm8k examples: {len(data)}; steps={args.steps} grad_accum={args.grad_accum}",
          flush=True)

    def chat_ids(sys_prompt, problem):
        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": problem}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        return tok(text, return_tensors="pt").input_ids[0].tolist()

    def encode_source(src_ids):
        with torch.no_grad():
            ids = torch.tensor([src_ids], device="cuda:0")
            out = model(input_ids=ids, use_cache=True)
            return [(lyr.keys.detach(), lyr.values.detach()) for lyr in out.past_key_values.layers]

    step, running, n_acc, t0 = 0, 0.0, 0, time.time()
    opt.zero_grad()
    di = 0
    while step < args.steps:
        problem, answer = data[di % len(data)]
        di += 1
        src_ids = chat_ids(ANALYST_SYS, problem)
        tgt_prompt = chat_ids(SOLVER_SYS, problem)
        if len(src_ids) > args.max_prompt_len or len(tgt_prompt) > args.max_prompt_len:
            continue
        ln = common_suffix_len(src_ids, tgt_prompt)  # 共享问题+生成提示后缀
        if ln < 8:
            continue
        src_span = (len(src_ids) - ln, ln)
        tgt_span = (len(tgt_prompt) - ln, ln)
        ans_ids = tok(answer, add_special_tokens=False).input_ids[:args.max_ans_len]
        ans_ids = ans_ids + [tok.eos_token_id]
        tgt_full = tgt_prompt + ans_ids
        labels = [-100] * len(tgt_prompt) + ans_ids

        src_layers = encode_source(src_ids)
        cache = make_fusion_cache(src_layers, projectors, model.config, src_span, tgt_span)
        ids = torch.tensor([tgt_full], device="cuda:0")
        lab = torch.tensor([labels], device="cuda:0")
        out = model(input_ids=ids, labels=lab, past_key_values=cache, use_cache=True)
        loss = out.loss / args.grad_accum
        loss.backward()
        running += float(out.loss.detach())
        n_acc += 1

        if n_acc % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in projectors.parameters()], 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            for proj in projectors:
                proj.update_temperature(step)
            if step % args.log_every == 0:
                avg = running / n_acc
                gate = float(torch.sigmoid(projectors[nL // 2].key_gate_logit).detach())
                print(f"[step {step}/{args.steps}] loss={avg:.4f} mid_gate={gate:.3f} "
                      f"{(time.time()-t0)/step:.2f}s/step", flush=True)
                running, n_acc = 0.0, 0
            if args.save_every and step % args.save_every == 0:
                _save(projectors, args, nL, nKV, hd, os.path.join(args.out, f"step{step}"))

    _save(projectors, args, nL, nKV, hd, os.path.join(args.out, "final"))
    print(f"[done] -> {os.path.join(args.out, 'final')}", flush=True)


def _save(projectors, args, nL, nKV, hd, out_dir):
    import torch
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "state_dicts": [p.state_dict() for p in projectors],
        "build": {"num_hidden_layers": nL, "head_dim": hd, "num_kv_heads": nKV,
                  "hidden_dim": args.proj_hidden, "intermediate_dim": args.proj_intermediate,
                  "num_layers": args.proj_layers}},
        os.path.join(out_dir, "projectors.pt"))


if __name__ == "__main__":
    main()
