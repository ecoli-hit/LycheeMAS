"""HF Qwen3-4B 后端 —— 生成 + latent 注入唯一发生的地方（迁移自 backends/backend_hf.py）。

这是探针阶段对「最终 vLLM client」的替身：绕开 vLLM，用 transformers 的 `inputs_embeds` 实现
latent 注入（约束 #3）。

通道需要的三个原语：
  - generate_chat(messages)                       -> 普通文本生成（NL / none 通道）
  - encode_hidden(text)                           -> 末层 hidden states（latent 的源）
  - generate_chat_with_prefix(messages, prefix)   -> 把 (1,P,H) soft prefix 拼到前面再生成

所有生成都是贪心（do_sample=False）以保证可复现。返回值带 token/延迟记账，供成本轴使用。

⚠️ 改动（CLAUDE.md 迁移要点）：
  - torch / transformers 惰性导入（仅在构造/方法内 import），保证无这些库时本模块可被 import。
  - 删除硬编码 DEFAULT_MODEL 绝对路径，改为构造参数 / 环境变量 `LYCHEE_HF_MODEL`。
  - hidden_size 硬约束（2560）保留为运行期 assert。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# 模型路径来源：构造参数 > 环境变量 LYCHEE_HF_MODEL（不再硬编码任何机器绝对路径）。
ENV_MODEL_KEY = "LYCHEE_HF_MODEL"
EXPECTED_HIDDEN = 2560  # 约束 #3：latent prefix 的 hidden 维必须等于此值（Qwen3-4B）


@dataclass
class GenResult:
    text: str
    n_prompt_pos: int  # 喂给应答者的输入位置数（成本轴；含 latent prefix 的 P）
    n_gen_tokens: int  # 生成的新 token 数
    latency_s: float  # 本次生成耗时（秒）
    prefix_len: int = 0  # n_prompt_pos 里属于 latent prefix 的位置数 P（无 latent 则 0）


class HFBackend:
    def __init__(self, model_name: Optional[str] = None, device: str = "cuda:0",
                 dtype: Any = None, enable_thinking: bool = False):
        # torch/transformers 在此惰性导入（无 GPU/无库的离线环境不应触发本类构造）
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = model_name or os.environ.get(ENV_MODEL_KEY)
        if not model_name:
            raise ValueError(
                f"HFBackend 需要模型路径：传入 model_name 或设置环境变量 {ENV_MODEL_KEY}")
        if dtype is None:
            dtype = torch.bfloat16
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.enable_thinking = enable_thinking  # Qwen3 的 think 模式；探针默认关，避免冗长思考
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token_id is None:
            # 无 pad 时用 eos 兜底，供 attention_mask/批处理
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype).to(self.device).eval()  # eval 模式（无 dropout，确定性）
        self.H = self.model.config.hidden_size
        assert self.H == EXPECTED_HIDDEN, \
            f"expected hidden {EXPECTED_HIDDEN}, got {self.H}"  # 约束 #3
        # 输入嵌入层；latent 拼接与 soft_token 都要用它
        self.embed = self.model.get_input_embeddings()

    # ---- 构造 prompt 的 token ids ----
    def _chat_ids(self, messages: List[Dict]):
        # 套 Qwen chat 模板、加生成提示符，再分词成 (1,T) 的 ids
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking)
        return self.tok(text, return_tensors="pt").input_ids.to(self.device)

    # ---- 原语 1：普通文本生成（NL / none 走这里）----
    def generate_chat(self, messages: List[Dict], max_new_tokens: int = 256) -> GenResult:
        import torch

        with torch.no_grad():
            ids = self._chat_ids(messages)
            t0 = time.time()
            out = self.model.generate(input_ids=ids, max_new_tokens=max_new_tokens,
                                      do_sample=False, pad_token_id=self.tok.pad_token_id)
            dt = time.time() - t0
            new = out[0, ids.shape[1]:]  # 用 input_ids 时 generate 回显输入，需切掉前缀只取新 token
            return GenResult(self.tok.decode(new, skip_special_tokens=True),
                             ids.shape[1], int(new.shape[0]), dt)

    # ---- 原语 2：编码文本 -> 末层 hidden states（latent 的源）----
    def encode_hidden(self, text: str, max_tokens: int = 4096):
        import torch

        with torch.no_grad():
            ids = self.tok(text, return_tensors="pt", truncation=True,
                           max_length=max_tokens).input_ids.to(self.device)  # 过长截断
            out = self.model(input_ids=ids, output_hidden_states=True)  # 前向一遍取隐藏态
            return out.hidden_states[-1]  # (1, T, H) 末层

    # ---- 原语 3：把 (1,P,H) soft prefix 拼到 token 嵌入前再生成（latent 注入）----
    def generate_chat_with_prefix(self, messages: List[Dict], prefix,
                                  max_new_tokens: int = 256) -> GenResult:
        import torch

        # 形状/hidden 维校验（约束 #3）
        assert prefix.dim() == 3 and prefix.shape[-1] == self.H, \
            f"prefix must be (1,P,{self.H}), got {tuple(prefix.shape)}"
        with torch.no_grad():
            ids = self._chat_ids(messages)
            tok_embeds = self.embed(ids)  # (1, T, H) 先把 ids 过嵌入层
            prefix = prefix.to(tok_embeds.dtype).to(self.device)  # 对齐 dtype/device
            P = prefix.shape[1]
            # (1, P+T, H) latent 在序列维拼到最前 —— 注入点
            inp = torch.cat([prefix, tok_embeds], dim=1)
            # 全 1 注意力
            attn = torch.ones(1, P + ids.shape[1], device=self.device, dtype=torch.long)
            t0 = time.time()
            out = self.model.generate(inputs_embeds=inp, attention_mask=attn,
                                      max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
            dt = time.time() - t0
            # 用 inputs_embeds 时 generate 只返回新 token（不回显输入），故无需切片（约束 #3）
            return GenResult(self.tok.decode(out[0], skip_special_tokens=True),
                             P + ids.shape[1], int(out.shape[1]), dt, prefix_len=P)
