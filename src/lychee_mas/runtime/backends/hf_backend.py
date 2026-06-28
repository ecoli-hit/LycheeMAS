"""HF Qwen3-4B 后端 —— 生成 + latent 注入唯一发生的地方（迁移自 backends/backend_hf.py）。

这是探针阶段对「最终 vLLM client」的替身：绕开 vLLM，用 transformers 的 `inputs_embeds` 实现
latent 注入（约束 #3）。

通道需要的三个原语：
  - generate_chat(messages)                       -> 普通文本生成（NL / none 通道）
  - encode_hidden(text)                           -> 末层 hidden states（latent 的源）
  - generate_chat_with_prefix(messages, prefix)   -> 把 (1,P,H) soft prefix 拼到前面再生成

默认贪心（do_sample=False）以保证可复现；可选采样解码（do_sample=True + temperature/top_p，构造时
设随机种子复现），用于免训练 latent 软前缀这类「贪心会重复塌缩」的场景。返回值带 token/延迟记账，
供成本轴使用。

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
                 dtype: Any = None, enable_thinking: bool = False,
                 do_sample: bool = False, temperature: float = 0.7,
                 top_p: float = 0.8, seed: int = 0):
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
        # 解码参数：默认贪心；采样时按 temperature/top_p，并设种子保证 run 级可复现
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        if do_sample:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def _gen_kwargs(self, max_new_tokens: int) -> Dict[str, Any]:
        """统一生成参数：默认贪心；do_sample=True 时加 temperature/top_p（构造已设种子）。"""
        kw: Dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": self.do_sample,
                              "pad_token_id": self.tok.pad_token_id}
        if self.do_sample:
            kw["temperature"] = self.temperature
            kw["top_p"] = self.top_p
        return kw

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
            out = self.model.generate(input_ids=ids, **self._gen_kwargs(max_new_tokens))
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
                                      **self._gen_kwargs(max_new_tokens))
            dt = time.time() - t0
            # 用 inputs_embeds 时 generate 只返回新 token（不回显输入），故无需切片（约束 #3）
            return GenResult(self.tok.decode(out[0], skip_special_tokens=True),
                             P + ids.shape[1], int(out.shape[1]), dt, prefix_len=P)

    # ---- C2C 原语：模型维度（建 projector 栈用）----
    def kv_dims(self) -> tuple:
        """返回 (num_hidden_layers, num_key_value_heads, head_dim)，给 C2C projector 栈定形。"""
        c = self.model.config
        head_dim = getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)
        return c.num_hidden_layers, c.num_key_value_heads, head_dim

    # ---- C2C 原语：把 messages 编码成逐层 KV cache（source 端，融合的来源）----
    def encode_kv_cache(self, messages: List[Dict]):
        """前向一遍取每层 (K,V)（post-RoPE，cache 里存的就是它）。返回 (n_tokens, [(K,V),...])。"""
        import torch

        with torch.no_grad():
            ids = self._chat_ids(messages)
            out = self.model(input_ids=ids, use_cache=True)
            layers = [(lyr.keys.detach(), lyr.values.detach())
                      for lyr in out.past_key_values.layers]
        return ids.shape[1], layers

    # ---- C2C 原语：prefill 时逐层把 source KV 经 projector 融合进 target KV，再生成 ----
    def generate_chat_with_cache_fusion(self, messages: List[Dict], source_layers, projectors,
                                        max_new_tokens: int = 256,
                                        src_span=None, tgt_span=None) -> GenResult:
        """source_layers: encode_kv_cache 产出的逐层 (K,V)；projectors: 逐层 C2CProjector。

        src_span/tgt_span 给定则只在该跨度融合（角色条件融合，问题后缀对齐）；
        均 None 则要求整段等长、融合全序列。
        """
        import torch

        with torch.no_grad():
            ids = self._chat_ids(messages)
            cache = make_fusion_cache(source_layers, projectors, self.model.config,
                                      src_span=src_span, tgt_span=tgt_span)
            t0 = time.time()
            out = self.model.generate(input_ids=ids, past_key_values=cache,
                                      **self._gen_kwargs(max_new_tokens))
            dt = time.time() - t0
            new = out[0, ids.shape[1]:]
            return GenResult(self.tok.decode(new, skip_special_tokens=True),
                             ids.shape[1], int(new.shape[0]), dt)


def common_suffix_len(a, b) -> int:
    """两个 1D token id 序列（list/tensor）的公共后缀长度。用于角色条件融合的跨度对齐：
    source/target 的 system 角色前缀不同（不融合），共享的问题+生成提示后缀对齐（融合）。
    """
    a = list(a)
    b = list(b)
    k = 0
    while k < len(a) and k < len(b) and a[-1 - k] == b[-1 - k]:
        k += 1
    return k


def make_fusion_cache(source_layers, projectors, config, src_span=None, tgt_span=None):
    """构造 FusionCache(DynamicCache)：prefill 时用 projector 把 source 的逐层 (K,V) 融合进
    target 的 (K,V)，再交父类存储。无需 monkeypatch 注意力，生成/前向都走原生路径。
    不强制 no_grad —— 训练时调用方保留梯度（projector 可训，base 冻结）。

    跨度对齐：
      - src_span=(s0,L) / tgt_span=(t0,L)：只融合各自该区间（角色条件融合用，问题后缀对齐）。
      - 均为 None：要求 source 与 target 整段等长，融合全序列（自增强/对齐场景）。
    用 config 构造（与模型内部 `DynamicCache(config=...)` 一致，正确建出各层 cache 类型）。
    """
    from transformers import DynamicCache

    class FusionCache(DynamicCache):
        def __init__(self):
            super().__init__(config=config)
            self._src = source_layers   # list[(K,V)]，每层 (1, Hkv, Ns, Dh)
            self._proj = projectors     # 逐层 C2CProjector
            self._ss = src_span
            self._ts = tgt_span
            self.fuse = True            # 只在 prefill 融合

        def update(self, key_states, value_states, layer_idx, *args, **kwargs):
            do = (self.fuse and key_states.shape[-2] > 1
                  and layer_idx < len(self._proj) and layer_idx < len(self._src)
                  and self._src[layer_idx][0] is not None)
            if do:
                ks, vs = self._src[layer_idx]
                proj = self._proj[layer_idx]
                pdt = next(proj.parameters()).dtype
                if self._ts is None:  # 全序列融合（要求等长）
                    if ks.shape[-2] == key_states.shape[-2]:
                        kf, vf = proj((ks.to(pdt), vs.to(pdt)),
                                      (key_states.to(pdt), value_states.to(pdt)))
                        key_states = kf.to(key_states.dtype)
                        value_states = vf.to(value_states.dtype)
                else:  # 跨度融合（角色条件）
                    s0, ln = self._ss
                    t0, _ = self._ts
                    if t0 + ln <= key_states.shape[-2] and s0 + ln <= ks.shape[-2]:
                        kf, vf = proj((ks[:, :, s0:s0 + ln].to(pdt), vs[:, :, s0:s0 + ln].to(pdt)),
                                      (key_states[:, :, t0:t0 + ln].to(pdt),
                                       value_states[:, :, t0:t0 + ln].to(pdt)))
                        import torch as _t  # 用 cat 重组，避免 in-place 破坏 autograd
                        key_states = _t.cat([key_states[:, :, :t0], kf.to(key_states.dtype),
                                             key_states[:, :, t0 + ln:]], dim=2)
                        value_states = _t.cat([value_states[:, :, :t0], vf.to(value_states.dtype),
                                               value_states[:, :, t0 + ln:]], dim=2)
            return super().update(key_states, value_states, layer_idx, *args, **kwargs)

    return FusionCache()


# 旧名保留（内部调用）
_make_fusion_cache = make_fusion_cache
