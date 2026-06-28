"""vLLM 生产 client（prompt_embeds 路径）—— 占位桩（迁移自 models/vllm_prompt_embeds_client.py）。

探针/研究路径用 InjectionClient（HF inputs_embeds）。为追求吞吐，相同的路由/注入逻辑应改打一个用
`--enable-prompt-embeds` 启动的 vLLM 服务器，通过 `/v1/completions` 的 `prompt_embeds` 发送 latent
prefix。

注册为 `model_client/vllm`（桩）。统一报错文案：not wired yet (TODO)。

TODO：
  - 镜像 InjectionClient.create()：observe -> router.decide -> memory.recall
  - NL 通道：照常把文本注入 prompt
  - latent 通道：POST /v1/completions，prompt_embeds=<(P,H) 张量>，复用同一套
  RoutingContext/Manager/Router
  - 调和 token 记账（prefix 位置）与 vLLM 的 usage 上报口径
"""
from __future__ import annotations

from ...core.registry import REGISTRY


@REGISTRY.register("model_client", "vllm")
class VLLMPromptEmbedsClient:  # pragma: no cover - 占位桩
    name = "vllm"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "vllm: not wired yet (TODO); use model_client/injection (HF inputs_embeds). "
            "See FEASIBILITY.md for the live-test prerequisite.")
