"""NLMemory —— 自然语言上下文通道。

默认行为：把传入的文本（MAS 里 = 上一个 agent 的输出）原样作为要注入的 NL 记忆向下传递，
**不做任何 budget 截断**。这对应「agent 间用自然语言传递工作状态」的通信假设——NL 通道就是
把上一个 agent 说了什么，原样转给下一个 agent。

TODO（真实组件）：需要时把「原样传递」换成检索器（contriever / MemGAS 流水线），接口
(text) -> injected_text 保持不变。
"""
from __future__ import annotations

# 注入「上一个 agent 输出」时加的来源标志：接收方在 prompt 里据此指认这是上一个 agent 的输入。
# ⚠️ 本常量在此【集中定义】，是全包唯一来源；roles/templates 的 # INPUT 段引用它（消除重复字符串）。
PREV_OUTPUT_HEADER = "## Input from the previous agent"


class NLMemory:
    """NL 通道：按 strategy 选择 NL 记忆方法。

    可用 strategy：
      - "prev_output"（默认）：把上一个 agent 的输出原样向下传递（不截断），并在前面加一个
        清晰的来源标志 PREV_OUTPUT_HEADER，便于接收 agent 指认「这是上一个 agent 的输入」。
    后续新方法（检索 / 摘要 / 压缩等）在 recall() 里按 strategy 加分支即可（各自给合适的标志）。
    """

    def __init__(self, tokenizer=None, strategy: str = "prev_output"):
        self.tok = tokenizer  # 预留（如需按 token 处理）；当前不使用
        self.strategy = strategy  # 选用哪种 NL 记忆方法

    def recall(self, text: str) -> str:
        """按 strategy 产出要注入的 NL 文本。text = MAS 里上一个 agent 的输出。"""
        if self.strategy == "prev_output":
            text = text or ""
            return f"{PREV_OUTPUT_HEADER}\n{text}" if text else ""  # 加来源标志，空内容则不注入
        raise ValueError(f"unknown NL strategy {self.strategy}")
