# LycheeMAS Makefile —— recipe 前缀用 '>'（CLAUDE.md §2 的 .RECIPEPREFIX 约定）。
# 环境约定：先 `conda activate LycheeMAS && source .venv/bin/activate`（uv 管理的 venv），再跑 install/install-all。
.RECIPEPREFIX := >
.PHONY: venv install install-all demo test lint format typecheck clean snapshot selfcheck

# 用 uv 创建项目 .venv（Python 3.12）；之后 `source .venv/bin/activate` 再 make install-all
venv:
> uv venv .venv --python 3.12

# 仅骨架 + 开发工具：纯离线 mock 即可跑通（无需 autogen/torch/API）。须先激活 .venv。
install:
> uv pip install -e ".[dev]"

# 全量：autogen + torch/transformers + 数据/数学评分依赖 + dev。须先激活 .venv。
# 注意：勿加 --upgrade，避免覆盖 .venv 内已装的 CUDA 版 torch / vLLM。
install-all:
> uv pip install -e ".[all]"

# 离线端到端示例（runtime=mock，零重依赖）
demo:
> PYTHONPATH=src python examples/01_static_chain_e2e.py

# 单元测试（mock runtime，无需 API key）
test:
> PYTHONPATH=src pytest -q

# 代码风格检查（只 lint 本框架，src/autogen 与旧 LycheeMAS 已排除）
lint:
> ruff check src

format:
> ruff format src/lychee_mas

typecheck:
> mypy src/lychee_mas

clean:
> find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
> rm -rf .pytest_cache .ruff_cache .mypy_cache *.egg-info src/*.egg-info

# 打印所有已注册组件（看现有可用实现）
snapshot:
> PYTHONPATH=src python -c "import lychee_mas; from lychee_mas.core.registry import REGISTRY; import pprint; pprint.pprint(REGISTRY.snapshot())"

# 验证零重依赖：import 框架后不应加载 torch/autogen 等（应输出 HEAVY LOADED: NONE）
selfcheck:
> PYTHONPATH=src python -c "import sys, lychee_mas; print('HEAVY LOADED:', [m for m in ('torch','transformers','autogen_core','autogen_agentchat','numpy','yaml','sympy','datasets') if m in sys.modules] or 'NONE')"
