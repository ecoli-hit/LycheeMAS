"""Command-line contract for one benchmark Run."""

from __future__ import annotations

import argparse


def build_run_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="LycheeMAS 真实 MAS 推理驱动（新框架版）")
    ap.add_argument("--config", default=None, help="YAML 配置（backend/memory/router/run/eval）")
    ap.add_argument(
        "--list-team-structure",
        action="store_true",
        help="检查 --team-spec 的 Node、Relation 和三框架绑定；不加载模型/数据",
    )
    ap.add_argument(
        "--runtime",
        default=None,
        choices=("autogen", "langgraph", "crewai"),
        help="覆盖 runtime.framework；TeamSpec 会编译为对应框架的运行对象",
    )
    ap.add_argument("--task", default=None, help="覆盖 run.task")
    ap.add_argument("--method", default=None, help="none|nl_only|latent_only|both")
    ap.add_argument("--backend", default=None, choices=("hf", "api"), help="覆盖 backend.provider")
    ap.add_argument(
        "--team-spec",
        dest="team_spec",
        default=None,
        help="框架无关 TeamSpec JSON；所有 Eval Run 均必须显式提供",
    )
    ap.add_argument(
        "--deployment-config",
        dest="deployment_config",
        default=None,
        help="Eval Studio deployment JSON；同一 deployment ID 在所有角色间复用 backend",
    )
    ap.add_argument("--n", default=None, help="样本数；all/0=全量")
    ap.add_argument(
        "--start-index",
        dest="start_index",
        type=int,
        default=None,
        help="从 loader 返回列表的第几个样本开始跑，0-based",
    )
    ap.add_argument(
        "--case-selection",
        choices=("head", "uniform", "stratified"),
        default=None,
        help="Case 选择方式：开头、全数据均匀抽取、或按字段分层抽取",
    )
    ap.add_argument(
        "--case-strata-field",
        default=None,
        help="stratified 时使用的 metadata/record 字段；留空自动检测",
    )
    ap.add_argument("--P", type=int, default=None, help="latent prefix 长度")
    ap.add_argument(
        "--c2c-ckpt",
        dest="c2c_ckpt",
        default=None,
        help="latent_strategy=c2c 时的 projector 栈 ckpt 路径（覆盖 memory.c2c_ckpt）",
    )
    ap.add_argument(
        "--trials-per-case",
        dest="trials_per_case",
        type=int,
        default=None,
        help="每个 Case 的独立 Trial 数 K（pass@K；K>1 建议启用采样解码）",
    )
    ap.add_argument(
        "--do-sample",
        dest="do_sample",
        action="store_true",
        default=None,
        help="启用采样解码；pass@K 通常应开启",
    )
    ap.add_argument(
        "--no-do-sample", dest="do_sample", action="store_false", help="强制使用确定性解码"
    )
    ap.add_argument("--temperature", type=float, default=None, help="覆盖 backend.temperature")
    ap.add_argument("--top-p", dest="top_p", type=float, default=None, help="覆盖 backend.top_p")
    ap.add_argument("--top-k", dest="top_k", type=int, default=None, help="覆盖 backend.top_k")
    ap.add_argument("--min-p", dest="min_p", type=float, default=None, help="覆盖 backend.min_p")
    ap.add_argument(
        "--presence-penalty",
        dest="presence_penalty",
        type=float,
        default=None,
        help="覆盖 backend.presence_penalty",
    )
    ap.add_argument(
        "--repetition-penalty",
        dest="repetition_penalty",
        type=float,
        default=None,
        help="覆盖 backend.repetition_penalty",
    )
    ap.add_argument("--seed", type=int, default=None, help="覆盖 backend.seed")
    ap.add_argument("--model-path", dest="model_path", default=None)
    ap.add_argument("--model-tag", dest="model_tag", default=None)
    ap.add_argument(
        "--api-model", dest="api_model", default=None, help="OpenAI-compatible model name"
    )
    ap.add_argument("--api-base-url", dest="api_base_url", default=None)
    ap.add_argument("--api-key-env", dest="api_key_env", default=None)
    ap.add_argument("--device", default=None, help="覆盖 backend.device（本地 HF 路径）")
    ap.add_argument(
        "--max-rounds", dest="max_rounds", type=int, default=None, help="覆盖 run.max_rounds"
    )
    ap.add_argument(
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=None,
        help="覆盖 run.max_new_tokens（每次模型调用最多生成多少 token）",
    )
    ap.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help="单次模型调用允许保留的最大输入 token 数",
    )
    ap.add_argument(
        "--min-output-reserve-tokens",
        type=int,
        default=None,
        help="裁剪输入时为思考与最终答案合计保留的最小输出容量",
    )
    ap.add_argument(
        "--min-thinking-reserve-tokens",
        type=int,
        default=None,
        help="思考模式开启时希望保留的最小 reasoning 容量",
    )
    ap.add_argument(
        "--max-thinking-budget-tokens",
        type=int,
        default=None,
        help="provider 支持时对 reasoning token 的独立硬上限",
    )
    ap.add_argument(
        "--min-final-reserve-tokens",
        type=int,
        default=None,
        help="从本次输出预算中为最终回答保留的最小容量",
    )
    ap.add_argument(
        "--safety-margin-tokens",
        type=int,
        default=None,
        help="模型上下文窗口末端不参与输入或输出分配的安全边距",
    )
    ap.add_argument(
        "--max-turns",
        dest="max_turns",
        type=int,
        default=None,
        help="覆盖 runtime.max_turns（AutoGen group chat 最大发言上限）",
    )
    ap.add_argument(
        "--max-model-calls-per-case",
        dest="max_model_calls_per_case",
        default=None,
        help="每个 case 允许的真实模型调用数；正整数或 unlimited",
    )
    ap.add_argument(
        "--max-case-wall-time-s",
        dest="max_case_wall_time_s",
        default=None,
        help="每个 Trial 的墙钟时间上限（秒）；正数或 unlimited",
    )
    ap.add_argument(
        "--code-executor",
        dest="code_executor",
        default=None,
        choices=("docker", "local"),
        help="需要执行代码时使用的执行器",
    )
    ap.add_argument(
        "--docker-image",
        dest="docker_image",
        default=None,
        help="覆盖 runtime.docker_image（docker executor）",
    )
    ap.add_argument(
        "--code-timeout",
        dest="code_timeout",
        type=int,
        default=None,
        help="覆盖 runtime.code_timeout",
    )
    ap.add_argument(
        "--work-root", dest="work_root", default=None, help="每个 case 的工具/附件 workspace 根目录"
    )
    ap.add_argument(
        "--web-proxy-url",
        default=None,
        help="WebSurfer/Playwright 使用的显式 HTTP(S) 代理地址",
    )
    ap.add_argument(
        "--container-proxy-url",
        default=None,
        help="代码执行容器内可访问的代理地址，例如 http://host.docker.internal:17897",
    )
    ap.add_argument(
        "--proxy-no-proxy",
        default=None,
        help="容器和显式网络客户端绕过代理的主机列表",
    )
    ap.add_argument(
        "--proxy-relay-upstream-url",
        default=None,
        help="需要为 Docker 建立网桥转发时的宿主机上游代理",
    )
    ap.add_argument(
        "--proxy-relay-listen-host",
        default=None,
        help="Docker 网桥上的 relay 监听地址",
    )
    ap.add_argument(
        "--proxy-relay-port",
        type=int,
        default=None,
        help="Docker 网桥上的 relay 监听端口",
    )
    ap.add_argument(
        "--trace-model-calls",
        dest="trace_model_calls",
        action="store_true",
        default=None,
        help="打印每次 LLM 调用的 start/done 摘要；默认开启",
    )
    ap.add_argument(
        "--no-trace-model-calls",
        dest="trace_model_calls",
        action="store_false",
        help="关闭每次 LLM 调用的 start/done 摘要",
    )
    ap.add_argument(
        "--collect-vllm-metrics",
        dest="collect_vllm_metrics",
        action="store_true",
        default=None,
        help="定时采集 vLLM /metrics，并写入独立 JSONL 时间序列",
    )
    ap.add_argument(
        "--no-collect-vllm-metrics",
        dest="collect_vllm_metrics",
        action="store_false",
        help="关闭 vLLM 服务级指标采集",
    )
    ap.add_argument(
        "--vllm-metrics-interval-s",
        type=float,
        default=None,
        help="vLLM /metrics 采样间隔（秒，默认 5）",
    )
    ap.add_argument(
        "--vllm-metrics-url",
        default=None,
        help="无 DeploymentConfig 时显式指定 vLLM Prometheus endpoint",
    )
    ap.add_argument(
        "--event-log-max-events-per-file",
        type=int,
        default=None,
        help="每个 run_events JSONL 分片最多包含的 Event 数，默认 10000",
    )
    ap.add_argument(
        "--event-log-max-mib-per-file",
        type=float,
        default=None,
        help="每个 run_events JSONL 分片的近似大小上限 MiB，默认 64",
    )
    ap.add_argument(
        "--runs-root",
        dest="runs_root",
        default=None,
        help="benchmark run result root",
    )
    ap.add_argument(
        "--run-dir",
        dest="run_dir",
        default=None,
        help=(
            "exact directory for this run; overrides the automatic "
            "<runs-root>/<model>/<team>/<method>/<task> path"
        ),
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume this run directory: append events/run_events.jsonl, keep successful "
            "trial_records, retry failed/missing cases"
        ),
    )
    ap.add_argument(
        "--segment-case-limit",
        default=None,
        help=(
            "本执行段最多选择多少个尚未完成的不同 Case；正整数或 unlimited。"
            "它不改变 ExperimentSpec 的完整 Case 范围"
        ),
    )
    ap.add_argument(
        "--on-trial-error",
        dest="on_trial_error",
        choices=("continue", "fail-fast"),
        default=None,
        help="Trial 的全部 Attempt 失败后继续后续 Trial（默认）或立即终止",
    )
    ap.add_argument(
        "--max-attempts-per-trial",
        dest="max_attempts_per_trial",
        type=int,
        default=None,
        help="每个 Trial 最多允许的 Attempt 总数，默认 1",
    )
    ap.add_argument(
        "--trial-concurrency",
        type=int,
        default=None,
        help="同一进程并发执行的 Trial 数；一个 Trial 对应一个 (case, sample index)",
    )
    ap.add_argument("--concurrency-mode", choices=("fixed", "auto"), default=None)
    ap.add_argument("--concurrency-initial", type=int, default=None)
    ap.add_argument("--concurrency-minimum", type=int, default=None)
    ap.add_argument("--concurrency-maximum", type=int, default=None)
    ap.add_argument("--concurrency-increase-step", type=int, default=None)
    ap.add_argument("--concurrency-decrease-factor", type=float, default=None)
    ap.add_argument("--concurrency-control-window-trials", type=int, default=None)
    return ap
