"""AutoGenRuntime —— 把 MASGraph 映射为 AutoGen 群聊并跑任务（迁移自 teams/groupchat.py）。

发言者选择走 AutoGen 原生 SelectorGroupChat 的 selector_func：默认提供轮转 selector_func（
round-robin）；
要换选择算法，传入自定义 selector_func(messages)->下一个发言者名（返回 None 则回退到内置 LLM 选择
）。
最后一个角色用 'APPROVE: <answer>' 结束（TextMentionTermination），并由最大轮数兜底。

路由信号经共享 RoutingContext 流动——每个 agent 的 InjectionClient 都读它。答案提取按任务可定制
（见 eval/task_config.extractor_for_task）。`run` 返回统一的 `Trajectory`，并把每条消息 + 决策喂给
intercept hook（写 TraceStore / L3 抽取）。

⚠️ 这是【唯一允许 import autogen 的文件之一】（CLAUDE.md 黄金法则 1：runtime/backends/autogen_*.py
）。
autogen_agentchat 的导入全部惰性化到方法内部，保证 import 本模块（与注册）不需要 autogen。
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

from ...core.registry import REGISTRY
from ...core.types import Answer, Message, TaskQuery, Trajectory
from ..base import BaseRuntime, MASGraph


GAIA_FINAL_ANSWER_PROMPT = """\
We have completed the following task:

{task}

The above messages contain the conversation that took place to complete the task.
Read the above conversation and output a FINAL ANSWER to the question.
Use this exact template: FINAL ANSWER: [YOUR FINAL ANSWER]
Your FINAL ANSWER should be a number OR as few words as possible OR a comma
separated list of numbers and/or strings.
ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions
specified in the original question.
If you are asked for a number, express it numerically with digits, do not use
commas, and do not include units unless specified otherwise.
If you are asked for a string, do not use articles or abbreviations unless
specified otherwise. Do not output final sentence punctuation.
"""


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return safe or "case"


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_content_to_text(item) for item in content)
    if hasattr(content, "model_dump"):
        try:
            return json.dumps(content.model_dump(mode="json"), ensure_ascii=False)
        except Exception:
            return str(content)
    if hasattr(content, "__dict__"):
        try:
            return json.dumps(vars(content), ensure_ascii=False, default=str)
        except Exception:
            return str(content)
    return str(content)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _extract_final_answer(messages: list[Any], *, prefer_code_from: Optional[str] = None) -> str:
    if prefer_code_from:
        for msg in reversed(messages):
            source = getattr(msg, "source", "")
            text = _content_to_text(getattr(msg, "content", ""))
            if source == prefer_code_from and "```" in text:
                return text.strip()

    for msg in reversed(messages):
        text = _content_to_text(getattr(msg, "content", ""))
        match = re.search(r"FINAL ANSWER\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def _agent_type(node) -> str:
    return str(node.meta.get("agent_type") or node.meta.get("type") or "assistant").lower()


def _has_tool_agents(graph: MASGraph) -> bool:
    return any(_agent_type(node) != "assistant" or node.tools or node.meta.get("tools")
               for node in graph.order())


def _graph_preset(graph: MASGraph) -> str:
    return str(graph.meta.get("team_preset") or graph.meta.get("team") or "").lower()


def round_robin_selector(agent_names: Sequence[str]) -> Callable:
    """生成一个轮转的 selector_func：按 agent 顺序循环选下一个发言者（默认选择算法）。"""
    state = {"i": 0}

    def _selector(messages) -> str:
        name = agent_names[state["i"] % len(agent_names)]
        state["i"] += 1
        return name

    return _selector


def build_groupchat(agents, max_rounds: int = 2, selector_func: Optional[Callable] = None,
                    selector_client=None):
    """把已构造好的 agents 包成 AutoGen SelectorGroupChat，发言者选择由 selector_func 定义。

    - selector_func: 选下一个发言者的函数 (messages)->name；None 时用默认轮转。
    - selector_client: selector_func 返回 None（交给 LLM 选）时所需的 model_client。
    单 agent 队伍 SelectorGroupChat 不支持（要求 >=2），回退 RoundRobinGroupChat。
    """
    from autogen_agentchat.conditions import (
        MaxMessageTermination,
        TextMentionTermination,
    )
    from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat

    n_turns = len(agents) * max_rounds
    termination = TextMentionTermination("APPROVE") | MaxMessageTermination(n_turns + 1)
    if len(agents) < 2:
        return RoundRobinGroupChat(agents, termination_condition=termination, max_turns=n_turns)
    if selector_func is None:
        selector_func = round_robin_selector([a.name for a in agents])
    return SelectorGroupChat(agents, model_client=selector_client, selector_func=selector_func,
                             termination_condition=termination, max_turns=n_turns,
                             allow_repeated_speaker=True)


def _builtin_workspace_tools(workspace: Optional[Path]) -> dict[str, Callable]:
    if workspace is None:
        return {}

    def list_workspace() -> str:
        """List files available in the current workspace."""
        return "\n".join(sorted(path.name for path in workspace.iterdir()))

    def read_text_file(path: str) -> str:
        """Read a UTF-8 text file from the current workspace."""
        target = (workspace / path).resolve()
        if workspace.resolve() not in target.parents and target != workspace.resolve():
            raise ValueError("path must stay inside the workspace")
        return target.read_text(encoding="utf-8", errors="replace")

    return {"list_workspace": list_workspace, "read_text_file": read_text_file}


def _resolve_function_tools(tool_names: Sequence[Any], workspace: Optional[Path]) -> list[Any]:
    builtins = _builtin_workspace_tools(workspace)
    tools: list[Any] = []
    seen: set[str] = set()
    for item in tool_names:
        key = getattr(item, "__name__", str(item))
        if key in seen:
            continue
        seen.add(key)
        if callable(item):
            tools.append(item)
        elif isinstance(item, str) and item in builtins:
            tools.append(builtins[item])
        elif isinstance(item, str) and ":" in item:
            module_name, func_name = item.split(":", 1)
            module = __import__(module_name, fromlist=[func_name])
            tools.append(getattr(module, func_name))
    return tools


def _build_agents(graph: MASGraph, backend, ctx, max_new_tokens: int = 256,
                  model_id: str = "qwen-3-4b", workspace: Optional[Path] = None,
                  code_executor=None, web_headless: bool = True,
                  save_screenshots: bool = False) -> List[Any]:
    """把 MASGraph 的节点实例化成 AutoGen agents.

    默认仍是 AssistantAgent + InjectionClient；当 AgentSpec.meta.agent_type
    声明工具型 participant 时，构造对应的 Coder/FileSurfer/WebSurfer/Terminal。
    """
    from autogen_agentchat.agents import AssistantAgent

    from .autogen_injection_client import make_injection_client

    agents = []
    for node in graph.order():
        mid = node.model or model_id
        ctx.model_of_role[node.name] = mid  # 登记角色->模型（驱动约束 #2 的同模型对判断）
        client = make_injection_client(backend, role=node.name, ctx=ctx,
                                       max_new_tokens=max_new_tokens, model_id=mid)
        desc = node.meta.get("description") or node.name
        kind = _agent_type(node)
        if kind == "assistant":
            tools = _resolve_function_tools([*(node.tools or []),
                                             *(node.meta.get("tools") or [])], workspace)
            agents.append(AssistantAgent(node.name, model_client=client, tools=tools or None,
                                         system_message=node.system_prompt, description=desc))
        elif kind in {"coder", "magentic_one_coder"}:
            from autogen_ext.agents.magentic_one import MagenticOneCoderAgent

            agents.append(MagenticOneCoderAgent(node.name, model_client=client))
        elif kind in {"computer_terminal", "terminal", "code_executor"}:
            from autogen_agentchat.agents import CodeExecutorAgent

            if code_executor is None:
                raise ValueError("computer_terminal agent requires a code_executor")
            agents.append(CodeExecutorAgent(
                node.name,
                code_executor=code_executor,
                sources=node.meta.get("sources") or ["Coder"],
            ))
        elif kind == "file_surfer":
            from autogen_ext.agents.file_surfer import FileSurfer

            if workspace is None:
                raise ValueError("file_surfer agent requires a workspace")
            agents.append(FileSurfer(node.name, model_client=client, base_path=str(workspace)))
        elif kind == "web_surfer":
            from autogen_ext.agents.web_surfer import MultimodalWebSurfer

            if workspace is None:
                raise ValueError("web_surfer agent requires a workspace")
            agents.append(MultimodalWebSurfer(
                node.name,
                model_client=client,
                downloads_folder=str(workspace),
                debug_dir=str(workspace / "web_logs"),
                headless=web_headless,
                to_save_screenshots=save_screenshots,
            ))
        else:
            raise ValueError(f"unknown agent_type {kind!r} for node {node.name!r}")
    return agents


@REGISTRY.register("runtime", "autogen")
class AutoGenRuntime(BaseRuntime):
    """AutoGen 后端：把 MASGraph 跑成 SelectorGroupChat，产出统一 Trajectory。"""

    name = "autogen"

    def __init__(self, backend=None, ctx=None, max_new_tokens: int = 256,
                 max_rounds: int = 2, model_id: str = "qwen-3-4b",
                 selector_func: Optional[Callable] = None,
                 max_turns: Optional[int] = None, max_stalls: int = 3,
                 work_root: str | Path = "runs/lychee_tool_workspaces",
                 code_executor: str = "docker", docker_image: str = "python:3.11-slim",
                 code_timeout: int = 120, web_headless: bool = True,
                 save_screenshots: bool = False,
                 trace_model_calls: Optional[bool] = None):
        super().__init__()
        self.backend = backend  # 共享 HFBackend（None 时延迟到 run 前必须由调用方注入）
        self.ctx = ctx  # 共享 RoutingContext（携带 router/memory/task）
        self.max_new_tokens = max_new_tokens
        self.max_rounds = max_rounds
        self.model_id = model_id
        self.selector_func = selector_func
        self.max_turns = max_turns
        self.max_stalls = max_stalls
        self.work_root = Path(work_root)
        self.code_executor = code_executor
        self.docker_image = docker_image
        self.code_timeout = code_timeout
        self.web_headless = web_headless
        self.save_screenshots = save_screenshots
        self.trace_model_calls = trace_model_calls

    async def run(self, team: MASGraph, query: TaskQuery) -> Trajectory:
        if self.backend is None or self.ctx is None:
            raise ValueError("AutoGenRuntime.run 需要 backend 与 ctx（构造时或运行前注入）")

        uses_tools = _has_tool_agents(team)
        workspace: Optional[Path] = None
        task_text = query.question
        copied_files: list[str] = []
        code_executor = None
        code_executor_started = False
        agents: list[Any] = []
        t0 = time.time()

        try:
            self.ctx.log_span(
                "runtime_start",
                runtime=self.name,
                team=team.meta.get("team"),
                team_preset=team.meta.get("team_preset"),
                uses_tools=uses_tools,
                max_turns=self.max_turns,
                max_rounds=self.max_rounds,
            )
            if uses_tools:
                workspace, task_text, copied_files = self._prepare_workspace(query)
                print(
                    f"[autogen:tools] workspace={workspace} copied_files={len(copied_files)}",
                    flush=True,
                )
                self.ctx.log_span(
                    "workspace_prepared",
                    workspace=str(workspace),
                    copied_files=copied_files,
                    task_text=task_text,
                )
                if self._needs_code_executor(team):
                    code_executor = self._build_code_executor(workspace)
                    print(
                        f"[autogen:tools] starting code_executor={self.code_executor} "
                        f"image={self.docker_image}",
                        flush=True,
                    )
                    self.ctx.log_span(
                        "code_executor_start",
                        executor=self.code_executor,
                        docker_image=self.docker_image,
                        timeout_s=self.code_timeout,
                        workspace=str(workspace),
                    )
                    await code_executor.start()
                    code_executor_started = True
                    print("[autogen:tools] code_executor ready", flush=True)
                    self.ctx.log_span(
                        "code_executor_ready",
                        executor=self.code_executor,
                        docker_image=self.docker_image,
                        workspace=str(workspace),
                    )

            self.ctx.trace_model_calls = True if self.trace_model_calls is None else self.trace_model_calls
            if uses_tools:
                print(
                    f"[autogen:tools] building agents team={team.meta.get('team')} "
                    f"max_turns={self.max_turns}",
                    flush=True,
                )
            agents = _build_agents(
                team, self.backend, self.ctx,
                max_new_tokens=self.max_new_tokens, model_id=self.model_id,
                workspace=workspace, code_executor=code_executor,
                web_headless=self.web_headless, save_screenshots=self.save_screenshots,
            )
            gc = self._build_chat(team, agents, task_text)
            if uses_tools:
                print("[autogen:tools] running group chat", flush=True)
            result = await self._run_group_chat_stream(gc, task_text)
        except Exception as exc:
            from ..spans import exception_record

            self.ctx.log_span("runtime_error", **exception_record(exc))
            raise
        finally:
            for agent in agents:
                close = getattr(agent, "close", None)
                if close is not None and uses_tools:
                    await close()
            if code_executor_started and code_executor is not None:
                await code_executor.stop()
                self.ctx.log_span(
                    "code_executor_stop",
                    executor=self.code_executor,
                    docker_image=self.docker_image,
                    workspace=str(workspace) if workspace else None,
                )
        elapsed = time.time() - t0
        msgs = result.messages

        # 组装统一 Trajectory：逐条 AutoGen 消息 -> core.Message，并 emit 给 hook
        traj = Trajectory(task_id=query.id, meta={
            "runtime": self.name,
            "task": self.ctx.task,
            "team": team.meta.get("team"),
            "team_preset": team.meta.get("team_preset"),
            "workspace": str(workspace) if workspace else None,
            "copied_files": copied_files,
            "stop_reason": getattr(result, "stop_reason", None),
            "case_wall_time_s": round(elapsed, 3),
        })
        tool_calls: list[dict[str, Any]] = []
        for r, m in enumerate(msgs):
            text = _content_to_text(getattr(m, "content", ""))
            usage = getattr(m, "models_usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            source = getattr(m, "source", "?")
            msg_type = getattr(m, "type", type(m).__name__)
            msg = Message(
                sender=source,
                content=text,
                round=r,
                role="assistant",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                meta={"autogen_type": msg_type},
            )
            traj.add(msg)
            self._emit(msg)
            if self._is_tool_event(source, msg_type):
                tool_calls.append({
                    "turn": r,
                    "source": source,
                    "type": msg_type,
                    "content": _jsonable(getattr(m, "content", text)),
                    "is_error": self._message_has_tool_error(m),
                })

        # 答案提取按任务定制（eval/task_config）
        from ...eval.task_config import extractor_for_task

        final_text = _extract_final_answer(
            list(msgs),
            prefer_code_from="Coder" if team.meta.get("team_preset") == "coder_executor" else None,
        ) or extractor_for_task(self.ctx.task)(msgs)
        traj.final_answer = Answer(content=final_text, source="verifier",
                                   meta={"n_messages": len(msgs),
                                         "case_wall_time_s": round(elapsed, 3)})
        traj.candidates = [traj.final_answer]
        traj.meta["decisions"] = list(getattr(self.ctx, "decisions", []))
        traj.meta["tool_calls"] = tool_calls
        traj.meta["tool_call_count"] = len(tool_calls)
        traj.meta["tool_error_count"] = sum(1 for call in tool_calls if call.get("is_error"))
        self.ctx.log_span(
            "runtime_end",
            stop_reason=getattr(result, "stop_reason", None),
            message_count=len(msgs),
            tool_call_count=len(tool_calls),
            tool_error_count=traj.meta["tool_error_count"],
            case_wall_time_s=round(elapsed, 3),
        )
        return traj

    async def _run_group_chat_stream(self, group_chat, task_text: str):
        from autogen_agentchat.base import TaskResult

        result = None
        event_index = 0
        async for event in group_chat.run_stream(task=task_text):
            if isinstance(event, TaskResult):
                result = event
                continue
            self._log_autogen_event(event, event_index)
            event_index += 1
        if result is None:
            raise RuntimeError("AutoGen group chat ended without TaskResult")
        return result

    def _log_autogen_event(self, event: Any, event_index: int) -> None:
        source = getattr(event, "source", "?")
        msg_type = getattr(event, "type", type(event).__name__)
        content = _jsonable(getattr(event, "content", ""))
        usage = getattr(event, "models_usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        is_tool_event = self._is_tool_event(source, msg_type)
        self.ctx.log_span(
            "tool_event" if is_tool_event else "autogen_message",
            event_index=event_index,
            source=source,
            autogen_type=msg_type,
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            is_error=self._message_has_tool_error(event) if is_tool_event else False,
        )

    def _build_chat(self, team: MASGraph, agents: list[Any], task_text: str):
        preset = _graph_preset(team)
        max_turns = self.max_turns or (len(agents) * self.max_rounds)
        if preset == "magentic_one":
            from autogen_agentchat.teams import MagenticOneGroupChat
            from .autogen_injection_client import make_injection_client

            orchestrator = make_injection_client(
                self.backend,
                role="Orchestrator",
                ctx=self.ctx,
                max_new_tokens=self.max_new_tokens,
                model_id=self.model_id,
            )
            return MagenticOneGroupChat(
                agents,
                model_client=orchestrator,
                max_turns=max_turns,
                max_stalls=self.max_stalls,
                final_answer_prompt=GAIA_FINAL_ANSWER_PROMPT.format(task=task_text),
            )
        if preset == "coder_executor":
            from autogen_agentchat.conditions import TextMentionTermination
            from autogen_agentchat.teams import RoundRobinGroupChat

            return RoundRobinGroupChat(
                agents,
                max_turns=max_turns,
                termination_condition=TextMentionTermination(
                    "TERMINATE", sources=["ComputerTerminal"]),
            )
        from .autogen_injection_client import make_plain_client

        return build_groupchat(
            agents,
            max_rounds=self.max_rounds,
            selector_func=self.selector_func,
            selector_client=make_plain_client(self.backend),
        )

    def _needs_code_executor(self, team: MASGraph) -> bool:
        return any(_agent_type(node) in {"computer_terminal", "terminal", "code_executor"}
                   for node in team.order())

    def _build_code_executor(self, workspace: Path):
        if self.code_executor == "local":
            from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

            return LocalCommandLineCodeExecutor(timeout=self.code_timeout, work_dir=workspace)
        if self.code_executor == "docker":
            from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

            return DockerCommandLineCodeExecutor(
                image=self.docker_image,
                timeout=self.code_timeout,
                work_dir=workspace,
            )
        raise ValueError(f"unknown code_executor {self.code_executor!r}; choose docker or local")

    def _prepare_workspace(self, query: TaskQuery) -> tuple[Path, str, list[str]]:
        workspace = self.work_root / _safe_id(query.id)
        workspace.mkdir(parents=True, exist_ok=True)
        text = query.question
        copied: list[str] = []
        for src_text in self._referenced_paths(query):
            src = Path(src_text).expanduser()
            if not src.exists():
                continue
            dest = workspace / src.name
            if src.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(src, dest)
            else:
                shutil.copy2(src, dest)
            copied.append(str(dest))
            text = text.replace(str(src), dest.name)
        if copied:
            names = ", ".join(Path(path).name for path in copied)
            text = f"{text}\n\nThe referenced file(s) are available in the current workspace: {names}"
        return workspace, text, copied

    def _referenced_paths(self, query: TaskQuery) -> list[str]:
        haystack = "\n".join(
            part for part in [query.question, query.context or ""] if isinstance(part, str)
        )
        refs: list[str] = []
        for match in re.finditer(r"Referenced file path:\s*(.+)", haystack):
            ref = match.group(1).strip()
            if ref not in refs:
                refs.append(ref)
        return refs

    def _message_has_tool_error(self, msg: Any) -> bool:
        if hasattr(msg, "result"):
            result = getattr(msg, "result")
            return bool(getattr(result, "exit_code", 0))
        content = getattr(msg, "content", [])
        if not isinstance(content, list):
            text = _content_to_text(content).lower()
            return "error" in text or "traceback" in text
        return any(bool(getattr(item, "is_error", False)) for item in content)

    def _is_tool_event(self, source: str, msg_type: str) -> bool:
        if msg_type in {"ToolCallRequestEvent", "ToolCallExecutionEvent", "CodeExecutionEvent"}:
            return True
        return source in {"ComputerTerminal", "FileSurfer", "WebSurfer"}
