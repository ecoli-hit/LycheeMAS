"""Framework-neutral code executor construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lychee_mas.runtime.adapters.infrastructure.docker import configured_docker_executor_type


class CodeExecutorFactory:
    def __init__(
        self,
        *,
        executor: str,
        docker_image: str,
        timeout_s: int,
        container_proxy_url: str | None,
        proxy_no_proxy: str,
    ) -> None:
        self.executor = executor
        self.docker_image = docker_image
        self.timeout_s = int(timeout_s)
        self.container_proxy_url = container_proxy_url
        self.proxy_no_proxy = proxy_no_proxy

    def build(self, workspace: Path) -> Any:
        if self.executor == "local":
            from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

            return LocalCommandLineCodeExecutor(timeout=self.timeout_s, work_dir=workspace)
        if self.executor != "docker":
            raise ValueError("code_executor must be local or docker")
        from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

        executor_type = configured_docker_executor_type(
            DockerCommandLineCodeExecutor,
            proxy_url=self.container_proxy_url,
            no_proxy=self.proxy_no_proxy,
        )
        kwargs: dict[str, Any] = {}
        if self.container_proxy_url:
            kwargs["extra_hosts"] = {"host.docker.internal": "host-gateway"}
        return executor_type(
            image=self.docker_image,
            timeout=self.timeout_s,
            work_dir=workspace,
            **kwargs,
        )
