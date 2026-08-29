from __future__ import annotations

import json
import subprocess
from collections import namedtuple
from pathlib import Path

from lychee_mas.eval.infrastructure.docker_image_cache import (
    GIB,
    DockerImageCachePolicy,
    ProjectDockerImageCache,
)


class FakeDocker:
    def __init__(self, images: list[str]) -> None:
        self.images = set(images)
        self.containers: dict[str, list[str]] = {}
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            image = command[3]
            if image not in self.images:
                return subprocess.CompletedProcess(command, 1, "", "missing")
            payload = [{"Id": f"sha256:{image}", "Created": "2026-01-01T00:00:00Z"}]
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        if command[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(command, 0, "/fake-docker\n", "")
        if command[:3] == ["docker", "ps", "-aq"]:
            image = command[-1].split("=", 1)[1]
            return subprocess.CompletedProcess(
                command, 0, "\n".join(self.containers.get(image, [])), ""
            )
        if command[:3] == ["docker", "image", "rm"]:
            image = command[3]
            if image not in self.images:
                return subprocess.CompletedProcess(command, 1, "", "missing")
            self.images.remove(image)
            return subprocess.CompletedProcess(command, 0, image, "")
        raise AssertionError(f"unexpected Docker command: {command}")


def _fixed_disk_usage(monkeypatch, *, free_gib: float = 100.0) -> None:
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        "lychee_mas.eval.infrastructure.docker_image_cache.shutil.disk_usage",
        lambda _path: usage(200 * GIB, (200 - free_gib) * GIB, free_gib * GIB),
    )


def test_bounded_cache_removes_oldest_project_images(tmp_path: Path, monkeypatch) -> None:
    _fixed_disk_usage(monkeypatch)
    docker = FakeDocker(["image-a", "image-b", "image-c"])
    cache = ProjectDockerImageCache(
        manifest_path=tmp_path / "cache.json",
        policy=DockerImageCachePolicy(max_images=1, min_free_gib=0, target_free_gib=0),
        command_runner=docker,
    )

    report = cache.adopt(
        {
            "image-a": {"last_used_at_utc": "2026-01-01T00:00:00Z"},
            "image-b": {"last_used_at_utc": "2026-01-02T00:00:00Z"},
            "image-c": {"last_used_at_utc": "2026-01-03T00:00:00Z"},
        }
    )

    assert report["trim"]["removed"] == ["image-a", "image-b"]
    assert docker.images == {"image-c"}


def test_active_lease_protects_image_until_release(tmp_path: Path, monkeypatch) -> None:
    _fixed_disk_usage(monkeypatch)
    docker = FakeDocker(["image-a"])
    cache = ProjectDockerImageCache(
        manifest_path=tmp_path / "cache.json",
        policy=DockerImageCachePolicy(max_images=1, min_free_gib=0, target_free_gib=0),
        command_runner=docker,
    )
    cache.adopt({"image-a": {"last_used_at_utc": "2026-01-01T00:00:00Z"}})
    cache.policy = DockerImageCachePolicy(max_images=0, min_free_gib=0, target_free_gib=0)

    with cache.lease(["image-a"], owner="test-run") as lease:
        assert "image-a" in docker.images
        assert lease.reports[0]["removed"] == []

    assert "image-a" not in docker.images
    assert lease.reports[-1]["removed"] == ["image-a"]


def test_images_referenced_by_containers_are_not_removed(tmp_path: Path, monkeypatch) -> None:
    _fixed_disk_usage(monkeypatch)
    docker = FakeDocker(["image-a"])
    docker.containers["image-a"] = ["container-1"]
    cache = ProjectDockerImageCache(
        manifest_path=tmp_path / "cache.json",
        policy=DockerImageCachePolicy(max_images=0, min_free_gib=0, target_free_gib=0),
        command_runner=docker,
    )

    report = cache.adopt({"image-a": {"last_used_at_utc": "2026-01-01T00:00:00Z"}})

    assert "image-a" in docker.images
    assert report["trim"]["skipped"] == [
        {"image": "image-a", "reason": "container", "ids": ["container-1"]}
    ]
