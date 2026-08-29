#!/usr/bin/env python3
"""Create or repair one LycheeMAS Python environment from install profiles."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--python", required=True, dest="python_executable")
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--create-readme-environment", action="store_true")
    parser.add_argument("--conda-executable", default="")
    parser.add_argument("--conda-env", default="LycheeMAS")
    parser.add_argument("--conda-channels", default="")
    return parser


def _conda_env_python(conda: str, env_name: str) -> Path:
    result = subprocess.run(
        [conda, "run", "-n", env_name, "python", "-c", "import sys; print(sys.executable)"],
        text=True,
        capture_output=True,
        check=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Conda environment {env_name!r} did not report a Python path")
    return Path(lines[-1]).resolve()


def _create_readme_environment(repo_root: Path, args: argparse.Namespace) -> None:
    conda = args.conda_executable or shutil.which("conda")
    if not conda or not Path(conda).is_file():
        raise SystemExit(
            "Conda was not found. Configure Conda initialization in Running Environment, "
            "or create .venv manually from the README commands."
        )
    try:
        conda_python = _conda_env_python(conda, args.conda_env)
    except subprocess.CalledProcessError:
        print(f"[progress] percent=3 creating Conda environment {args.conda_env}", flush=True)
        command = [conda, "create", "-n", args.conda_env, "python=3.12", "-y"]
        for channel in [item.strip() for item in args.conda_channels.split(",") if item.strip()]:
            command.extend(["-c", channel])
        subprocess.run(command, check=True)
        conda_python = _conda_env_python(conda, args.conda_env)

    print("[progress] percent=8 installing uv bootstrap", flush=True)
    subprocess.run(
        [conda, "run", "-n", args.conda_env, "python", "-m", "pip", "install", "uv"],
        check=True,
    )
    print("[progress] percent=12 creating README .venv", flush=True)
    subprocess.run(
        [
            conda,
            "run",
            "-n",
            args.conda_env,
            "python",
            "-m",
            "uv",
            "venv",
            str(repo_root / ".venv"),
            "--python",
            str(conda_python),
            "--seed",
        ],
        check=True,
    )


def main() -> None:
    args = _parser().parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    python = Path(args.python_executable).expanduser().absolute()
    profiles = [item.strip() for item in args.profiles.split(",") if item.strip()]
    if not python.is_file():
        expected = repo_root / ".venv/bin/python"
        if not args.create_readme_environment or python != expected:
            raise SystemExit(f"Python executable not found: {python}")
        _create_readme_environment(repo_root, args)
    sys.path.insert(0, str(repo_root / "src"))
    from lychee_mas.eval.environment.service import installation_command

    command = installation_command(repo_root, str(python), profiles)
    print(f"[progress] percent=15 installing profiles={','.join(profiles)}", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    subprocess.run(command, cwd=repo_root, env=env, check=True)
    print("[progress] percent=100 environment ready", flush=True)


if __name__ == "__main__":
    main()
