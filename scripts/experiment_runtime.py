#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence


def ensure_runtime_dirs(base_dir: Path) -> dict[str, Path]:
    """Create repo-local cache dirs used by experiment subprocesses."""
    dirs = {
        "mpl": base_dir / ".tmp_mpl",
        "ultralytics": base_dir / ".tmp_ultralytics",
        "home": base_dir / ".tmp_home",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def build_experiment_env(
    base_dir: Path,
    *,
    cuda_visible_devices: str | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the standard child-process env for reproducible experiments.

    `cuda_visible_devices=None` preserves the caller's CUDA visibility. Any
    string value is an explicit user mask and is passed through unchanged.
    """
    base_dir = base_dir.resolve()
    dirs = ensure_runtime_dirs(base_dir)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(dirs["mpl"])
    env["YOLO_CONFIG_DIR"] = str(dirs["ultralytics"])
    env.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


def requested_device(device: str, require_cuda: bool) -> str:
    if require_cuda and device == "auto":
        return "cuda"
    return device


def cuda_probe_code() -> str:
    return (
        "import json, os, sys, torch; "
        "info={'python': sys.version.split()[0], "
        "'executable': sys.executable, "
        "'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), "
        "'torch_version': torch.__version__, "
        "'torch_cuda_build': torch.version.cuda, "
        "'cuda_available_flag': bool(torch.cuda.is_available()), "
        "'device_count_flag': int(torch.cuda.device_count()), "
        "'allocation_ok': False, "
        "'devices': [], "
        "'error': None}; "
        "\ntry:\n"
        "    if torch.cuda.is_available():\n"
        "        x = torch.zeros((1,), device='cuda')\n"
        "        torch.cuda.synchronize()\n"
        "        info['allocation_ok'] = True\n"
        "        info['devices'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]\n"
        "except Exception as exc:\n"
        "    info['error'] = type(exc).__name__ + ': ' + str(exc)\n"
        "print(json.dumps(info))\n"
        "raise SystemExit(0 if info['allocation_ok'] else 1)"
    )


def cuda_preflight(
    python_exec: Path,
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> dict[str, object]:
    """Run a real CUDA allocation probe in a subprocess."""
    proc = subprocess.run(
        [str(python_exec), "-c", cuda_probe_code()],
        cwd=str(cwd) if cwd is not None else None,
        env=dict(env) if env is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    payload: dict[str, object]
    try:
        payload = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        payload = {"error": lines[-1] if lines else "empty CUDA probe output"}
    payload["returncode"] = proc.returncode
    payload["raw_output"] = proc.stdout or ""
    return payload


def require_cuda_preflight(
    python_exec: Path,
    *,
    env: Mapping[str, str],
    cwd: Path,
) -> dict[str, object]:
    payload = cuda_preflight(python_exec, env=env, cwd=cwd)
    if not bool(payload.get("allocation_ok")):
        raise RuntimeError(
            "CUDA was required, but the selected Python runtime could not allocate a CUDA tensor. "
            f"Probe: {json.dumps(payload, indent=2, default=str)}"
        )
    return payload


def run_logged(
    cmd: Sequence[str],
    log_path: Path,
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> float:
    start = time.time()
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        env=dict(env) if env is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    runtime = time.time() - start
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "$ " + " ".join(str(part) for part in cmd) + "\n\n"
        + (proc.stdout or "")
        + f"\n[exit_code={proc.returncode} runtime_sec={runtime:.3f}]\n",
        encoding="utf-8",
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}; see {log_path}")
    return runtime


def assert_isolated_python() -> dict[str, object]:
    """Fail when the active venv can import distro site-packages."""
    bad_paths = [path for path in sys.path if path.startswith("/usr/lib/python3/dist-packages")]
    result = {
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "bad_sys_path_entries": bad_paths,
    }
    if bad_paths:
        raise RuntimeError(f"Python environment is not isolated: {json.dumps(result, indent=2)}")
    return result
