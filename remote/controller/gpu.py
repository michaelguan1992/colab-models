"""nvidia-smi probes. Per-process VRAM lets the policy use *measured* usage."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class GpuSnapshot:
    name: str = "unknown"
    used_gb: float = 0.0
    total_gb: float = 0.0
    per_pid_gb: dict[int, float] = field(default_factory=dict)
    available: bool = False
    compute_cap: str = ""  # e.g. "12.0"; empty when unknown


def _smi(args: list[str]) -> list[list[str]]:
    out = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10, check=True).stdout
    return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def probe() -> GpuSnapshot:
    if not shutil.which("nvidia-smi"):
        return GpuSnapshot()
    try:
        gpu = _smi(["--query-gpu=name,memory.used,memory.total"])[0]
        apps = _smi(["--query-compute-apps=pid,used_memory"])
    except (subprocess.SubprocessError, IndexError, OSError):
        return GpuSnapshot()
    per_pid: dict[int, float] = {}
    for row in apps:
        try:
            per_pid[int(row[0])] = float(row[1]) / 1024
        except (ValueError, IndexError):
            continue
    try:
        compute_cap = _smi(["--query-gpu=compute_cap"])[0][0]
    except (subprocess.SubprocessError, IndexError, OSError):
        compute_cap = ""  # older drivers lack the field; routes with `requires` then cannot match
    return GpuSnapshot(name=gpu[0], used_gb=float(gpu[1]) / 1024, total_gb=float(gpu[2]) / 1024,
                       per_pid_gb=per_pid, available=True, compute_cap=compute_cap)


def process_tree(pid: int) -> set[int]:
    try:
        import psutil
        p = psutil.Process(pid)
        return {pid, *(c.pid for c in p.children(recursive=True))}
    except Exception:
        return {pid}
