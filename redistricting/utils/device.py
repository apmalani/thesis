"""Device and backend utility helpers."""

import os
import subprocess
from pathlib import Path
from typing import Optional

import torch


def check_rocm_installed() -> bool:
    """Return True if ROCm appears available on the system."""
    try:
        subprocess.run(["rocm-smi", "--version"], capture_output=True, check=True, timeout=2)
        return True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        return Path("/opt/rocm/bin/rocm-smi").exists()


def get_device(device: Optional[torch.device] = None) -> torch.device:
    """Return compute device with CUDA/ROCm preference and CPU fallback.

    Environment override (optional):
      REDISTRICTING_DEVICE=cpu   — force CPU even if CUDA is available
      REDISTRICTING_DEVICE=cuda  — use CUDA if available, else raise RuntimeError
      unset / auto               — CUDA when available, else ROCm heuristic, else CPU
    """
    if device is not None:
        return device
    override = os.environ.get("REDISTRICTING_DEVICE", "").strip().lower()
    if override == "cpu":
        return torch.device("cpu")
    if override == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("REDISTRICTING_DEVICE=cuda but torch.cuda.is_available() is False")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if check_rocm_installed() or Path("/opt/rocm").exists():
        try:
            _ = torch.zeros(1).to(torch.device("cuda"))
            return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


def get_device_name() -> str:
    """Return human-readable backend name."""
    if torch.cuda.is_available():
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            return f"ROCm (HIP {torch.version.hip})"
        return "CUDA"
    return "CPU"


def supports_compile() -> bool:
    """Return True if `torch.compile` appears available."""
    return hasattr(torch, "compile")


def setup_kernel_optimizations() -> None:
    """Set backend optimization knobs where possible."""
    if torch.cuda.is_available() and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"
        os.environ["MIOPEN_FIND_MODE"] = "1"

