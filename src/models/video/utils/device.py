# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

"""
Device detection and GPU utilities for ASD Pipeline training.

Provides:
  - Automatic device selection (CUDA → CPU fallback)
  - GPU info reporting
  - Memory monitoring helpers
  - CUDA optimization toggles
"""

import os
import torch
import torch.backends.cudnn as cudnn

# Track memory limit (GB) if set, for logging purposes
_GPU_LIMIT_GB = None


def get_device(prefer: str = "cuda") -> str:
    """
    Select best available device.

    Args:
        prefer: preferred device ("cuda" or "cpu")

    Returns:
        "cuda" if GPU available and preferred, else "cpu"
    """
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def configure_cuda_optimizations():
    """
    Apply standard CUDA training optimizations.
    Call once before training starts.
    """
    if not torch.cuda.is_available():
        print("[device] No CUDA GPU detected — running on CPU")
        return

    # cuDNN auto-tuner — finds optimal algorithms for your input sizes
    cudnn.benchmark = True

    # Allow TF32 on Ampere+ GPUs (faster matmuls with ~same accuracy)
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.allow_tf32 = True

    # Higher precision for float32 matmuls (good on Ampere+)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def print_gpu_info():
    """Print GPU device info for logging."""
    if not torch.cuda.is_available():
        print("[device] CUDA: not available")
        print("[device] Training will use CPU")
        return

    n_gpus = torch.cuda.device_count()
    print(f"[device] CUDA: {torch.version.cuda}")
    print(f"[device] cuDNN: {cudnn.version() if cudnn.is_available() else 'N/A'}")
    print(f"[device] GPUs: {n_gpus}")

    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / (1024 ** 3)
        print(
            f"  [{i}] {props.name}  |  {mem_gb:.1f} GB  |  "
            f"{props.multi_processor_count} SMs  |  "
            f"Compute {props.major}.{props.minor}"
        )

    print(f"[device] cuDNN benchmark: {cudnn.benchmark}")
    print(f"[device] TF32 allowed: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"[device] torch.compile: {'available' if hasattr(torch, 'compile') else 'N/A'}")


def get_gpu_memory_usage(device_id: int = 0) -> dict:
    """
    Get current GPU memory usage.

    Returns:
        dict with 'allocated_mb', 'reserved_mb', 'free_mb', 'total_mb'
    """
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0, "free_mb": 0, "total_mb": 0}

    allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 2)
    total = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 2)
    free = total - reserved

    res = {
        "allocated_mb": round(allocated, 1),
        "reserved_mb": round(reserved, 1),
        "free_mb": round(free, 1),
        "total_mb": round(total, 1),
    }
    if _GPU_LIMIT_GB is not None:
        res["limit_mb"] = _GPU_LIMIT_GB * 1024
    return res


def log_memory(tag: str = "", device_id: int = 0):
    """Print GPU memory snapshot with optional tag."""
    if not torch.cuda.is_available():
        return
    mem = get_gpu_memory_usage(device_id)
    msg = (
        f"[GPU mem{' | ' + tag if tag else ''}] "
        f"alloc={mem['allocated_mb']:.0f}MB  "
        f"reserved={mem['reserved_mb']:.0f}MB  "
        f"free={mem['free_mb']:.0f}MB  "
        f"total={mem['total_mb']:.0f}MB"
    )
    if "limit_mb" in mem:
        msg += f"  (limit={mem['limit_mb']:.0f}MB)"
    print(msg)


def optimal_workers() -> int:
    """
    Suggest a reasonable num_workers for DataLoader.

    Heuristic: min(cpu_count // 2, 8), at least 2 if multicore.
    """
    try:
        cpus = os.cpu_count() or 1
        if cpus <= 2:
            return 0  # avoid overhead on low-core machines
        return min(cpus // 2, 8)
    except Exception:
        return 0


def limit_gpu_memory(limit_gb: float):
    """
    Limits the GPU memory usage to the specified amount (in GB).
    """
    global _GPU_LIMIT_GB
    if not torch.cuda.is_available():
        return
    
    device_id = torch.cuda.current_device()
    total_mem = torch.cuda.get_device_properties(device_id).total_memory
    fraction = (limit_gb * (1024**3)) / total_mem
    
    if fraction >= 1.0:
        return
    
    try:
        torch.cuda.set_per_process_memory_fraction(fraction, device_id)
        _GPU_LIMIT_GB = limit_gb
        print(f"[device] GPU memory limit set to {limit_gb:.1f}GB ({fraction:.1%} of total)")
    except RuntimeError as e:
        print(f"[device] Failed to set GPU memory limit (must be set before tensor creation): {e}")

