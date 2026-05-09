from __future__ import annotations

from typing import Any, Callable

import torch


def resolve_torch_device(gpu_id: int = 0) -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise ValueError(
                f"❌ Invalid --gpu_id {gpu_id}. Available CUDA devices: 0-{torch.cuda.device_count() - 1}"
            )
        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}"), torch.bfloat16

    return torch.device("cpu"), torch.float32


def format_device_label(device: torch.device, gpu_id: int | None = None) -> str:
    if device.type != "cuda":
        return str(device)

    resolved_gpu_id = gpu_id if gpu_id is not None else (device.index or 0)
    return f"{device} ({torch.cuda.get_device_name(resolved_gpu_id)})"


def clear_cuda_memory(gpu_id: int | None = None):
    if not torch.cuda.is_available():
        return

    try:
        if gpu_id is not None:
            torch.cuda.synchronize(gpu_id)
    except Exception:
        pass

    torch.cuda.empty_cache()


def get_cuda_memory_snapshot(gpu_id: int) -> dict[str, float]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
    total_gb = total_bytes / (1024**3)
    free_gb = free_bytes / (1024**3)
    used_gb = total_gb - free_gb
    return {
        "free_gb": free_gb,
        "used_gb": used_gb,
        "total_gb": total_gb,
    }


def get_pipeline_execution_device(pipe: Any, fallback_device: torch.device) -> torch.device:
    execution_device = getattr(pipe, "_execution_device", None)
    if execution_device is None:
        execution_device = getattr(pipe, "device", fallback_device)

    if isinstance(execution_device, torch.device):
        return execution_device

    return torch.device(execution_device)


def will_use_tiled_denoise(width: int, height: int, tile_size: int = 32) -> bool:
    latent_h = max(height // 16, 1)
    latent_w = max(width // 16, 1)
    return latent_h * latent_w >= tile_size**2


def _auto_thresholds_gb(
    width: int,
    height: int,
    condition_count: int,
    tiled_expected: bool,
) -> tuple[float, float]:
    scale = max((width * height) / float(512 * 512), 1.0)
    scale_penalty = max(scale - 1.0, 0.0)
    condition_penalty = max(condition_count - 1, 0)

    if tiled_expected:
        # Once tiled denoising is active, transformer memory depends much more on per-tile work
        # than on the full image area, so the old area-scaled threshold was far too conservative.
        model_threshold_gb = 22.70 + 0.20 * condition_penalty
        none_threshold_gb = 24.50 + 0.30 * condition_penalty
    else:
        model_threshold_gb = 22.75 + 0.35 * scale_penalty + 0.15 * condition_penalty
        none_threshold_gb = 25.50 + 0.50 * scale_penalty + 0.25 * condition_penalty

    return model_threshold_gb, none_threshold_gb


def select_flux_offload_candidates(
    requested_mode: str,
    gpu_id: int,
    width: int,
    height: int,
    condition_count: int,
    safety_margin_gb: float,
    tiled_expected: bool,
) -> tuple[list[str], str | None]:
    if requested_mode != "auto":
        return [requested_mode], None

    snapshot = get_cuda_memory_snapshot(gpu_id)
    effective_free_gb = max(snapshot["free_gb"] - safety_margin_gb, 0.0)
    model_threshold_gb, none_threshold_gb = _auto_thresholds_gb(
        width=width,
        height=height,
        condition_count=condition_count,
        tiled_expected=tiled_expected,
    )

    if effective_free_gb >= none_threshold_gb:
        candidates = ["none", "model", "sequential"]
    elif effective_free_gb >= model_threshold_gb:
        candidates = ["model", "sequential"]
    else:
        candidates = ["sequential"]

    summary = (
        f"🧠 VRAM snapshot on cuda:{gpu_id}: free {snapshot['free_gb']:.2f} / {snapshot['total_gb']:.2f} GB, "
        f"used {snapshot['used_gb']:.2f} GB, safety margin {safety_margin_gb:.2f} GB, "
        f"effective free {effective_free_gb:.2f} GB. "
        f"Tiled denoise expected: {tiled_expected}. "
        f"Auto candidates for {width}x{height} with {condition_count} condition(s): {', '.join(candidates)}"
    )
    return candidates, summary


def apply_flux_offload_mode(
    pipe: Any,
    mode: str,
    device: torch.device,
    gpu_id: int,
    logger: Callable[[str], None] = print,
) -> str:
    current_mode = getattr(pipe, "_genfocus_offload_mode", None)
    current_gpu_id = getattr(pipe, "_genfocus_gpu_id", None)

    if device.type != "cuda":
        if current_mode != "cpu":
            pipe.to(device)
            pipe._genfocus_offload_mode = "cpu"
        return "cpu"

    if current_mode == mode and current_gpu_id == gpu_id:
        logger(f"🧠 Reusing FLUX memory mode: {mode} on cuda:{gpu_id}")
        return mode

    clear_cuda_memory(gpu_id)

    if mode == "sequential":
        pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)
    elif mode == "model":
        pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    elif mode == "none":
        if current_mode in {"model", "sequential"} and hasattr(pipe, "remove_all_hooks"):
            pipe.remove_all_hooks()
        pipe.to(device)
    else:
        raise ValueError(f"Unsupported offload mode: {mode}")

    pipe._genfocus_offload_mode = mode
    pipe._genfocus_gpu_id = gpu_id
    logger(f"🧠 FLUX memory mode: {mode} on cuda:{gpu_id}")
    return mode


def run_with_flux_memory_policy(
    pipe: Any,
    requested_mode: str,
    device: torch.device,
    gpu_id: int,
    width: int,
    height: int,
    condition_count: int,
    safety_margin_gb: float,
    tiled_expected: bool,
    run_fn: Callable[[str], Any],
    logger: Callable[[str], None] = print,
) -> tuple[Any, str]:
    if device.type != "cuda":
        apply_flux_offload_mode(pipe, "none", device, gpu_id, logger=logger)
        return run_fn("cpu"), "cpu"

    candidates, summary = select_flux_offload_candidates(
        requested_mode=requested_mode,
        gpu_id=gpu_id,
        width=width,
        height=height,
        condition_count=condition_count,
        safety_margin_gb=safety_margin_gb,
        tiled_expected=tiled_expected,
    )

    if summary is not None:
        logger(summary)

    last_error: Exception | None = None
    for idx, mode in enumerate(candidates):
        try:
            apply_flux_offload_mode(pipe, mode, device, gpu_id, logger=logger)
            return run_fn(mode), mode
        except torch.OutOfMemoryError as exc:
            last_error = exc
            if requested_mode != "auto" or idx == len(candidates) - 1:
                raise

            next_mode = candidates[idx + 1]
            logger(f"⚠️ CUDA OOM while running FLUX with mode '{mode}'. Retrying with '{next_mode}'.")
            clear_cuda_memory(gpu_id)

    if last_error is not None:
        raise last_error

    raise RuntimeError("FLUX memory policy did not produce a runnable mode.")
