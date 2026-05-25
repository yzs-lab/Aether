"""Reconstructed GreenServe/IPW formulas."""

from __future__ import annotations

import math


BYTES_PER_GB = 1024 ** 3


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, kv_dtype_bytes: int) -> int:
    return int(2 * layers * kv_heads * head_dim * kv_dtype_bytes)


def effective_kv_bytes(
    kv_bytes: float,
    compression_ratio: float,
    metadata_bytes_per_token: float = 0.0,
) -> float:
    if compression_ratio <= 0:
        raise ValueError("compression_ratio must be positive")
    return kv_bytes * compression_ratio + metadata_bytes_per_token


def available_kv_vram_bytes(
    vram_gb: float,
    model_weight_gb: float,
    gpu_memory_utilization: float = 0.9,
) -> float:
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    return max(0.0, (vram_gb * gpu_memory_utilization - model_weight_gb) * BYTES_PER_GB)


def max_inflight_sequences(
    available_vram_bytes: float,
    effective_kv_bytes_per_token: float,
    context_tokens: int,
) -> int:
    if effective_kv_bytes_per_token <= 0 or context_tokens <= 0:
        return 0
    return int(math.floor(available_vram_bytes / (effective_kv_bytes_per_token * context_tokens)))


def phase_energy_j(seconds: float, power_w: float) -> float:
    return max(0.0, seconds) * max(0.0, power_w)


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def risk_adjusted_cost(expected_cost: float, uncertainty: float, risk_weight: float) -> float:
    return expected_cost + risk_weight * uncertainty


def recompute_cost(
    context_tokens: int,
    prefill_tps: float,
    tdp_w: float,
    gpu_count: int,
    prefill_power_fraction: float = 0.86,
    uncertainty_fraction: float = 0.10,
) -> dict:
    seconds = safe_div(context_tokens, prefill_tps)
    energy = phase_energy_j(seconds, gpu_count * tdp_w * prefill_power_fraction)
    return {
        "seconds": seconds,
        "energy_j": energy,
        "uncertainty_j": energy * uncertainty_fraction,
    }


def swap_cost(
    kv_bytes: float,
    pcie_bandwidth_gbps: float,
    tdp_w: float,
    gpu_count: int,
    decode_power_fraction: float = 0.43,
    uncertainty_fraction: float = 0.20,
) -> dict:
    bandwidth_bytes_s = max(1.0, pcie_bandwidth_gbps * 1_000_000_000.0)
    seconds = kv_bytes / bandwidth_bytes_s
    energy = phase_energy_j(seconds, gpu_count * tdp_w * decode_power_fraction)
    return {
        "seconds": seconds,
        "energy_j": energy,
        "uncertainty_j": energy * uncertainty_fraction,
    }
