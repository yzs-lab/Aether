"""Scenario expansion and GreenServe/IPW simulation."""

from __future__ import annotations

import itertools
from typing import Any, Dict, Iterable, List, Mapping

from . import formulas
from .config import as_list, experiment_name


def _first(items: Iterable[Dict[str, Any]], default: Dict[str, Any]) -> Dict[str, Any]:
    for item in items:
        return dict(item)
    return dict(default)


def _hardware_profiles(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    simulation = dict(config.get("simulation", {}) or {})
    profiles = simulation.get("hardware_profiles") or []
    if not profiles:
        profiles = [{"name": "default-gpu", "tdp_w": 700, "vram_gb": 80, "gpu_count": 1}]
    return [dict(profile) for profile in profiles]


def _model_profile(config: Mapping[str, Any]) -> Dict[str, Any]:
    model = dict(config.get("model", {}) or {})
    model.setdefault("name", "unknown-model")
    model.setdefault("layers", 32)
    model.setdefault("kv_heads", 8)
    model.setdefault("head_dim", 128)
    model.setdefault("kv_dtype_bytes", 2)
    model.setdefault("weight_gb", 0)
    model.setdefault("quality_score", 1.0)
    return model


def _workload(config: Mapping[str, Any]) -> Dict[str, Any]:
    workload = dict(config.get("workload", {}) or {})
    workload.setdefault("prompt_tokens", 4096)
    workload.setdefault("generated_tokens", 256)
    workload.setdefault("target_concurrency", 1)
    workload.setdefault("preemption_probability", 0.0)
    workload.setdefault("ttft_slo_ms", 2000)
    workload.setdefault("tbt_slo_ms", 200)
    return workload


def _sweeps(config: Mapping[str, Any], workload: Mapping[str, Any]) -> Dict[str, List[Any]]:
    simulation = dict(config.get("simulation", {}) or {})
    sweep = dict(simulation.get("sweeps", simulation.get("sweep", {})) or {})
    return {
        "context_lengths": as_list(sweep.get("context_lengths"), [workload["prompt_tokens"]]),
        "kv_compression_ratios": as_list(sweep.get("kv_compression_ratios"), [1.0]),
        "routing_policies": as_list(sweep.get("routing_policies"), ["single_pool"]),
        "scheduling_policies": as_list(sweep.get("scheduling_policies"), ["none"]),
    }


def _select_pool(config: Mapping[str, Any], policy: str, context_tokens: int, hardware: Mapping[str, Any]) -> Dict[str, Any]:
    simulation = dict(config.get("simulation", {}) or {})
    pools = [dict(pool) for pool in simulation.get("routing_pools", []) or []]
    if not pools:
        return {
            "name": "default",
            "gpu_count": int(hardware.get("gpu_count", 1)),
            "max_context_tokens": context_tokens,
        }
    if policy == "fleetopt":
        candidates = [pool for pool in pools if int(pool.get("max_context_tokens", context_tokens)) >= context_tokens]
        if candidates:
            return sorted(candidates, key=lambda item: int(item.get("max_context_tokens", context_tokens)))[0]
    return pools[0]


def _scheduler_overhead(
    policy: str,
    context_tokens: int,
    active_sequences: int,
    effective_kv_bytes_per_token: float,
    workload: Mapping[str, Any],
    hardware: Mapping[str, Any],
    prefill_tps: float,
) -> Dict[str, Any]:
    probability = float(workload.get("preemption_probability", 0.0))
    preempted_sequences = int(round(active_sequences * probability))
    if policy == "none" or preempted_sequences <= 0:
        return {"action": "none", "seconds": 0.0, "energy_j": 0.0, "swap_bytes": 0.0, "recompute_tokens": 0}

    gpu_count = int(hardware.get("gpu_count", 1))
    tdp_w = float(hardware.get("tdp_w", 700))
    pcie_gbps = float(hardware.get("pcie_bandwidth_gbps", 64))
    risk_weight = float(hardware.get("scheduler_risk_weight", 1.0))
    preempted_context_tokens = context_tokens * preempted_sequences
    kv_bytes = effective_kv_bytes_per_token * preempted_context_tokens

    recompute = formulas.recompute_cost(preempted_context_tokens, prefill_tps, tdp_w, gpu_count)
    swap = formulas.swap_cost(kv_bytes, pcie_gbps, tdp_w, gpu_count)

    if policy == "recompute":
        action = "recompute"
    elif policy == "swap":
        action = "swap"
    else:
        recompute_score = formulas.risk_adjusted_cost(recompute["energy_j"], recompute["uncertainty_j"], risk_weight)
        swap_score = formulas.risk_adjusted_cost(swap["energy_j"], swap["uncertainty_j"], risk_weight)
        action = "recompute" if recompute_score < swap_score else "swap"

    selected = recompute if action == "recompute" else swap
    return {
        "action": action,
        "seconds": selected["seconds"],
        "energy_j": selected["energy_j"],
        "swap_bytes": kv_bytes if action == "swap" else 0.0,
        "recompute_tokens": preempted_context_tokens if action == "recompute" else 0,
    }


def simulate(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    model = _model_profile(config)
    workload = _workload(config)
    sweeps = _sweeps(config, workload)
    rows = []
    exp_name = experiment_name(config)
    for hardware, context_tokens, compression_ratio, routing_policy, scheduling_policy in itertools.product(
        _hardware_profiles(config),
        sweeps["context_lengths"],
        sweeps["kv_compression_ratios"],
        sweeps["routing_policies"],
        sweeps["scheduling_policies"],
    ):
        rows.append(
            simulate_one(
                config,
                exp_name,
                model,
                workload,
                hardware,
                int(context_tokens),
                float(compression_ratio),
                str(routing_policy),
                str(scheduling_policy),
            )
        )
    return rows


def simulate_one(
    config: Mapping[str, Any],
    exp_name: str,
    model: Mapping[str, Any],
    workload: Mapping[str, Any],
    hardware: Mapping[str, Any],
    context_tokens: int,
    compression_ratio: float,
    routing_policy: str,
    scheduling_policy: str,
) -> Dict[str, Any]:
    pool = _select_pool(config, routing_policy, context_tokens, hardware)
    gpu_count = int(pool.get("gpu_count", hardware.get("gpu_count", 1)))
    tdp_w = float(hardware.get("tdp_w", 700))
    prefill_tps = float(hardware.get("prefill_tps_per_gpu", 5000)) * gpu_count
    decode_tps = float(hardware.get("decode_tps_per_gpu", 250)) * gpu_count
    target_concurrency = int(workload.get("target_concurrency", 1))
    generated_tokens_per_request = int(workload.get("generated_tokens", 256))
    prompt_tokens = int(workload.get("prompt_tokens", context_tokens))

    kv_bytes = formulas.kv_bytes_per_token(
        int(model["layers"]),
        int(model["kv_heads"]),
        int(model["head_dim"]),
        int(model["kv_dtype_bytes"]),
    )
    effective_kv = formulas.effective_kv_bytes(
        kv_bytes,
        compression_ratio,
        float(model.get("kv_metadata_bytes_per_token", 0.0)),
    )
    model_weight_gb_per_gpu = float(
        model.get("weight_gb_per_gpu", formulas.safe_div(float(model.get("weight_gb", 0.0)), max(1, gpu_count)))
    )
    available_vram = formulas.available_kv_vram_bytes(
        float(hardware.get("vram_gb", 80)),
        model_weight_gb_per_gpu,
        float(hardware.get("gpu_memory_utilization", 0.9)),
    ) * gpu_count
    max_inflight = formulas.max_inflight_sequences(available_vram, effective_kv, context_tokens)
    active_sequences = max(0, min(target_concurrency, max_inflight))
    useful_concurrency_fraction = min(1.0, formulas.safe_div(max_inflight, max(1, target_concurrency)))
    effective_decode_tps = max(1.0, decode_tps * useful_concurrency_fraction)

    total_generated_tokens = active_sequences * generated_tokens_per_request
    prefill_seconds = formulas.safe_div(active_sequences * prompt_tokens, max(1.0, prefill_tps))
    decode_seconds = formulas.safe_div(total_generated_tokens, effective_decode_tps)
    prefill_power = gpu_count * tdp_w * float(hardware.get("prefill_power_fraction", 0.86))
    decode_power = gpu_count * tdp_w * float(hardware.get("decode_power_fraction", 0.43))
    prefill_energy = formulas.phase_energy_j(prefill_seconds, prefill_power)
    decode_energy = formulas.phase_energy_j(decode_seconds, decode_power)

    overhead = _scheduler_overhead(
        scheduling_policy,
        context_tokens,
        active_sequences,
        effective_kv,
        workload,
        dict(hardware, gpu_count=gpu_count),
        prefill_tps,
    )
    elapsed = prefill_seconds + decode_seconds + overhead["seconds"]
    energy = prefill_energy + decode_energy + overhead["energy_j"]
    avg_power = formulas.safe_div(energy, elapsed)
    tokens_per_second = formulas.safe_div(total_generated_tokens, elapsed)
    tokens_per_watt = formulas.safe_div(tokens_per_second, avg_power)
    tokens_per_joule = formulas.safe_div(total_generated_tokens, energy)
    ttft_ms = 1000.0 * prefill_seconds
    tbt_ms = 1000.0 * formulas.safe_div(decode_seconds, max(1, generated_tokens_per_request))
    slo_violations = int(ttft_ms > float(workload.get("ttft_slo_ms", 2000))) + int(
        tbt_ms > float(workload.get("tbt_slo_ms", 200))
    )

    scenario_id = "%s-%s-ctx%s-kv%s-%s-%s" % (
        exp_name,
        hardware.get("name", "hardware"),
        context_tokens,
        compression_ratio,
        routing_policy,
        scheduling_policy,
    )
    quality = float(model.get("quality_score", 1.0))
    return {
        "experiment": exp_name,
        "backend": "simulation",
        "scenario_id": scenario_id,
        "model": model.get("name", "unknown-model"),
        "hardware": hardware.get("name", "unknown-hardware"),
        "routing_policy": routing_policy,
        "routing_pool": pool.get("name", "default"),
        "scheduling_policy": scheduling_policy,
        "scheduling_action": overhead["action"],
        "context_tokens": context_tokens,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": total_generated_tokens,
        "target_concurrency": target_concurrency,
        "active_sequences": active_sequences,
        "kv_compression_ratio": compression_ratio,
        "kv_bytes_per_token": kv_bytes,
        "effective_kv_bytes_per_token": round(effective_kv, 6),
        "max_inflight_sequences": max_inflight,
        "prefill_seconds": round(prefill_seconds, 6),
        "decode_seconds": round(decode_seconds, 6),
        "elapsed_seconds": round(elapsed, 6),
        "energy_j": round(energy, 6),
        "avg_power_w": round(avg_power, 6),
        "tokens_per_second": round(tokens_per_second, 6),
        "tokens_per_watt": round(tokens_per_watt, 9),
        "tokens_per_joule": round(tokens_per_joule, 9),
        "quality_score": quality,
        "quality_normalized_ipw": round(quality * tokens_per_joule, 9),
        "swap_bytes": round(overhead["swap_bytes"], 6),
        "recompute_tokens": overhead["recompute_tokens"],
        "ttft_ms": round(ttft_ms, 6),
        "tbt_ms": round(tbt_ms, 6),
        "slo_violations": slo_violations,
    }
