"""Common measurement summarization for all serving and hardware backends."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from .. import formulas
from ..config import experiment_name


def as_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def as_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def request_summary(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    request_events = [event for event in events if event.get("type") == "request"]
    prompt_tokens = sum(as_int(event.get("prompt_tokens")) for event in request_events)
    generated_tokens = sum(as_int(event.get("generated_tokens")) for event in request_events)
    request_latency = sum(as_float(event.get("latency_s")) for event in request_events)
    response_latency = sum(
        as_float(event.get("response_latency_s")) for event in request_events
    )
    ttft_values = [
        as_float(event.get("ttft_ms"))
        for event in request_events
        if as_float(event.get("ttft_ms")) > 0
    ]
    tbt_values = [
        as_float(event.get("tbt_ms"))
        for event in request_events
        if as_float(event.get("tbt_ms")) > 0
    ]
    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "request_latency_s": request_latency,
        "response_latency_s": response_latency,
        "ttft_ms": round(sum(ttft_values) / len(ttft_values), 6)
        if ttft_values
        else 0.0,
        "tbt_ms": round(sum(tbt_values) / len(tbt_values), 6) if tbt_values else 0.0,
    }


def power_summary(events: List[Dict[str, Any]], fallback_duration: float) -> Tuple[float, float, float]:
    by_ts: Dict[float, float] = {}
    for event in events:
        if event.get("type") == "power":
            ts = float(event.get("time_s", 0.0))
            by_ts.setdefault(ts, 0.0)
            by_ts[ts] += as_float(event.get("power_w"))
    if not by_ts:
        return fallback_duration, 0.0, 0.0
    elapsed = max(float(fallback_duration), max(by_ts))
    avg_power = sum(by_ts.values()) / len(by_ts)
    return elapsed, avg_power, avg_power * elapsed


def apply_patched_metric_overrides(
    requests: Dict[str, Any],
    patched_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    patched_summary = (
        patched_metrics.get("summary", {})
        if isinstance(patched_metrics.get("summary"), Mapping)
        else {}
    )
    prompt_tokens = as_int(patched_summary.get("prompt_tokens_total"))
    generated_tokens = as_int(patched_summary.get("completion_tokens_total"))
    ttft_s = as_float(patched_summary.get("ttft_avg_s"))
    tbt_s = as_float(patched_summary.get("tbt_avg_s"))
    e2e_latency_sum_s = as_float(patched_summary.get("e2e_latency_sum_s"))
    e2e_latency_avg_s = as_float(patched_summary.get("e2e_latency_avg_s"))
    request_count = as_int(patched_summary.get("request_count"))
    updated = dict(requests)
    if prompt_tokens:
        updated["prompt_tokens"] = prompt_tokens
    if generated_tokens:
        updated["generated_tokens"] = generated_tokens
    if e2e_latency_sum_s:
        updated["patched_e2e_latency_s"] = e2e_latency_sum_s
    elif e2e_latency_avg_s and request_count:
        updated["patched_e2e_latency_s"] = e2e_latency_avg_s * request_count
    if ttft_s and not as_float(updated.get("ttft_ms")):
        updated["ttft_ms"] = round(ttft_s * 1000.0, 6)
    if tbt_s and not as_float(updated.get("tbt_ms")):
        updated["tbt_ms"] = round(tbt_s * 1000.0, 6)
    return updated


def measurement_summary_row(
    config: Mapping[str, Any],
    events: List[Dict[str, Any]],
    *,
    backend: str,
    scenario_id: str,
    hardware: str,
    routing_policy: str,
    routing_pool: str,
    scheduling_policy: str,
    scheduling_action: str,
    fallback_elapsed_seconds: float,
    patched_metrics: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    requests = request_summary(events)
    if patched_metrics:
        requests = apply_patched_metric_overrides(requests, patched_metrics)
    fallback_elapsed = max(
        fallback_elapsed_seconds,
        as_float(requests.get("request_latency_s")),
        as_float(requests.get("response_latency_s")),
        as_float(requests.get("patched_e2e_latency_s")),
        0.000001,
    )
    elapsed, avg_power, energy = power_summary(events, fallback_elapsed)
    generated_tokens = int(requests["generated_tokens"])
    tokens_per_second = formulas.safe_div(generated_tokens, elapsed)
    tokens_per_watt = formulas.safe_div(tokens_per_second, avg_power)
    tokens_per_joule = formulas.safe_div(generated_tokens, energy)
    model = config.get("model", {}) or {}
    quality = float(model.get("quality_score", 1.0))
    row: Dict[str, Any] = {
        "experiment": experiment_name(config),
        "backend": backend,
        "scenario_id": scenario_id,
        "model": model.get("name", "unknown-model"),
        "hardware": hardware,
        "routing_policy": routing_policy,
        "routing_pool": routing_pool,
        "scheduling_policy": scheduling_policy,
        "scheduling_action": scheduling_action,
        "prompt_tokens": int(requests["prompt_tokens"]),
        "generated_tokens": generated_tokens,
        "elapsed_seconds": round(elapsed, 6),
        "energy_j": round(energy, 6),
        "avg_power_w": round(avg_power, 6),
        "tokens_per_second": round(tokens_per_second, 6),
        "tokens_per_watt": round(tokens_per_watt, 9),
        "tokens_per_joule": round(tokens_per_joule, 9),
        "quality_score": quality,
        "quality_normalized_ipw": round(quality * tokens_per_joule, 9),
        "ttft_ms": requests["ttft_ms"],
        "tbt_ms": requests["tbt_ms"],
    }
    if extra:
        row.update(dict(extra))
    return row
