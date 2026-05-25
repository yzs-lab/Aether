"""Deterministic CPU-only measurement backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from .. import formulas
from ..config import experiment_name
from .summary import measurement_summary_row
from ..results import write_csv, write_jsonl


def _measurement(config: Mapping[str, Any]) -> Dict[str, Any]:
    measurement = dict(config.get("measurement", {}) or {})
    measurement.setdefault("duration_seconds", 5)
    measurement.setdefault("sample_hz", 10)
    measurement.setdefault("request_count", 8)
    return measurement


def _model(config: Mapping[str, Any]) -> Dict[str, Any]:
    model = dict(config.get("model", {}) or {})
    model.setdefault("name", "mock-model")
    model.setdefault("layers", 32)
    model.setdefault("kv_heads", 8)
    model.setdefault("head_dim", 128)
    model.setdefault("kv_dtype_bytes", 2)
    model.setdefault("quality_score", 1.0)
    return model


def _workload(config: Mapping[str, Any]) -> Dict[str, Any]:
    workload = dict(config.get("workload", {}) or {})
    workload.setdefault("prompt_tokens", 2048)
    workload.setdefault("generated_tokens", 128)
    workload.setdefault("target_concurrency", 8)
    return workload


def run_mock(config: Mapping[str, Any], out_dir: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    measurement = _measurement(config)
    model = _model(config)
    workload = _workload(config)
    exp_name = experiment_name(config)
    duration = float(measurement["duration_seconds"])
    sample_hz = int(measurement["sample_hz"])
    request_count = int(measurement["request_count"])
    prompt_tokens = int(workload["prompt_tokens"])
    generated_per_request = int(workload["generated_tokens"])
    kv_bytes = formulas.kv_bytes_per_token(
        int(model["layers"]),
        int(model["kv_heads"]),
        int(model["head_dim"]),
        int(model["kv_dtype_bytes"]),
    )

    events = []
    samples = max(1, int(duration * sample_hz))
    for index in range(samples):
        ts = round(index / float(sample_hz), 6)
        power_w = 225.0 + float((index * 7) % 19)
        events.append(
            {
                "type": "power",
                "time_s": ts,
                "power_w": power_w,
                "backend": "mock",
                "hardware_backend": "mock",
            }
        )

    for request_id in range(request_count):
        token_count = generated_per_request
        events.append(
            {
                "type": "request",
                "backend": "mock",
                "request_id": request_id,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": token_count,
                "ttft_ms": 35.0 + request_id,
                "tbt_ms": 12.0 + (request_id % 3),
            }
        )

    events.extend(
        [
            {
                "type": "kv",
                "allocated_bytes": kv_bytes * prompt_tokens * request_count,
                "kv_bytes_per_token": kv_bytes,
            },
            {
                "type": "scheduler",
                "action": "recompute",
                "request_id": 0,
                "tokens": prompt_tokens // 2,
            },
            {
                "type": "scheduler",
                "action": "swap",
                "request_id": 1,
                "bytes": kv_bytes * prompt_tokens,
            },
        ]
    )

    quality = float(model.get("quality_score", 1.0))
    row = measurement_summary_row(
        config,
        events,
        backend="mock",
        scenario_id=exp_name + "-mock",
        hardware="mock-cpu",
        routing_policy="mock",
        routing_pool="mock",
        scheduling_policy="mock",
        scheduling_action="mixed",
        fallback_elapsed_seconds=duration,
        extra={
            "context_tokens": prompt_tokens,
            "target_concurrency": workload.get("target_concurrency", request_count),
            "active_sequences": request_count,
            "kv_compression_ratio": 1.0,
            "kv_bytes_per_token": kv_bytes,
            "effective_kv_bytes_per_token": kv_bytes,
            "max_inflight_sequences": request_count,
            "prefill_seconds": 0.035,
            "decode_seconds": round(max(0.0, duration - 0.035), 6),
            "quality_score": quality,
            "swap_bytes": kv_bytes * prompt_tokens,
            "recompute_tokens": prompt_tokens // 2,
            "slo_violations": 0,
        },
    )

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv([row], str(output_dir / "summary.csv"))
    write_jsonl(events, str(output_dir / "events.jsonl"))
    return row, events
