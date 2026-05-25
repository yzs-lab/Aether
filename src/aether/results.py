"""Normalized result writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


SUMMARY_FIELDS = [
    "experiment",
    "backend",
    "scenario_id",
    "model",
    "hardware",
    "routing_policy",
    "routing_pool",
    "scheduling_policy",
    "scheduling_action",
    "context_tokens",
    "prompt_tokens",
    "generated_tokens",
    "target_concurrency",
    "active_sequences",
    "kv_compression_ratio",
    "kv_bytes_per_token",
    "effective_kv_bytes_per_token",
    "max_inflight_sequences",
    "prefill_seconds",
    "decode_seconds",
    "elapsed_seconds",
    "energy_j",
    "avg_power_w",
    "tokens_per_second",
    "tokens_per_watt",
    "tokens_per_joule",
    "quality_score",
    "quality_normalized_ipw",
    "swap_bytes",
    "recompute_tokens",
    "ttft_ms",
    "tbt_ms",
    "slo_violations",
]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalized_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {field: row.get(field, "") for field in SUMMARY_FIELDS}


def write_csv(rows: Iterable[Dict[str, Any]], path: str) -> None:
    output_path = Path(path)
    ensure_parent(output_path)
    normalized_rows = [normalized_row(row) for row in rows]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(normalized_rows)


def write_jsonl(events: Iterable[Dict[str, Any]], path: str) -> None:
    output_path = Path(path)
    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
