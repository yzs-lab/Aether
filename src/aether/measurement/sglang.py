"""Real SGLang launcher and measurement backend.

This module is intentionally optional at runtime. It imports GPU-specific
libraries only inside functions so CPU simulation and tests remain clean.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..config import render_sglang_args
from ..results import write_csv, write_jsonl
from .profiling import HardwareProfilerError, build_hardware_profiler
from .summary import as_float, as_int, measurement_summary_row, request_summary


class SGLangMeasurementError(RuntimeError):
    """Raised when the real SGLang backend cannot run."""


def _require_sglang() -> None:
    if importlib.util.find_spec("sglang") is None:
        raise SGLangMeasurementError(
            "SGLang is not installed in this environment. Install patched SGLang "
            "from third_party/sglang before running --backend sglang."
        )


def _server_base_url(config: Mapping[str, Any]) -> str:
    args = dict((config.get("sglang", {}) or {}).get("args", {}) or {})
    host = str(args.get("host", "127.0.0.1"))
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    port = str(args.get("port", "30000"))
    return "http://%s:%s" % (host, port)


def _health_url(config: Mapping[str, Any]) -> str:
    return _server_base_url(config) + "/health"


def _wait_for_health(url: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                if 200 <= response.status < 500:
                    return True
        except URLError:
            time.sleep(1)
    return False


def _post_json(url: str, payload: Mapping[str, Any], timeout_s: float) -> Tuple[int, Dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
            parsed = json.loads(raw) if raw else {}
            if not isinstance(parsed, dict):
                parsed = {"response": parsed}
            return int(response.status), parsed
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise SGLangMeasurementError(
            "SGLang request failed with HTTP %s: %s" % (exc.code, raw[-2000:])
        ) from exc
    except (OSError, URLError, json.JSONDecodeError) as exc:
        raise SGLangMeasurementError("SGLang request failed: %s" % exc) from exc


def _get_json(url: str, timeout_s: float) -> Tuple[int, Dict[str, Any]]:
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
            parsed = json.loads(raw) if raw else {}
            if not isinstance(parsed, dict):
                parsed = {"response": parsed}
            return int(response.status), parsed
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise SGLangMeasurementError(
            "SGLang metrics request failed with HTTP %s: %s" % (exc.code, raw[-2000:])
        ) from exc
    except (OSError, URLError, json.JSONDecodeError) as exc:
        raise SGLangMeasurementError("SGLang metrics request failed: %s" % exc) from exc


def _metrics_endpoint(measurement: Mapping[str, Any]) -> str | None:
    endpoint = measurement.get("sglang_metrics_endpoint", "/aether/metrics")
    if endpoint in (None, False):
        return None
    endpoint = str(endpoint)
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    return endpoint


def _read_sglang_metrics(
    base_url: str,
    measurement: Mapping[str, Any],
) -> Tuple[str, int, Dict[str, Any]] | None:
    endpoint = _metrics_endpoint(measurement)
    if endpoint is None:
        return None
    request_timeout = float(measurement.get("request_timeout_seconds", 120))
    status, payload = _get_json(base_url + endpoint, request_timeout)
    return endpoint, status, payload


def _snapshot_sglang_metrics(
    base_url: str,
    measurement: Mapping[str, Any],
) -> Dict[str, Any]:
    require_metrics = bool(measurement.get("require_sglang_metrics", False))
    try:
        snapshot = _read_sglang_metrics(base_url, measurement)
    except SGLangMeasurementError:
        if require_metrics:
            raise
        return {}
    if snapshot is None:
        return {}
    _, _, payload = snapshot
    if require_metrics and not bool(payload.get("enabled", False)):
        raise SGLangMeasurementError(
            "SGLang Aether metrics endpoint is present but disabled"
        )
    return payload


def _summary_delta(
    after: Mapping[str, Any],
    before: Mapping[str, Any],
) -> Dict[str, Any]:
    delta = dict(after)
    counter_fields = [
        "request_count",
        "prompt_tokens_total",
        "completion_tokens_total",
        "total_tokens_total",
    ]
    for field in counter_fields:
        delta[field] = max(0, as_int(after.get(field)) - as_int(before.get(field)))

    e2e_delta = max(
        0.0,
        as_float(after.get("e2e_latency_sum_s")) - as_float(before.get("e2e_latency_sum_s")),
    )
    delta["e2e_latency_sum_s"] = round(e2e_delta, 9)
    request_count = as_int(delta.get("request_count"))
    delta["e2e_latency_avg_s"] = round(e2e_delta / request_count, 9) if request_count else 0.0

    for field in ["ttft_avg_s", "tbt_avg_s"]:
        if request_count:
            delta[field] = as_float(after.get(field))
        else:
            delta[field] = 0.0
    if as_int(delta.get("request_count")):
        delta["last_request_unix_s"] = after.get("last_request_unix_s")
    else:
        delta["last_request_unix_s"] = None
    return delta


def _delta_sglang_metrics(
    payload: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    if not baseline:
        return dict(payload)

    payload_summary = (
        payload.get("summary", {}) if isinstance(payload.get("summary"), Mapping) else {}
    )
    baseline_summary = (
        baseline.get("summary", {}) if isinstance(baseline.get("summary"), Mapping) else {}
    )
    delta_summary = _summary_delta(payload_summary, baseline_summary)

    payload_events = payload.get("events", [])
    if not isinstance(payload_events, list):
        payload_events = []
    baseline_last_request = as_float(baseline_summary.get("last_request_unix_s"))
    if baseline_last_request:
        delta_events = [
            event
            for event in payload_events
            if isinstance(event, Mapping)
            and as_float(event.get("timestamp_unix_s")) > baseline_last_request
        ]
    else:
        request_count = as_int(delta_summary.get("request_count"))
        delta_events = payload_events[-request_count:] if request_count else []

    delta_payload = dict(payload)
    delta_payload["summary"] = delta_summary
    delta_payload["events"] = delta_events
    delta_payload["baseline_applied"] = True
    return delta_payload


def _extract_response_metrics(response: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract comparable token and latency metrics from SGLang or OpenAI output."""

    usage = response.get("usage")
    if isinstance(usage, dict):
        prompt_tokens = as_int(usage.get("prompt_tokens"))
        completion_tokens = as_int(usage.get("completion_tokens"))
        total_tokens = as_int(usage.get("total_tokens")) or prompt_tokens + completion_tokens
        return {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    responses: List[Mapping[str, Any]]
    if isinstance(response.get("response"), list):
        responses = [item for item in response["response"] if isinstance(item, dict)]
    else:
        responses = [response]

    prompt_tokens = 0
    generated_tokens = 0
    total_tokens = 0
    latencies = []
    ttfts = []
    tbts = []
    for item in responses:
        meta = item.get("meta_info", {})
        if not isinstance(meta, dict):
            meta = {}
        prompt_tokens += as_int(meta.get("prompt_tokens") or meta.get("input_tokens"))
        generated_tokens += as_int(
            meta.get("completion_tokens") or meta.get("output_tokens")
        )
        total_tokens += as_int(meta.get("total_tokens"))
        latency = as_float(meta.get("e2e_latency") or meta.get("latency"))
        if latency > 0:
            latencies.append(latency)
        ttft = as_float(
            meta.get("ttft")
            or meta.get("time_to_first_token")
            or meta.get("first_token_latency")
            or meta.get("first_token_latency_s")
        )
        if ttft > 0:
            ttfts.append(ttft)
        tbt = as_float(
            meta.get("tbt")
            or meta.get("inter_token_latency")
            or meta.get("decode_token_latency")
        )
        if tbt > 0:
            tbts.append(tbt)

    if total_tokens == 0:
        total_tokens = prompt_tokens + generated_tokens

    metrics: Dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_tokens": total_tokens,
    }
    if latencies:
        metrics["response_latency_s"] = round(sum(latencies), 6)
    if ttfts:
        metrics["ttft_ms"] = round(1000.0 * sum(ttfts) / len(ttfts), 6)
    if tbts:
        metrics["tbt_ms"] = round(1000.0 * sum(tbts) / len(tbts), 6)
    return metrics


def _configured_requests(measurement: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    requests = (
        measurement.get("requests")
        or measurement.get("sglang_requests")
        or measurement.get("openai_requests")
        or []
    )
    if not isinstance(requests, list):
        raise SGLangMeasurementError("measurement.requests must be a list")
    for request in requests:
        if not isinstance(request, dict):
            raise SGLangMeasurementError("each measurement request must be a mapping")
    return requests


def _run_configured_requests(
    base_url: str,
    measurement: Mapping[str, Any],
    events: List[Dict[str, Any]],
) -> None:
    request_timeout = float(measurement.get("request_timeout_seconds", 120))
    for index, item in enumerate(_configured_requests(measurement)):
        endpoint = str(item.get("endpoint", "/generate"))
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        payload = item.get("payload", {})
        if not isinstance(payload, dict):
            raise SGLangMeasurementError(
                "measurement.requests[%s].payload must be a mapping" % index
            )
        started = time.perf_counter()
        status, response = _post_json(base_url + endpoint, payload, request_timeout)
        latency = time.perf_counter() - started
        metrics = _extract_response_metrics(response)
        event = {
            "type": "request",
            "backend": "sglang",
            "request_id": index,
            "endpoint": endpoint,
            "status": status,
            "latency_s": round(latency, 6),
            "response_keys": sorted(response.keys()),
        }
        event.update(metrics)
        events.append(event)


def _collect_sglang_metrics(
    base_url: str,
    measurement: Mapping[str, Any],
    events: List[Dict[str, Any]],
    baseline_metrics: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    require_metrics = bool(measurement.get("require_sglang_metrics", False))
    try:
        snapshot = _read_sglang_metrics(base_url, measurement)
    except SGLangMeasurementError:
        if require_metrics:
            raise
        events.append(
            {
                "type": "sglang_aether_metrics",
                "backend": "sglang",
                "endpoint": endpoint,
                "ok": False,
            }
        )
        return {}
    if snapshot is None:
        return {}
    endpoint, status, payload = snapshot
    payload = _delta_sglang_metrics(payload, baseline_metrics)

    event = {
        "type": "sglang_aether_metrics",
        "backend": "sglang",
        "endpoint": endpoint,
        "status": status,
        "ok": bool(payload.get("enabled", False)),
        "payload": payload,
    }
    events.append(event)
    if require_metrics and not event["ok"]:
        raise SGLangMeasurementError(
            "SGLang Aether metrics endpoint is present but disabled"
        )
    return payload


def _request_summary(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    return request_summary(events)


def run_sglang(config: Mapping[str, Any], out_dir: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    _require_sglang()
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "sglang.log"
    measurement = dict(config.get("measurement", {}) or {})
    duration = float(measurement.get("duration_seconds", 60))
    timeout = float(measurement.get("health_timeout_seconds", 120))
    args = render_sglang_args(config)
    command = [sys.executable, "-m", "sglang.launch_server"] + args

    events: List[Dict[str, Any]] = [{"type": "launch", "command": command}]
    profiler = None
    collection_elapsed = duration
    patched_metrics: Dict[str, Any] = {}
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
        try:
            health_url = _health_url(config)
            if not _wait_for_health(health_url, timeout):
                raise SGLangMeasurementError(
                    "SGLang did not become healthy at %s" % health_url
                )
            events.append({"type": "health", "url": health_url, "ok": True})
            try:
                profiler = build_hardware_profiler(
                    config,
                    events,
                    backend_label="sglang",
                    process_id=process.pid,
                )
                events.append(
                    {
                        "type": "hardware_profiler",
                        "backend": "sglang",
                        "hardware_backend": profiler.hardware_backend,
                        "ok": True,
                    }
                )
                profiler.start()
            except HardwareProfilerError as exc:
                raise SGLangMeasurementError(str(exc)) from exc
            metrics_baseline = _snapshot_sglang_metrics(
                _server_base_url(config), measurement
            )
            collection_start = time.monotonic()
            configured_requests = _configured_requests(measurement)
            workload_command = measurement.get("workload_command")
            if configured_requests:
                _run_configured_requests(_server_base_url(config), measurement, events)
            elif workload_command:
                if not isinstance(workload_command, list):
                    raise SGLangMeasurementError(
                        "measurement.workload_command must be a list of command arguments"
                    )
                workload = subprocess.run(
                    workload_command, check=False, capture_output=True, text=True
                )
                events.append(
                    {
                        "type": "workload",
                        "returncode": workload.returncode,
                        "stdout": workload.stdout[-2000:],
                        "stderr": workload.stderr[-2000:],
                    }
                )
            else:
                time.sleep(duration)
                events.append({"type": "idle_collection", "duration_seconds": duration})
            patched_metrics = _collect_sglang_metrics(
                _server_base_url(config), measurement, events, metrics_baseline
            )
            collection_elapsed = max(time.monotonic() - collection_start, 0.000001)
        finally:
            profiler_error = None
            if profiler is not None:
                try:
                    profiler.stop()
                except HardwareProfilerError as exc:
                    profiler_error = exc
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
            if profiler_error is not None:
                raise SGLangMeasurementError(str(profiler_error)) from profiler_error

    row = measurement_summary_row(
        config,
        events,
        backend="sglang",
        scenario_id="sglang-real",
        hardware="real-sglang",
        routing_policy="real",
        routing_pool="sglang",
        scheduling_policy="real",
        scheduling_action="observed",
        fallback_elapsed_seconds=collection_elapsed,
        patched_metrics=patched_metrics,
    )
    write_csv([row], str(output_dir / "summary.csv"))
    write_jsonl(events, str(output_dir / "events.jsonl"))
    with (output_dir / "command.json").open("w", encoding="utf-8") as handle:
        json.dump({"command": command}, handle, indent=2)
    if patched_metrics:
        with (output_dir / "sglang_aether_metrics.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(patched_metrics, handle, indent=2, sort_keys=True)
    return row, events
