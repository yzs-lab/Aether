"""Real SGLang launcher and measurement backend.

This module is intentionally optional at runtime. It imports GPU-specific
libraries only inside functions so CPU simulation and tests remain clean.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .. import formulas
from ..config import render_sglang_args
from ..results import write_csv, write_jsonl


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


class _NvmlSampler:
    def __init__(self, sample_hz: int, events: List[Dict[str, Any]]):
        self.sample_hz = max(1, int(sample_hz))
        self.events = events
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.error: Optional[BaseException] = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="aether-nvml-sampler")
        self.thread.daemon = True
        self.thread.start()
        time.sleep(0.05)
        if self.error is not None:
            raise SGLangMeasurementError(str(self.error))

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        if self.error is not None:
            raise SGLangMeasurementError(str(self.error))

    def _run(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            interval = 1.0 / float(self.sample_hz)
            start = time.time()
            while not self.stop_event.is_set():
                ts = round(time.time() - start, 6)
                for gpu_id in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self.events.append(
                        {
                            "type": "power",
                            "backend": "sglang",
                            "time_s": ts,
                            "gpu_id": gpu_id,
                            "power_w": power_w,
                            "memory_used_bytes": int(memory.used),
                            "memory_total_bytes": int(memory.total),
                        }
                    )
                self.stop_event.wait(interval)
            pynvml.nvmlShutdown()
        except BaseException as exc:  # pragma: no cover - requires GPU/NVML
            self.error = exc


def _power_summary(events: List[Dict[str, Any]], fallback_duration: float) -> Tuple[float, float, float]:
    by_ts: Dict[float, float] = {}
    for event in events:
        if event.get("type") == "power":
            by_ts.setdefault(float(event.get("time_s", 0.0)), 0.0)
            by_ts[float(event.get("time_s", 0.0))] += float(event.get("power_w", 0.0))
    if not by_ts:
        return fallback_duration, 0.0, 0.0
    elapsed = max(by_ts) if max(by_ts) > 0 else fallback_duration
    avg_power = sum(by_ts.values()) / len(by_ts)
    return elapsed, avg_power, avg_power * elapsed


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


def _as_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_response_metrics(response: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract comparable token and latency metrics from SGLang or OpenAI output."""

    usage = response.get("usage")
    if isinstance(usage, dict):
        prompt_tokens = _as_int(usage.get("prompt_tokens"))
        completion_tokens = _as_int(usage.get("completion_tokens"))
        total_tokens = _as_int(usage.get("total_tokens")) or prompt_tokens + completion_tokens
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
        prompt_tokens += _as_int(meta.get("prompt_tokens") or meta.get("input_tokens"))
        generated_tokens += _as_int(
            meta.get("completion_tokens") or meta.get("output_tokens")
        )
        total_tokens += _as_int(meta.get("total_tokens"))
        latency = _as_float(meta.get("e2e_latency") or meta.get("latency"))
        if latency > 0:
            latencies.append(latency)
        ttft = _as_float(
            meta.get("ttft")
            or meta.get("time_to_first_token")
            or meta.get("first_token_latency")
            or meta.get("first_token_latency_s")
        )
        if ttft > 0:
            ttfts.append(ttft)
        tbt = _as_float(
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
) -> Dict[str, Any]:
    endpoint = measurement.get("sglang_metrics_endpoint", "/aether/metrics")
    if endpoint in (None, False):
        return {}
    endpoint = str(endpoint)
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    require_metrics = bool(measurement.get("require_sglang_metrics", False))
    request_timeout = float(measurement.get("request_timeout_seconds", 120))
    try:
        status, payload = _get_json(base_url + endpoint, request_timeout)
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
    request_events = [event for event in events if event.get("type") == "request"]
    prompt_tokens = sum(_as_int(event.get("prompt_tokens")) for event in request_events)
    generated_tokens = sum(
        _as_int(event.get("generated_tokens")) for event in request_events
    )
    request_latency = sum(_as_float(event.get("latency_s")) for event in request_events)
    ttft_values = [
        _as_float(event.get("ttft_ms"))
        for event in request_events
        if _as_float(event.get("ttft_ms")) > 0
    ]
    tbt_values = [
        _as_float(event.get("tbt_ms"))
        for event in request_events
        if _as_float(event.get("tbt_ms")) > 0
    ]
    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "request_latency_s": request_latency,
        "ttft_ms": round(sum(ttft_values) / len(ttft_values), 6)
        if ttft_values
        else 0.0,
        "tbt_ms": round(sum(tbt_values) / len(tbt_values), 6) if tbt_values else 0.0,
    }


def run_sglang(config: Mapping[str, Any], out_dir: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    _require_sglang()
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "sglang.log"
    measurement = dict(config.get("measurement", {}) or {})
    duration = float(measurement.get("duration_seconds", 60))
    timeout = float(measurement.get("health_timeout_seconds", 120))
    sample_hz = int(measurement.get("sample_hz", 100))
    nvml_enabled = bool(measurement.get("nvml_enabled", True))
    args = render_sglang_args(config)
    command = [sys.executable, "-m", "sglang.launch_server"] + args

    events: List[Dict[str, Any]] = [{"type": "launch", "command": command}]
    sampler = _NvmlSampler(sample_hz, events) if nvml_enabled else None
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
            if sampler is not None:
                sampler.start()
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
                _server_base_url(config), measurement, events
            )
            collection_elapsed = max(time.monotonic() - collection_start, 0.000001)
        finally:
            sampler_error = None
            if sampler is not None:
                try:
                    sampler.stop()
                except SGLangMeasurementError as exc:
                    sampler_error = exc
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
            if sampler_error is not None:
                raise sampler_error

    requests = _request_summary(events)
    patched_summary = (
        patched_metrics.get("summary", {})
        if isinstance(patched_metrics.get("summary"), dict)
        else {}
    )
    patched_prompt_tokens = _as_int(patched_summary.get("prompt_tokens_total"))
    patched_generated_tokens = _as_int(patched_summary.get("completion_tokens_total"))
    if patched_prompt_tokens:
        requests["prompt_tokens"] = patched_prompt_tokens
    if patched_generated_tokens:
        requests["generated_tokens"] = patched_generated_tokens
    fallback_elapsed = max(
        collection_elapsed,
        requests["request_latency_s"],
        duration if not requests["generated_tokens"] else 0.000001,
    )
    elapsed, avg_power, energy = _power_summary(events, fallback_elapsed)
    generated_tokens = int(requests["generated_tokens"])
    tokens_per_second = formulas.safe_div(generated_tokens, elapsed)
    tokens_per_watt = formulas.safe_div(tokens_per_second, avg_power)
    tokens_per_joule = formulas.safe_div(generated_tokens, energy)
    quality = float((config.get("model", {}) or {}).get("quality_score", 1.0))
    row = {
        "experiment": (config.get("experiment", {}) or {}).get("name", "aether-sglang"),
        "backend": "sglang",
        "scenario_id": "sglang-real",
        "model": (config.get("model", {}) or {}).get("name", "unknown-model"),
        "hardware": "real-sglang",
        "routing_policy": "real",
        "routing_pool": "sglang",
        "scheduling_policy": "real",
        "scheduling_action": "observed",
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
