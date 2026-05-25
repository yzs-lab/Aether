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
from urllib.error import URLError
from urllib.request import urlopen

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


def _health_url(config: Mapping[str, Any]) -> str:
    args = dict((config.get("sglang", {}) or {}).get("args", {}) or {})
    host = str(args.get("host", "127.0.0.1"))
    port = str(args.get("port", "30000"))
    return "http://%s:%s/health" % (host, port)


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
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
        try:
            health_url = _health_url(config)
            if not _wait_for_health(health_url, timeout):
                raise SGLangMeasurementError("SGLang did not become healthy at %s" % health_url)
            events.append({"type": "health", "url": health_url, "ok": True})
            if sampler is not None:
                sampler.start()
            workload_command = measurement.get("workload_command")
            if workload_command:
                if not isinstance(workload_command, list):
                    raise SGLangMeasurementError("measurement.workload_command must be a list of command arguments")
                workload = subprocess.run(workload_command, check=False, capture_output=True, text=True)
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

    elapsed, avg_power, energy = _power_summary(events, duration)
    row = {
        "experiment": (config.get("experiment", {}) or {}).get("name", "aether-sglang"),
        "backend": "sglang",
        "scenario_id": "sglang-real",
        "model": (config.get("model", {}) or {}).get("name", "unknown-model"),
        "hardware": "real-gpu",
        "routing_policy": "real",
        "routing_pool": "sglang",
        "scheduling_policy": "real",
        "scheduling_action": "observed",
        "elapsed_seconds": round(elapsed, 6),
        "energy_j": round(energy, 6),
        "avg_power_w": round(avg_power, 6),
    }
    write_csv([row], str(output_dir / "summary.csv"))
    write_jsonl(events, str(output_dir / "events.jsonl"))
    with (output_dir / "command.json").open("w", encoding="utf-8") as handle:
        json.dump({"command": command}, handle, indent=2)
    return row, events
