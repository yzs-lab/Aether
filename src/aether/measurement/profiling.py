"""Hardware profiling backends for Aether measurement runs."""

from __future__ import annotations

import importlib
import os
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Protocol


class HardwareProfilerError(RuntimeError):
    """Raised when a selected hardware profiler cannot run."""


class HardwareProfiler(Protocol):
    """Minimal profiler lifecycle used by serving backends."""

    hardware_backend: str

    def start(self) -> None:
        """Begin sampling."""

    def stop(self) -> None:
        """Stop sampling and surface deferred errors."""


class NullProfiler:
    """No-op profiler for explicitly unprofiled runs."""

    hardware_backend = "none"

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


class _ThreadedProfiler:
    def __init__(self, sample_hz: int, events: List[Dict[str, Any]]):
        self.sample_hz = max(1, int(sample_hz))
        self.events = events
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.error: Optional[BaseException] = None
        self._started_at = 0.0

    def start(self) -> None:
        self._started_at = time.time()
        self.thread = threading.Thread(target=self._run, name="aether-hardware-profiler")
        self.thread.daemon = True
        self.thread.start()
        time.sleep(0.05)
        if self.error is not None:
            raise HardwareProfilerError(str(self.error))

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        if self.error is not None:
            raise HardwareProfilerError(str(self.error))

    def _run(self) -> None:
        interval = 1.0 / float(self.sample_hz)
        try:
            while not self.stop_event.is_set():
                self.sample_once(round(time.time() - self._started_at, 6))
                self.stop_event.wait(interval)
        except BaseException as exc:
            self.error = exc

    def sample_once(self, time_s: float) -> None:
        raise NotImplementedError


def _read_linux_process_times(pid: int) -> Dict[str, float]:
    stat_path = "/proc/%s/stat" % pid
    with open(stat_path, "r", encoding="utf-8") as handle:
        raw = handle.read()
    # comm can contain spaces inside parentheses; split after the final ")".
    fields = raw.rsplit(")", 1)[1].strip().split()
    clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    return {
        "user_seconds": float(fields[11]) / float(clock_ticks),
        "system_seconds": float(fields[12]) / float(clock_ticks),
    }


def _read_linux_rss_bytes(pid: int) -> int:
    statm_path = "/proc/%s/statm" % pid
    with open(statm_path, "r", encoding="utf-8") as handle:
        fields = handle.read().strip().split()
    resident_pages = int(fields[1]) if len(fields) > 1 else 0
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


class CpuProfiler(_ThreadedProfiler):
    """CPU process profiler with an explicit power estimate."""

    hardware_backend = "cpu"

    def __init__(
        self,
        sample_hz: int,
        events: List[Dict[str, Any]],
        *,
        backend_label: str,
        process_id: Optional[int],
        estimated_power_w: float,
    ):
        super().__init__(sample_hz, events)
        self.backend_label = backend_label
        self.process_id = process_id or os.getpid()
        self.estimated_power_w = float(estimated_power_w)
        self._last_cpu_seconds: Optional[float] = None
        self._last_sample_time: Optional[float] = None

    def sample_once(self, time_s: float) -> None:
        user_seconds = 0.0
        system_seconds = 0.0
        rss_bytes = 0
        try:
            times = _read_linux_process_times(self.process_id)
            user_seconds = times["user_seconds"]
            system_seconds = times["system_seconds"]
            rss_bytes = _read_linux_rss_bytes(self.process_id)
        except (FileNotFoundError, OSError, ValueError, IndexError):
            # Non-Linux hosts still get a deterministic CPU profiler event.
            rss_bytes = 0

        cpu_seconds = user_seconds + system_seconds
        cpu_percent = 0.0
        if self._last_cpu_seconds is not None and self._last_sample_time is not None:
            elapsed = max(0.000001, time_s - self._last_sample_time)
            cpu_count = max(1, os.cpu_count() or 1)
            cpu_percent = max(
                0.0,
                min(100.0 * cpu_count, 100.0 * (cpu_seconds - self._last_cpu_seconds) / elapsed),
            )
        self._last_cpu_seconds = cpu_seconds
        self._last_sample_time = time_s

        self.events.append(
            {
                "type": "power",
                "backend": self.backend_label,
                "hardware_backend": self.hardware_backend,
                "source": "cpu_process_estimate",
                "time_s": round(time_s, 6),
                "process_id": self.process_id,
                "cpu_count": os.cpu_count() or 1,
                "cpu_user_seconds": round(user_seconds, 6),
                "cpu_system_seconds": round(system_seconds, 6),
                "process_cpu_percent": round(cpu_percent, 6),
                "rss_bytes": int(rss_bytes),
                "power_w": round(self.estimated_power_w, 6),
                "power_estimated": True,
            }
        )


class NvmlGpuProfiler(_ThreadedProfiler):
    """NVIDIA GPU profiler backed by optional pynvml."""

    hardware_backend = "gpu-nvml"

    def __init__(
        self,
        sample_hz: int,
        events: List[Dict[str, Any]],
        *,
        backend_label: str,
        nvml_module: Any = None,
    ):
        super().__init__(sample_hz, events)
        self.backend_label = backend_label
        self.nvml = nvml_module
        self._initialized = False

    def start(self) -> None:
        if self.nvml is None:
            try:
                self.nvml = importlib.import_module("pynvml")
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise HardwareProfilerError(
                    "pynvml is not installed. Install Aether with the sglang extra "
                    "and run on a host with NVIDIA drivers for gpu-nvml profiling."
                ) from exc
        self.nvml.nvmlInit()
        self._initialized = True
        super().start()

    def stop(self) -> None:
        try:
            super().stop()
        finally:
            if self._initialized:
                self.nvml.nvmlShutdown()
                self._initialized = False

    def sample_once(self, time_s: float) -> None:
        device_count = int(self.nvml.nvmlDeviceGetCount())
        for gpu_id in range(device_count):
            handle = self.nvml.nvmlDeviceGetHandleByIndex(gpu_id)
            memory = self.nvml.nvmlDeviceGetMemoryInfo(handle)
            event = {
                "type": "power",
                "backend": self.backend_label,
                "hardware_backend": self.hardware_backend,
                "source": "nvml",
                "time_s": round(time_s, 6),
                "gpu_id": gpu_id,
                "power_w": round(float(self.nvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0, 6),
                "memory_used_bytes": int(memory.used),
                "memory_total_bytes": int(memory.total),
                "power_estimated": False,
            }
            if hasattr(self.nvml, "nvmlDeviceGetUtilizationRates"):
                utilization = self.nvml.nvmlDeviceGetUtilizationRates(handle)
                event["gpu_utilization_pct"] = int(getattr(utilization, "gpu", 0))
                event["memory_utilization_pct"] = int(getattr(utilization, "memory", 0))
            if hasattr(self.nvml, "nvmlDeviceGetClockInfo"):
                if hasattr(self.nvml, "NVML_CLOCK_SM"):
                    event["sm_clock_mhz"] = int(
                        self.nvml.nvmlDeviceGetClockInfo(handle, self.nvml.NVML_CLOCK_SM)
                    )
                if hasattr(self.nvml, "NVML_CLOCK_MEM"):
                    event["memory_clock_mhz"] = int(
                        self.nvml.nvmlDeviceGetClockInfo(handle, self.nvml.NVML_CLOCK_MEM)
                    )
            if hasattr(self.nvml, "nvmlDeviceGetTemperature") and hasattr(
                self.nvml, "NVML_TEMPERATURE_GPU"
            ):
                event["temperature_c"] = int(
                    self.nvml.nvmlDeviceGetTemperature(
                        handle, self.nvml.NVML_TEMPERATURE_GPU
                    )
                )
            self.events.append(event)


class MockGpuProfiler(_ThreadedProfiler):
    """Deterministic GPU-shaped profiler for tests and dry runs."""

    hardware_backend = "mock-gpu"

    def __init__(
        self,
        sample_hz: int,
        events: List[Dict[str, Any]],
        *,
        backend_label: str,
        gpu_count: int = 1,
        base_power_w: float = 250.0,
    ):
        super().__init__(sample_hz, events)
        self.backend_label = backend_label
        self.gpu_count = max(1, int(gpu_count))
        self.base_power_w = float(base_power_w)
        self._sample_index = 0

    def sample_once(self, time_s: float) -> None:
        for gpu_id in range(self.gpu_count):
            power_w = self.base_power_w + float((self._sample_index * 11 + gpu_id * 5) % 17)
            self.events.append(
                {
                    "type": "power",
                    "backend": self.backend_label,
                    "hardware_backend": self.hardware_backend,
                    "source": "mock_gpu",
                    "time_s": round(time_s, 6),
                    "gpu_id": gpu_id,
                    "power_w": round(power_w, 6),
                    "memory_used_bytes": 2_000_000_000 + gpu_id * 100_000_000,
                    "memory_total_bytes": 24_000_000_000,
                    "gpu_utilization_pct": 42 + gpu_id,
                    "memory_utilization_pct": 25 + gpu_id,
                    "power_estimated": False,
                }
            )
        self._sample_index += 1


def _first_hardware_profile(config: Mapping[str, Any]) -> Mapping[str, Any]:
    simulation = config.get("simulation", {}) or {}
    if not isinstance(simulation, Mapping):
        return {}
    profiles = simulation.get("hardware_profiles", []) or []
    if isinstance(profiles, list) and profiles and isinstance(profiles[0], Mapping):
        return profiles[0]
    return {}


def _cpu_power_estimate(config: Mapping[str, Any], measurement: Mapping[str, Any]) -> float:
    if measurement.get("cpu_power_w") is not None:
        return float(measurement["cpu_power_w"])
    hardware = _first_hardware_profile(config)
    if hardware.get("tdp_w") is not None:
        fraction = float(measurement.get("cpu_power_fraction", 0.5))
        return max(1.0, float(hardware["tdp_w"]) * fraction)
    return float(measurement.get("default_cpu_power_w", 35.0))


def select_hardware_backend(measurement: Mapping[str, Any]) -> str:
    configured = measurement.get("hardware_backend")
    if configured:
        return str(configured)
    if measurement.get("nvml_enabled") is True:
        return "gpu-nvml"
    if measurement.get("nvml_enabled") is False:
        return "none"
    return "none"


def build_hardware_profiler(
    config: Mapping[str, Any],
    events: List[Dict[str, Any]],
    *,
    backend_label: str,
    process_id: Optional[int] = None,
) -> HardwareProfiler:
    measurement = config.get("measurement", {}) or {}
    if not isinstance(measurement, Mapping):
        measurement = {}
    sample_hz = int(measurement.get("sample_hz", 100))
    backend = select_hardware_backend(measurement)
    if backend in {"none", "off", "disabled"}:
        return NullProfiler()
    if backend == "cpu":
        return CpuProfiler(
            sample_hz,
            events,
            backend_label=backend_label,
            process_id=process_id,
            estimated_power_w=_cpu_power_estimate(config, measurement),
        )
    if backend in {"gpu-nvml", "nvml", "gpu"}:
        return NvmlGpuProfiler(sample_hz, events, backend_label=backend_label)
    if backend in {"mock-gpu", "mock_gpu"}:
        return MockGpuProfiler(
            sample_hz,
            events,
            backend_label=backend_label,
            gpu_count=int(measurement.get("mock_gpu_count", 1)),
            base_power_w=float(measurement.get("mock_gpu_base_power_w", 250.0)),
        )
    raise HardwareProfilerError("unknown measurement.hardware_backend: %s" % backend)
