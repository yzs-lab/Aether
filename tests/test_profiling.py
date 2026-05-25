import os

from aether.measurement.profiling import (
    CpuProfiler,
    MockGpuProfiler,
    NvmlGpuProfiler,
    build_hardware_profiler,
    select_hardware_backend,
)
from aether.measurement.summary import measurement_summary_row, power_summary


def test_cpu_profiler_emits_process_power_event():
    events = []
    profiler = CpuProfiler(
        1,
        events,
        backend_label="test",
        process_id=os.getpid(),
        estimated_power_w=12.5,
    )
    profiler.sample_once(0.0)

    assert events[0]["type"] == "power"
    assert events[0]["hardware_backend"] == "cpu"
    assert events[0]["power_w"] == 12.5
    assert events[0]["power_estimated"] is True
    assert "cpu_user_seconds" in events[0]


def test_mock_gpu_profiler_is_deterministic():
    events = []
    profiler = MockGpuProfiler(
        1,
        events,
        backend_label="test",
        gpu_count=2,
        base_power_w=200,
    )
    profiler.sample_once(0.0)
    profiler.sample_once(1.0)

    assert [event["power_w"] for event in events] == [200.0, 205.0, 211.0, 216.0]
    assert {event["hardware_backend"] for event in events} == {"mock-gpu"}


class _FakeMemory:
    used = 123
    total = 456


class _FakeUtilization:
    gpu = 77
    memory = 33


class _FakeNvml:
    NVML_CLOCK_SM = 1
    NVML_CLOCK_MEM = 2
    NVML_TEMPERATURE_GPU = 3

    def __init__(self):
        self.initialized = False
        self.shutdown = False

    def nvmlInit(self):
        self.initialized = True

    def nvmlShutdown(self):
        self.shutdown = True

    def nvmlDeviceGetCount(self):
        return 1

    def nvmlDeviceGetHandleByIndex(self, index):
        return "gpu-%s" % index

    def nvmlDeviceGetPowerUsage(self, handle):
        return 321000

    def nvmlDeviceGetMemoryInfo(self, handle):
        return _FakeMemory()

    def nvmlDeviceGetUtilizationRates(self, handle):
        return _FakeUtilization()

    def nvmlDeviceGetClockInfo(self, handle, clock):
        return 1500 if clock == self.NVML_CLOCK_SM else 2500

    def nvmlDeviceGetTemperature(self, handle, sensor):
        return 61


def test_nvml_profiler_normalizes_gpu_api_values():
    fake_nvml = _FakeNvml()
    events = []
    profiler = NvmlGpuProfiler(
        1,
        events,
        backend_label="test",
        nvml_module=fake_nvml,
    )
    profiler.start()
    profiler.stop()

    assert fake_nvml.initialized is True
    assert fake_nvml.shutdown is True
    event = events[0]
    assert event["hardware_backend"] == "gpu-nvml"
    assert event["power_w"] == 321.0
    assert event["memory_used_bytes"] == 123
    assert event["gpu_utilization_pct"] == 77
    assert event["sm_clock_mhz"] == 1500
    assert event["temperature_c"] == 61


def test_build_hardware_profiler_selects_cpu_and_legacy_nvml_modes():
    assert select_hardware_backend({"hardware_backend": "cpu"}) == "cpu"
    assert select_hardware_backend({"nvml_enabled": True}) == "gpu-nvml"
    assert select_hardware_backend({"nvml_enabled": False}) == "none"

    profiler = build_hardware_profiler(
        {
            "measurement": {"hardware_backend": "cpu", "cpu_power_w": 9},
        },
        [],
        backend_label="test",
        process_id=os.getpid(),
    )
    assert profiler.hardware_backend == "cpu"


def test_common_measurement_summary_uses_power_and_patched_metrics():
    events = [
        {"type": "power", "time_s": 0.0, "power_w": 10.0},
        {"type": "power", "time_s": 1.0, "power_w": 20.0},
        {"type": "request", "prompt_tokens": 1, "generated_tokens": 1, "latency_s": 0.5},
    ]
    elapsed, avg_power, energy = power_summary(events, 1.0)
    assert elapsed == 1.0
    assert avg_power == 15.0
    assert energy == 15.0

    row = measurement_summary_row(
        {"experiment": {"name": "x"}, "model": {"name": "m", "quality_score": 0.5}},
        events,
        backend="sglang",
        scenario_id="s",
        hardware="h",
        routing_policy="r",
        routing_pool="p",
        scheduling_policy="sched",
        scheduling_action="observed",
        fallback_elapsed_seconds=1.0,
        patched_metrics={
            "summary": {
                "prompt_tokens_total": 7,
                "completion_tokens_total": 3,
                "e2e_latency_sum_s": 2.0,
            }
        },
    )
    assert row["prompt_tokens"] == 7
    assert row["generated_tokens"] == 3
    assert row["elapsed_seconds"] == 2.0
    assert row["tokens_per_joule"] == 0.1
    assert row["quality_normalized_ipw"] == 0.05
