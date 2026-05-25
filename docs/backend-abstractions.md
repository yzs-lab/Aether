# Backend Abstraction Plan

Aether separates serving backends from hardware profiling backends.

Serving backends answer the question: how do we launch or exercise the model
server? Hardware profiling backends answer the question: what can the current
machine tell us about power, memory, clocks, and utilization while the serving
backend is running?

## Serving Backends

The current serving backends are:

- `mock`: deterministic CPU-only fixture for result-writer and collector tests.
- `sglang`: real SGLang process launcher, health polling, request runner, and
  patched `/aether/metrics` collector.

Future serving backends should provide:

- launch lifecycle: start, health check, stop, log capture
- workload execution: configured HTTP requests or external workload command
- backend-native metrics: SGLang `/aether/metrics`, vLLM metrics, or log readers
- normalized request events: prompt tokens, generated tokens, TTFT, TBT, latency

Serving backends should not implement hardware-specific power APIs directly.

## Hardware Profiling Backends

Hardware profiling backends provide normalized `power` events and optional
hardware-specific fields. Every profiler should expose:

- `start()`: begin sampling
- `stop()`: stop sampling and surface deferred errors
- emitted events: `type=power`, `hardware_backend`, `time_s`, `power_w`, and
  backend-specific fields

### CPU Profiler

The CPU profiler is the default for real SGLang CPU CI. It samples the launched
server process with standard library and Linux `/proc` data where available:

- process CPU user/system seconds
- process CPU percent between samples
- RSS bytes
- host CPU count
- estimated package power in watts

GitHub-hosted runners do not expose RAPL package energy reliably, so CPU power
is explicitly marked as an estimate. The estimate can be configured with
`measurement.cpu_power_w`; otherwise Aether falls back to a conservative value.
This still validates the real measurement pipeline on CPU: launch, workload,
sampling cadence, normalized power events, summary writing, and patched SGLang
metrics collection.

### GPU NVML Profiler

The GPU profiler uses `pynvml` only when selected. It must remain optional so
imports, simulation, mock measurement, and CPU CI do not require CUDA or NVML.
When available, it samples:

- power draw watts
- memory used/total bytes
- GPU and memory utilization percent
- SM and memory clocks when exposed
- GPU temperature when exposed

GPU tests should mock the NVML API and assert Aether converts hardware-specific
values into normalized power events. Real GPU experiments are run only on Linux
hosts with NVIDIA drivers and NVML.

### Mock GPU Profiler

The mock GPU profiler is for unit tests and future dry runs. It emits
deterministic GPU-shaped power events without importing NVML.

## Common Measurement Logic

The common measurement layer owns:

- request-event summarization
- power-event summarization across one or more devices
- patched backend metric overrides where appropriate
- normalized `summary.csv` rows
- `events.jsonl` writing

This keeps CPU, GPU, mock GPU, and future hardware profilers interchangeable.

## Backend-Specific Logic

Backend-specific modules should keep only what is unique:

- `measurement/mock.py`: deterministic synthetic events
- `measurement/sglang.py`: SGLang command rendering, process lifecycle,
  health polling, HTTP request execution, `/aether/metrics` baseline/delta
  collection
- future vLLM/SGLang variants: server-specific launch flags and metric readers

## Simulator Abstraction

The current GreenServe simulator remains formula-led and deterministic. It now
has an explicit simulation backend class so future simulator variants can
override scenario expansion, hardware models, or scheduling equations without
rewriting the CLI.

Use the default GreenServe simulation backend unless a new design doc explains
why a new simulator is needed.
