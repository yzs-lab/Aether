# Aether Architecture

Aether has three layers:

1. Config and CLI: load YAML, render SGLang args, run commands, and write
   normalized outputs.
2. Simulation: compute GreenServe/IPW metrics from a pluggable simulation
   backend. The default backend is the formula-led GreenServe simulator.
3. Measurement: collect normalized data from serving backends and hardware
   profiling backends.

Real measurement is isolated from simulation and mock validation. The SGLang
backend imports optional dependencies only inside the backend code path.
Hardware profilers import optional hardware APIs only when selected.

## Data Flow

```text
YAML config
  -> config loader
  -> simulate: sweep expander -> formula engine -> summary rows -> CSV
  -> launch mock: synthetic events -> normalized summary/events -> CSV/JSONL
  -> launch sglang:
       SGLang process launcher + workload runner
       + selected hardware profiler
       + patched metric reader
       -> normalized summary/events
```

## Backend Boundaries

Serving backends launch and exercise model servers. The current serving
backends are `mock` and `sglang`.

Hardware profiling backends sample the machine while a serving backend runs.
The current hardware profilers are:

- `cpu`: process CPU time, RSS, and explicit estimated CPU power
- `gpu-nvml`: NVIDIA power, memory, utilization, clocks, and temperature
- `mock-gpu`: deterministic GPU-shaped samples for tests
- `none`: no hardware sampling

The common measurement summary layer consumes normalized request events and
`power` events from any hardware profiler. This keeps energy/IPW row generation
shared across CPU, GPU, mock, and future profilers.

## Normalized Outputs

All execution paths write the same summary fields where possible:

- scenario identity: experiment, backend, scenario_id
- configuration: model, hardware, context length, compression ratio, routing
  policy, scheduling policy
- capacity: KV bytes per token, effective KV bytes, max in-flight sequences
- timing: prefill seconds, decode seconds, elapsed seconds, TTFT/TBT estimates
- energy: joules, average watts, tokens/sec, tokens/watt, tokens/joule
- scheduling: swap bytes, recompute tokens, selected scheduling action
- quality: quality score and quality-normalized IPW

## SGLang Integration

The submodule is pinned to SGLang `v0.5.12`. The first patch series is
observability-only: it adds or documents hooks for power, token, KV, swap, and
preemption metrics without changing scheduling behavior.

See `docs/backend-abstractions.md` for the extension plan and interface split.
