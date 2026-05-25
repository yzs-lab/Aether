# Aether Architecture

Aether has three layers:

1. Config and CLI: load YAML, render SGLang args, run commands, and write
   normalized outputs.
2. Simulation: compute GreenServe/IPW metrics from reconstructed formulas and
   scenario sweeps.
3. Measurement: collect normalized data from either deterministic CPU mock
   fixtures or future real SGLang/GPU runs.

Real measurement is isolated from simulation and mock validation. The SGLang
backend imports optional dependencies only inside the backend code path and
fails with setup guidance if SGLang, CUDA, or NVML are unavailable.

## Data Flow

```text
YAML config
  -> config loader
  -> simulate: sweep expander -> formula engine -> summary rows -> CSV
  -> launch mock: synthetic events -> normalized summary/events -> CSV/JSONL
  -> launch sglang: process launcher + samplers -> normalized summary/events
```

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
