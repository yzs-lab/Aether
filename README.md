# Aether

Aether is an installable Python CLI for GreenServe-style intelligence-per-watt
(IPW) experiments. The current codebase is intentionally runnable on CPU: it
supports pure simulation and a deterministic mock measurement backend that
validates the same result schema used by future SGLang/GPU runs.

The GitHub CI is CPU-only. It uses uv with Python 3.13, runs tests and smoke
commands, verifies the SGLang metrics patch applies to the pinned submodule,
and has a dedicated job that installs SGLang's CPU source build, applies the
Aether metrics patch, launches an actual SGLang CPU server, sends an
Aether-managed request, reads `/aether/metrics`, and prints the normalized
outputs.

Real GPU experiments are kept separate. The SGLang backend is present, but GPU
power sampling only runs when SGLang, CUDA/NVML, and a workload are available
in the active environment.

Note: SGLang `v0.5.12` real serving is Linux-oriented. On this local macOS arm64
machine, full install is blocked by upstream Linux-only `sgl-deep-gemm` wheels;
see `docs/real-data-collection.md` for the smoke-test result and the Linux CPU
CI path.

## Install

```bash
uv python install 3.13
uv sync --group dev
```

After install:

```bash
uv run aether doctor --config experiments/greensserve/baseline.yaml
uv run aether simulate --config experiments/greensserve/baseline.yaml --out results/sim.csv
uv run aether launch --backend mock --config experiments/greensserve/mock_measurement.yaml --out results/mock-run
uv run pytest
```

When published, the intended user install is:

```bash
pip install aether
# or, with uv:
uv tool install aether
```

## CLI

```bash
uv run aether simulate --config CONFIG.yaml --out results/sim.csv
uv run aether launch --backend mock --config CONFIG.yaml --out results/mock-run
uv run aether launch --backend sglang --config CONFIG.yaml --out results/real-run
uv run aether doctor --config CONFIG.yaml
uv run aether explain --config CONFIG.yaml
```

`simulate` expands YAML sweeps and writes one normalized CSV row per scenario.
`launch --backend mock` runs a deterministic CPU-only measurement fixture and
writes `summary.csv` plus `events.jsonl`. `launch --backend sglang` starts
`python -m sglang.launch_server` with the configured SGLang arguments, then
collects normalized measurement output if the real environment is available.

## Config

Examples live in `experiments/greensserve/`.

- `baseline.yaml`: pure simulation sweep.
- `mock_measurement.yaml`: CPU-only launcher and collector validation.
- `sglang_cpu_ci.yaml`: real SGLang CPU launch used by GitHub Actions.
- `sglang_real.yaml`: template for future SGLang/GPU measurement.

The SGLang config accepts structured args:

```yaml
sglang:
  args:
    model-path: meta-llama/Llama-3.1-8B-Instruct
    host: 127.0.0.1
    port: 30000
    context-length: 8192
    enable-metrics: true
    enable-aether-metrics: true
    aether-metrics-retain-events: 4096
  extra_args:
    - --trust-remote-code
```

Booleans set to `true` are rendered as flags; `false` and `null` are omitted.
All other values are rendered as `--flag value`.

The patched SGLang CPU CI config enables:

```yaml
sglang:
  args:
    enable-aether-metrics: true
    aether-metrics-retain-events: 32
measurement:
  sglang_metrics_endpoint: /aether/metrics
  require_sglang_metrics: true
```

## Real Data Collection

Real collection is documented in `docs/real-data-collection.md`. In short:

1. Initialize the SGLang submodule pinned to `v0.5.12`.
2. Apply the metrics-only patch series from `patches/sglang/v0.5.12/`.
3. Install patched SGLang in the active GPU environment.
4. Run `uv run aether launch --backend sglang --config experiments/greensserve/sglang_real.yaml --out results/real-run`.
5. Use `summary.csv`, `events.jsonl`, SGLang logs, and NVML samples for analysis.

The unit test suite does not require SGLang, CUDA, NVML, or a GPU. GitHub CI
adds an integration job that installs SGLang CPU separately and exercises the
same Aether result writer against a live local SGLang server.
