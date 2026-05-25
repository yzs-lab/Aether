# Real SGLang Data Collection

Real measurement is separate from CPU validation. Use it only on a machine with
NVIDIA GPUs, working drivers, CUDA, NVML, and an installed SGLang environment.

## Prepare SGLang

```bash
uv sync --group dev --group sglang
git submodule update --init --recursive third_party/sglang
cd third_party/sglang
git checkout v0.5.12
git apply --check ../../patches/sglang/v0.5.12/*.patch
git apply ../../patches/sglang/v0.5.12/*.patch
uv pip install -e "python[all]"
```

The bootstrap patch series is metrics-only. It must not change scheduling,
DVFS, or recompute/swap behavior.

## macOS Local Smoke Result

SGLang `v0.5.12` is not currently installable as a real local macOS arm64
serving stack in this workspace:

- Full dependency install fails because `sglang==0.5.12` depends on
  `sgl-deep-gemm==0.1.0`, and uv reports wheels only for
  `manylinux2014_aarch64` and `manylinux2014_x86_64`.
- A no-dependency editable smoke install from a patched SGLang worktree reaches
  SGLang's build step, then fails because `rustc` is not installed locally and
  SGLang builds a Rust extension.

What is verified on macOS:

```bash
git -C third_party/sglang worktree add /private/tmp/aether-sglang-v0.5.12-test v0.5.12
git -C /private/tmp/aether-sglang-v0.5.12-test apply /path/to/Aether/patches/sglang/v0.5.12/0001-aether-ipw-metrics-scaffold.patch
```

The patch applies cleanly and creates `AETHER_IPW_METRICS.md`. A real SGLang
server run should be tested on Linux with the SGLang-supported GPU stack.

## Run Aether

Edit `experiments/greensserve/sglang_real.yaml` for the model, port, context
length, workload, and SGLang arguments.

```bash
uv run aether doctor --config experiments/greensserve/sglang_real.yaml
uv run aether launch --backend sglang --config experiments/greensserve/sglang_real.yaml --out results/real-run
```

The SGLang backend will:

1. render configured SGLang args
2. launch `python -m sglang.launch_server`
3. wait for `/health`
4. sample NVML power when `measurement.nvml_enabled` is true
5. optionally run the configured workload command
6. write normalized `summary.csv` and `events.jsonl`

## Data To Record

- NVML power samples at 100 Hz or higher
- TTFT, TBT, and end-to-end latency
- prompt and generation token counts
- KV cache usage
- swap, recompute, and preemption counters
- SGLang server logs and metrics endpoint snapshots

Keep raw outputs alongside normalized summaries so future formula changes can
recompute metrics from the source data.
