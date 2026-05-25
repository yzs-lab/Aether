# Real SGLang Data Collection

Real measurement is separate from mock validation. Aether can run SGLang in CPU
mode for integration checks, and it can run GPU/NVML collection on a machine
with NVIDIA GPUs, working drivers, CUDA, NVML, and an installed SGLang
environment.

## CI CPU SGLang Check

GitHub Actions has a dedicated `SGLang CPU launch` job that performs a real
source install from the pinned `third_party/sglang` submodule:

1. configure uv to use PyTorch CPU wheels
2. install the Linux system libraries required by SGLang's CPU backend
3. apply `patches/sglang/v0.5.12/0001-aether-ipw-metrics-scaffold.patch`
4. copy SGLang's `pyproject_cpu.toml` files into place
5. install `third_party/sglang/python` and `third_party/sglang/sgl-kernel`
6. run `aether launch --backend sglang` with
   `experiments/greensserve/sglang_cpu_ci.yaml`

The CI log prints `command.json`, `summary.csv`, `events.jsonl`, and the tail
of `sglang.log`. The run is intentionally tiny: it uses
`hf-internal-testing/tiny-random-LlamaForCausalLM`, `--load-format dummy`,
`--device cpu`, and a two-token `/generate` request. This validates Aether's
real launcher, health polling, request path, metric extraction, and normalized
result writer without requiring a GPU.

The same CPU path can be run manually on Linux:

```bash
export UV_CONFIG_FILE=$PWD/.github/uv-sglang-cpu.toml
export SGLANG_USE_CPU_ENGINE=1
sudo apt-get update
sudo apt-get install --no-install-recommends -y google-perftools libtbb-dev libnuma-dev numactl
uv sync --group dev --group sglang
git -C third_party/sglang apply ../../patches/sglang/v0.5.12/0001-aether-ipw-metrics-scaffold.patch
cp third_party/sglang/python/pyproject_cpu.toml third_party/sglang/python/pyproject.toml
cp third_party/sglang/sgl-kernel/pyproject_cpu.toml third_party/sglang/sgl-kernel/pyproject.toml
uv pip install third_party/sglang/python
uv pip install third_party/sglang/sgl-kernel
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=$PWD/.venv/lib/libiomp5.so:/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4:/usr/lib/x86_64-linux-gnu/libtbbmalloc.so.2
uv run aether launch --backend sglang --config experiments/greensserve/sglang_cpu_ci.yaml --out results/sglang-cpu-ci
```

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
5. run configured HTTP requests or an optional workload command
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
