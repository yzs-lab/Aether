# AGENTS.md

## Project

Aether is a Python package and CLI for GreenServe intelligence-per-watt (IPW)
simulation and measurement. Use uv and Python 3.13. Keep the package installable with:

```bash
uv python install 3.13
uv sync --group dev
```

Run the CPU-safe checks with:

```bash
uv run pytest
uv run aether simulate --config experiments/greensserve/baseline.yaml --out /tmp/aether-sim.csv
uv run aether launch --backend mock --config experiments/greensserve/mock_measurement.yaml --out /tmp/aether-mock
git -C third_party/sglang apply --check ../../patches/sglang/v0.5.12/0001-aether-ipw-metrics-scaffold.patch
```

GitHub CI also runs a real SGLang CPU integration job. It installs SGLang from
the pinned submodule using SGLang's `pyproject_cpu.toml`, applies the Aether
metrics patch, launches `python -m sglang.launch_server --device cpu`, sends a
tiny `/generate` request through `aether launch --backend sglang`, and prints
`summary.csv`, `events.jsonl`, and `sglang.log`.

## Network Proxy Preference

When a download or dependency fetch appears stuck or blocked by network
restrictions, retry with these proxy variables before giving up:

```bash
http_proxy=http://127.0.0.1:10808
https_proxy=http://127.0.0.1:10808
all_proxy=socks5://127.0.0.1:10808
HTTP_PROXY=http://127.0.0.1:10808
HTTPS_PROXY=http://127.0.0.1:10808
ALL_PROXY=socks5://127.0.0.1:10808
```

## Repo Rules

- Keep real SGLang/GPU measurement code optional. Importing `aether` and running
  unit tests must not require CUDA, NVML, SGLang, or a GPU. CI may install
  SGLang separately for the dedicated CPU integration job.
- Use mock measurement as the correctness gate for launcher, collector, and
  result-writer behavior.
- Keep normalized CSV/JSONL schemas shared across simulation, mock, and real
  measurement paths.
- Store experiment outputs under `results/`; this directory is ignored.
- Keep SGLang work under `third_party/sglang` and patch files under
  `patches/sglang/v0.5.12/`.
- SGLang patches in this bootstrap are metrics-only. Do not add scheduler,
  recompute/swap, or DVFS behavior changes without a new design doc.
- SGLang `v0.5.12` does not fully install on local macOS arm64 in this
  workspace because `sgl-deep-gemm==0.1.0` provides Linux wheels only. Use
  macOS for patch-apply and Aether mock validation; use GitHub Actions or Linux
  hosts for SGLang CPU serving tests, and Linux GPU hosts for NVML/GPU runs.
