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
metrics patch, launches `python -m sglang.launch_server --device cpu
--enable-aether-metrics`, sends a tiny `/generate` request through
`aether launch --backend sglang`, collects `/aether/metrics`, and prints
`summary.csv`, `events.jsonl`, `sglang_aether_metrics.json`, and `sglang.log`.

## Documentation Map

Treat `AGENTS.md` as the maintainer handoff and routing guide. It should point
to the authoritative project docs instead of duplicating long explanations.
When code behavior changes, update the relevant doc below in the same PR.

- `README.md` is the user-facing entry point. Keep install commands, CLI
  examples, config examples, and the high-level CPU/GPU support story there.
- `docs/greensserve-ipw-plan.md` is the compact product/research brief. Keep
  the GreenServe system components there: context-aware routing pools, KV-cache
  compression, prefill/decode energy asymmetry, recompute-vs-swap scheduling,
  and SGLang metrics collection.
- `docs/formula-ledger.md` is the formula source of truth for simulation math.
  Update it whenever `src/aether/formulas.py` or `src/aether/simulator.py`
  changes any equation, default, unit, or scheduling score.
- `docs/architecture.md` is the system boundary and data-flow map. Update it
  when package layout, CLI command flow, normalized output schema, or backend
  boundaries change.
- `docs/mock-validation.md` describes the deterministic CPU correctness gate.
  Update it when `src/aether/measurement/mock.py`, mock events, expected files,
  or golden-output assumptions change.
- `docs/real-data-collection.md` is the operational runbook for real SGLang
  CPU/GPU measurement. Keep Linux setup, macOS limitations, patched metrics
  flags, endpoint behavior, NVML notes, and output artifacts there.
- `patches/sglang/v0.5.12/README.md` is the patch-series-local guide. Keep it
  aligned with the patch files and use it for SGLang-specific apply commands,
  scope constraints, and upstream-version notes.
- `experiments/greensserve/*.yaml` are executable examples. Keep them in sync
  with README snippets, docs runbooks, and tests.
- `.github/workflows/ci.yml` is the executable proof for docs that claim a flow
  works in CI. If a doc says the SGLang CPU path is validated, the workflow
  should actually install SGLang, apply the patch, run Aether, and assert the
  patched metrics payload.

## Change Routing

- Formula or GreenServe model changes: update `docs/formula-ledger.md`,
  `docs/greensserve-ipw-plan.md` if the conceptual model changed, simulation
  tests, and at least one config example if new inputs are required.
- CLI/config changes: update `README.md`, `docs/architecture.md`,
  `experiments/greensserve/*.yaml`, config tests, and CLI tests.
- Normalized output schema changes: update `docs/architecture.md`,
  `docs/mock-validation.md`, `docs/real-data-collection.md`, result-writer
  tests, and any CI assertions that parse output files.
- Mock backend changes: update `docs/mock-validation.md` and keep the mock
  path deterministic on CPU.
- Real SGLang backend changes: update `docs/real-data-collection.md`,
  `experiments/greensserve/sglang_cpu_ci.yaml`,
  `experiments/greensserve/sglang_real.yaml`, and the dedicated SGLang CI job.
- SGLang patch changes: update `patches/sglang/v0.5.12/README.md`,
  `docs/real-data-collection.md`, the patch apply checks, and the CI assertions
  that verify `/aether/metrics`.

## Current SGLang Metrics Patch Context

The active patch is
`patches/sglang/v0.5.12/0001-aether-ipw-metrics-scaffold.patch`. Despite the
name, it now implements the first real metrics path for Aether:

- SGLang flags: `--enable-aether-metrics` and
  `--aether-metrics-retain-events`.
- SGLang endpoint: `GET /aether/metrics`.
- Aether collection: `src/aether/measurement/sglang.py` reads the endpoint,
  emits an `sglang_aether_metrics` event, writes `sglang_aether_metrics.json`,
  and can require the endpoint with `measurement.require_sglang_metrics: true`.
- CI proof: the `SGLang CPU launch` job installs patched SGLang on Linux CPU,
  runs `aether launch --backend sglang`, and asserts the metrics summary has at
  least one request and generated tokens.

Keep this patch metrics-only. Do not add scheduler, recompute/swap, DVFS, model
execution, or routing behavior changes without a new design doc and matching
tests.

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
- SGLang patches in this bootstrap are metrics-only. The current patch adds
  `--enable-aether-metrics`, `--aether-metrics-retain-events`, and
  `GET /aether/metrics`. Do not add scheduler, recompute/swap, or DVFS
  behavior changes without a new design doc.
- SGLang `v0.5.12` does not fully install on local macOS arm64 in this
  workspace because `sgl-deep-gemm==0.1.0` provides Linux wheels only. Use
  macOS for patch-apply and Aether mock validation; use GitHub Actions or Linux
  hosts for SGLang CPU serving tests, and Linux GPU hosts for NVML/GPU runs.
