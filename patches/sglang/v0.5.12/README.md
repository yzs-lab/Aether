# Aether SGLang Patch Series For v0.5.12

This directory contains the first Aether patch scaffold for SGLang `v0.5.12`.
The bootstrap goal is metrics only:

- power and clock samples
- request token counts
- TTFT/TBT/end-to-end latency markers
- KV cache usage
- preemption, recompute, and swap counters where SGLang exposes them

No patch in this series should alter scheduling behavior, DVFS behavior, model
execution, or request routing.

Apply from the repo root after initializing the submodule:

```bash
git submodule update --init --recursive third_party/sglang
cd third_party/sglang
git checkout v0.5.12
git apply --check ../../patches/sglang/v0.5.12/*.patch
```

`0001-aether-ipw-metrics-scaffold.patch` is intentionally minimal until the
exact SGLang metrics insertion points are selected in a follow-up patch.
