# GreenServe Formula Ledger

Some inline formulas in the source Google Doc are embedded objects and were not
visible through text extraction. This ledger records the reconstructed formulas
implemented by Aether v0.1.

## KV Cache

```text
kv_bytes_per_token = 2 * layers * kv_heads * head_dim * kv_dtype_bytes
effective_kv_bytes = kv_bytes_per_token * compression_ratio + metadata_bytes_per_token
model_weight_bytes_per_gpu = model_weight_bytes / gpu_count
available_kv_vram_bytes = gpu_count * max(0, vram_bytes * gpu_memory_utilization - model_weight_bytes_per_gpu)
max_inflight = floor(available_kv_vram_bytes / (effective_kv_bytes * context_tokens))
```

`compression_ratio=1.0` means uncompressed KV. Smaller values represent a
smaller retained memory footprint, for example `0.5` for half-size KV.

## Energy And IPW

```text
phase_energy_j = phase_seconds * phase_power_w
prefill_power_w = gpu_count * tdp_w * prefill_power_fraction
decode_power_w = gpu_count * tdp_w * decode_power_fraction
energy_j = prefill_energy_j + decode_energy_j + scheduling_energy_j
avg_power_w = energy_j / elapsed_seconds
tokens_per_second = generated_tokens / elapsed_seconds
tokens_per_watt = tokens_per_second / avg_power_w
tokens_per_joule = generated_tokens / energy_j
quality_normalized_ipw = quality_score * tokens_per_joule
```

Defaults from the GreenServe plan:

- Prefill is compute-bound and uses `0.86 * TDP`.
- Decode is memory-bound and has a static floor of `0.43 * TDP`.

## 1/W Context Law

Aether models the 1/W law through KV capacity:

```text
max_inflight is inversely proportional to context_tokens
effective_decode_throughput = decode_tps * min(1, max_inflight / target_concurrency)
```

As context grows, KV footprint grows linearly, maximum in-flight sequences fall,
and useful decode throughput per watt drops.

## Recompute Vs Swap

```text
risk_adjusted_cost = expected_cost + risk_weight * uncertainty
```

Recompute cost is modeled as another prefill/refill pass over preempted context.
Swap cost is modeled as PCIe transfer time over the request KV footprint plus
decode/static waiting power. The Bayesian scheduler chooses the lower
risk-adjusted cost.
