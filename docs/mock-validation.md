# Mock Validation

The mock backend is the CPU-only correctness gate for Aether. It does not start
SGLang and does not require GPU libraries.

Run:

```bash
aether launch --backend mock --config experiments/greensserve/mock_measurement.yaml --out /tmp/aether-mock
```

Expected outputs:

- `/tmp/aether-mock/summary.csv`
- `/tmp/aether-mock/events.jsonl`

The mock backend emits deterministic event streams:

- power samples
- token events
- KV cache events
- scheduler events for recompute and swap
- latency samples

Because the events are deterministic, tests can compare stable CSV columns and
known metric ranges without relying on hardware.
