from aether import formulas


def test_kv_bytes_per_token():
    assert formulas.kv_bytes_per_token(layers=80, kv_heads=8, head_dim=128, kv_dtype_bytes=2) == 327680


def test_compression_improves_capacity():
    available = formulas.available_kv_vram_bytes(80, 40, 0.9)
    kv = formulas.kv_bytes_per_token(80, 8, 128, 2)
    base = formulas.max_inflight_sequences(available, kv, 4096)
    compressed = formulas.max_inflight_sequences(available, formulas.effective_kv_bytes(kv, 0.5), 4096)

    assert compressed == base * 2 or compressed == base * 2 + 1


def test_risk_adjusted_cost():
    assert formulas.risk_adjusted_cost(10, 2, 1.5) == 13
