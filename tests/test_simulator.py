from aether.config import load_config
from aether.simulator import SimulationBackend, simulate


def test_baseline_sweep_expands():
    config = load_config("experiments/greensserve/baseline.yaml")
    rows = simulate(config)

    assert len(rows) == 4 * 3 * 2 * 4
    assert rows[0]["backend"] == "simulation"
    assert "tokens_per_joule" in rows[0]
    assert max(row["max_inflight_sequences"] for row in rows) > 0
    assert max(row["generated_tokens"] for row in rows) > 0


def test_bayesian_scheduler_selects_action():
    config = load_config("experiments/greensserve/mock_measurement.yaml")
    rows = simulate(config)

    assert len(rows) == 1
    assert rows[0]["scheduling_action"] in {"recompute", "swap", "none"}


def test_simulator_accepts_pluggable_backend():
    class TinyBackend(SimulationBackend):
        name = "tiny"

        def simulate(self, config):
            return [{"backend": self.name, "experiment": config["experiment"]["name"]}]

    rows = simulate({"experiment": {"name": "custom"}}, backend=TinyBackend())

    assert rows == [{"backend": "tiny", "experiment": "custom"}]
