from aether.cli import main
from aether.results import read_csv_rows


def test_cli_simulate(tmp_path):
    output = tmp_path / "sim.csv"
    rc = main(["simulate", "--config", "experiments/greensserve/baseline.yaml", "--out", str(output)])

    assert rc == 0
    rows = read_csv_rows(str(output))
    assert len(rows) == 96


def test_cli_mock_launch(tmp_path):
    output = tmp_path / "mock"
    rc = main(
        [
            "launch",
            "--backend",
            "mock",
            "--config",
            "experiments/greensserve/mock_measurement.yaml",
            "--out",
            str(output),
        ]
    )

    assert rc == 0
    rows = read_csv_rows(str(output / "summary.csv"))
    assert rows[0]["backend"] == "mock"
    assert (output / "events.jsonl").exists()


def test_cli_sglang_backend_is_optional(tmp_path):
    output = tmp_path / "sglang"
    rc = main(
        [
            "launch",
            "--backend",
            "sglang",
            "--config",
            "experiments/greensserve/sglang_real.yaml",
            "--out",
            str(output),
        ]
    )

    assert rc == 2


def test_cli_doctor_prints_hardware_backend(capsys):
    rc = main(["doctor", "--config", "experiments/greensserve/sglang_cpu_ci.yaml"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "measurement_hardware_backend: cpu" in output
