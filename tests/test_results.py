from aether.results import SUMMARY_FIELDS, read_csv_rows, write_csv


def test_write_csv_stable_schema(tmp_path):
    path = tmp_path / "out.csv"
    write_csv([{"experiment": "x", "backend": "mock", "extra": "ignored"}], str(path))
    rows = read_csv_rows(str(path))

    assert list(rows[0].keys()) == SUMMARY_FIELDS
    assert rows[0]["experiment"] == "x"
    assert rows[0]["backend"] == "mock"
