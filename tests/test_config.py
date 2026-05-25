from aether.config import render_sglang_args


def test_render_sglang_args_sorts_and_renders_flags():
    config = {
        "sglang": {
            "args": {
                "port": 30000,
                "enable-metrics": True,
                "disabled": False,
                "none": None,
            },
            "extra_args": ["--trust-remote-code"],
        }
    }

    assert render_sglang_args(config) == [
        "--enable-metrics",
        "--port",
        "30000",
        "--trust-remote-code",
    ]
