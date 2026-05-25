import aether.measurement.sglang as sglang_measurement
from aether.measurement.sglang import (
    _extract_response_metrics,
    _request_summary,
    _run_configured_requests,
)


def test_extract_response_metrics_from_sglang_generate_response():
    metrics = _extract_response_metrics(
        {
            "text": "ok",
            "meta_info": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "e2e_latency": 0.25,
                "ttft": 0.04,
                "tbt": 0.01,
            },
        }
    )

    assert metrics["prompt_tokens"] == 5
    assert metrics["generated_tokens"] == 3
    assert metrics["total_tokens"] == 8
    assert metrics["response_latency_s"] == 0.25
    assert metrics["ttft_ms"] == 40.0
    assert metrics["tbt_ms"] == 10.0


def test_extract_response_metrics_from_openai_usage_response():
    metrics = _extract_response_metrics(
        {
            "choices": [{"text": "ok"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        }
    )

    assert metrics == {"prompt_tokens": 4, "generated_tokens": 2, "total_tokens": 6}


def test_run_configured_requests_posts_json_and_records_metrics(monkeypatch):
    calls = []

    def fake_post_json(url, payload, timeout_s):
        calls.append((url, payload, timeout_s))
        return (
            200,
            {
                "text": "hi",
                "meta_info": {
                    "prompt_tokens": 7,
                    "completion_tokens": 2,
                    "e2e_latency": 0.5,
                },
            },
        )

    monkeypatch.setattr(sglang_measurement, "_post_json", fake_post_json)
    events = []
    _run_configured_requests(
        "http://127.0.0.1:30080",
        {
            "requests": [
                {
                    "endpoint": "/generate",
                    "payload": {
                        "text": "hello",
                        "sampling_params": {"temperature": 0.0, "max_new_tokens": 2},
                    },
                }
            ]
        },
        events,
    )

    assert calls == [
        (
            "http://127.0.0.1:30080/generate",
            {"text": "hello", "sampling_params": {"temperature": 0.0, "max_new_tokens": 2}},
            120.0,
        )
    ]
    assert events[0]["type"] == "request"
    assert events[0]["status"] == 200
    assert events[0]["prompt_tokens"] == 7
    assert events[0]["generated_tokens"] == 2
    assert _request_summary(events)["generated_tokens"] == 2
