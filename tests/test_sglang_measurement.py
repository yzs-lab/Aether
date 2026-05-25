import aether.measurement.sglang as sglang_measurement
from aether.measurement.sglang import (
    _collect_sglang_metrics,
    _delta_sglang_metrics,
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


def test_collect_sglang_metrics_records_patched_endpoint(monkeypatch):
    def fake_get_json(url, timeout_s):
        assert url == "http://127.0.0.1:30080/aether/metrics"
        assert timeout_s == 9.0
        return (
            200,
            {
                "enabled": True,
                "summary": {
                    "request_count": 1,
                    "prompt_tokens_total": 7,
                    "completion_tokens_total": 2,
                },
                "events": [],
            },
        )

    monkeypatch.setattr(sglang_measurement, "_get_json", fake_get_json)
    events = []
    payload = _collect_sglang_metrics(
        "http://127.0.0.1:30080",
        {
            "sglang_metrics_endpoint": "/aether/metrics",
            "request_timeout_seconds": 9,
            "require_sglang_metrics": True,
        },
        events,
    )

    assert payload["summary"]["completion_tokens_total"] == 2
    assert events == [
        {
            "type": "sglang_aether_metrics",
            "backend": "sglang",
            "endpoint": "/aether/metrics",
            "status": 200,
            "ok": True,
            "payload": payload,
        }
    ]


def test_delta_sglang_metrics_removes_server_warmup_request():
    baseline = {
        "enabled": True,
        "summary": {
            "request_count": 1,
            "prompt_tokens_total": 7,
            "completion_tokens_total": 8,
            "total_tokens_total": 15,
            "e2e_latency_sum_s": 0.112,
            "last_request_unix_s": 100.0,
        },
        "events": [{"timestamp_unix_s": 100.0, "completion_tokens": 8}],
    }
    after = {
        "enabled": True,
        "summary": {
            "request_count": 2,
            "prompt_tokens_total": 22,
            "completion_tokens_total": 10,
            "total_tokens_total": 32,
            "e2e_latency_sum_s": 0.119,
            "last_request_unix_s": 101.0,
        },
        "events": [
            {"timestamp_unix_s": 100.0, "completion_tokens": 8},
            {"timestamp_unix_s": 101.0, "completion_tokens": 2},
        ],
    }

    delta = _delta_sglang_metrics(after, baseline)

    assert delta["baseline_applied"] is True
    assert delta["summary"]["request_count"] == 1
    assert delta["summary"]["prompt_tokens_total"] == 15
    assert delta["summary"]["completion_tokens_total"] == 2
    assert delta["summary"]["total_tokens_total"] == 17
    assert delta["summary"]["e2e_latency_sum_s"] == 0.007
    assert delta["events"] == [{"timestamp_unix_s": 101.0, "completion_tokens": 2}]


def test_collect_sglang_metrics_applies_baseline_delta(monkeypatch):
    def fake_get_json(url, timeout_s):
        return (
            200,
            {
                "enabled": True,
                "summary": {
                    "request_count": 2,
                    "prompt_tokens_total": 22,
                    "completion_tokens_total": 10,
                    "total_tokens_total": 32,
                    "e2e_latency_sum_s": 0.119,
                    "last_request_unix_s": 101.0,
                },
                "events": [
                    {"timestamp_unix_s": 100.0, "completion_tokens": 8},
                    {"timestamp_unix_s": 101.0, "completion_tokens": 2},
                ],
            },
        )

    monkeypatch.setattr(sglang_measurement, "_get_json", fake_get_json)
    events = []
    payload = _collect_sglang_metrics(
        "http://127.0.0.1:30080",
        {"sglang_metrics_endpoint": "/aether/metrics"},
        events,
        {
            "summary": {
                "request_count": 1,
                "prompt_tokens_total": 7,
                "completion_tokens_total": 8,
                "total_tokens_total": 15,
                "e2e_latency_sum_s": 0.112,
                "last_request_unix_s": 100.0,
            }
        },
    )

    assert payload["summary"]["request_count"] == 1
    assert payload["summary"]["completion_tokens_total"] == 2
    assert events[0]["payload"]["baseline_applied"] is True
