"""Aether command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .config import ConfigError, load_config, render_sglang_args
from .measurement.mock import run_mock
from .measurement.sglang import SGLangMeasurementError, run_sglang
from .results import write_csv
from .simulator import simulate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aether", description="Aether IPW simulation and measurement CLI")
    subcommands = parser.add_subparsers(dest="command", required=True)

    simulate_parser = subcommands.add_parser("simulate", help="run an offline IPW scenario sweep")
    simulate_parser.add_argument("--config", required=True)
    simulate_parser.add_argument("--out", required=True)

    launch_parser = subcommands.add_parser("launch", help="launch a measurement backend")
    launch_parser.add_argument("--backend", choices=["mock", "sglang"], required=True)
    launch_parser.add_argument("--config", required=True)
    launch_parser.add_argument("--out", required=True)

    doctor_parser = subcommands.add_parser("doctor", help="validate config and print environment guidance")
    doctor_parser.add_argument("--config", required=True)

    explain_parser = subcommands.add_parser("explain", help="print expanded scenarios and rendered SGLang args")
    explain_parser.add_argument("--config", required=True)
    return parser


def _cmd_simulate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    rows = simulate(config)
    write_csv(rows, args.out)
    print("wrote %s rows to %s" % (len(rows), args.out))
    return 0


def _cmd_launch(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.backend == "mock":
        row, events = run_mock(config, args.out)
    else:
        row, events = run_sglang(config, args.out)
    print("wrote summary.csv and events.jsonl to %s" % args.out)
    print("backend=%s scenario_id=%s events=%s" % (args.backend, row.get("scenario_id"), len(events)))
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    rows = simulate(config)
    print("config: %s" % Path(args.config))
    print("simulation_scenarios: %s" % len(rows))
    print("sglang_args: %s" % " ".join(render_sglang_args(config)))
    print("cpu_safe: true")
    print("real_sglang_requires: SGLang plus a workload; GPU/NVML only when nvml_enabled=true")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    payload = {
        "sglang_args": render_sglang_args(config),
        "scenarios": simulate(config),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "simulate":
            return _cmd_simulate(args)
        if args.command == "launch":
            return _cmd_launch(args)
        if args.command == "doctor":
            return _cmd_doctor(args)
        if args.command == "explain":
            return _cmd_explain(args)
    except (ConfigError, SGLangMeasurementError, ValueError) as exc:
        print("aether: %s" % exc, file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
