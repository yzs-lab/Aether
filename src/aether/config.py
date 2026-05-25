"""Configuration loading and SGLang argument rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when an Aether YAML config is invalid."""


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError("config file does not exist: %s" % config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError("config root must be a mapping")
    return data


def require_mapping(data: Mapping[str, Any], key: str) -> Dict[str, Any]:
    value = data.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError("%s must be a mapping" % key)
    return dict(value)


def as_list(value: Any, default: Iterable[Any]) -> List[Any]:
    if value is None:
        return list(default)
    if isinstance(value, list):
        return value
    return [value]


def render_flag_name(name: str) -> str:
    return "--" + name.replace("_", "-")


def render_cli_args(args: Mapping[str, Any]) -> List[str]:
    rendered = []
    for key in sorted(args):
        value = args[key]
        flag = render_flag_name(str(key))
        if value is None or value is False:
            continue
        if value is True:
            rendered.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                rendered.extend([flag, str(item)])
            continue
        if isinstance(value, dict):
            raise ConfigError("nested SGLang arg %s is not supported" % key)
        rendered.extend([flag, str(value)])
    return rendered


def render_sglang_args(config: Mapping[str, Any]) -> List[str]:
    sglang = require_mapping(config, "sglang")
    args = sglang.get("args", {})
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ConfigError("sglang.args must be a mapping")
    extra_args = sglang.get("extra_args", [])
    if extra_args is None:
        extra_args = []
    if not isinstance(extra_args, list):
        raise ConfigError("sglang.extra_args must be a list")
    return render_cli_args(args) + [str(item) for item in extra_args]


def experiment_name(config: Mapping[str, Any]) -> str:
    experiment = require_mapping(config, "experiment")
    return str(experiment.get("name", "aether-experiment"))
