"""Load config.toml into frozen dataclasses.

TOML because it is in the standard library (tomllib, Python 3.11+) and it
allows comments. The loader is strict: an unknown key, a missing key or a
wrong type fails at startup naming the key, instead of silently falling
back to a default somewhere downstream.

The schema starts empty. Each phase adds the section it reads, so no key
exists that nothing consumes.
"""

import dataclasses
import tomllib
from dataclasses import MISSING, dataclass
from pathlib import Path
from typing import get_type_hints


class ConfigError(Exception):
    """The config file is missing, malformed, or has a bad key."""


@dataclass(frozen=True)
class SourceConfig:
    device: int  # webcam index as OpenCV enumerates it
    width: int  # requested capture size; the driver picks the nearest mode it has
    height: int


@dataclass(frozen=True)
class DetectorConfig:
    model_path: str  # SCRFD .onnx with keypoints, relative to the working directory
    conf_threshold: float  # faces scoring below this are dropped before NMS
    input_size: int  # side of the square the frame is letterboxed into; multiple of 32


@dataclass(frozen=True)
class EmbedderConfig:
    model_path: str  # ArcFace-family .onnx taking one aligned square crop, relative to the working directory


@dataclass(frozen=True)
class Config:
    source: SourceConfig
    detector: DetectorConfig
    embedder: EmbedderConfig


def load_config(path: str | Path) -> Config:
    path = Path(path)
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        raise ConfigError("file not found") from None
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"not valid TOML: {e}") from None
    return _build(Config, data, where="")


def _build(cls, table: dict, where: str):
    """Map a TOML table onto dataclass `cls`, recursing into nested sections."""
    hints = get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(table) - set(fields))
    if unknown:
        scope = f"[{where}]" if where else "top level"
        raise ConfigError(f"unknown key(s) at {scope}: {', '.join(unknown)}")
    kwargs = {}
    for name, field in fields.items():
        key = f"{where}.{name}" if where else name
        if name not in table:
            if field.default is MISSING and field.default_factory is MISSING:
                raise ConfigError(f"missing required key: {key}")
            continue
        value = table[name]
        typ = hints[name]
        if dataclasses.is_dataclass(typ):
            if not isinstance(value, dict):
                raise ConfigError(f"{key} must be a table: [{key}]")
            kwargs[name] = _build(typ, value, where=key)
        else:
            kwargs[name] = _check(value, typ, key)
    return cls(**kwargs)


def _check(value, typ, key):
    # TOML `1` parses as int; accept it where a float is expected. bool is a
    # subclass of int in Python, so `width = true` needs an explicit rejection.
    if typ is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if (isinstance(value, bool) and typ is not bool) or not isinstance(value, typ):
        raise ConfigError(f"{key} must be {typ.__name__}, got {type(value).__name__}")
    return value
