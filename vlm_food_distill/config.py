"""Config loading and small shared helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config into a plain dict."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg.setdefault("classes", [])
    cfg.setdefault("max_train_per_class", None)
    cfg.setdefault("max_test_per_class", None)
    cfg.setdefault("teacher", {})
    cfg.setdefault("student", {})
    return cfg


def humanize(class_name: str) -> str:
    """``hot_dog`` -> ``hot dog`` for prompting / display."""
    return class_name.replace("_", " ")


def canonicalize(text: str) -> str:
    """Normalise free text to the Food-101 ``snake_case`` convention."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")
