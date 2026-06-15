"""Load and access config.yaml. The engine reads tunables only from here."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class Config:
    """Thin typed-ish wrapper over the parsed YAML with dotted access."""

    def __init__(self, data: dict[str, Any], source: Path | None = None):
        self._data = data
        self.source = source

    # -- dotted getters -----------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    # -- convenience accessors used across the engine ----------------------
    @property
    def aliases(self) -> dict[str, list[str]]:
        return self._data["aliases"]

    @property
    def scoring(self) -> dict[str, Any]:
        return self._data["scoring"]

    @property
    def bands(self) -> list[dict[str, Any]]:
        return self._data["bands"]

    @property
    def dedup(self) -> dict[str, Any]:
        return self._data["dedup"]

    @property
    def flags(self) -> dict[str, Any]:
        return self._data["flags"]

    @property
    def clean(self) -> dict[str, Any]:
        return self._data["clean"]

    def snapshot_json(self) -> str:
        """Serialized config stored alongside each run for auditability."""
        return json.dumps(self._data, sort_keys=True, default=str)


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(data, source=p)
