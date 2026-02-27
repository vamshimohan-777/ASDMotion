import os
import yaml


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def _set_by_path(root: dict, keys: list[str], value):
    cur = root
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    if not overrides:
        return cfg
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw = item.split("=", 1)
        key = key.strip()
        # Parse value using yaml so numbers/bools/lists work
        value = yaml.safe_load(raw)
        _set_by_path(cfg, key.split("."), value)
    return cfg
