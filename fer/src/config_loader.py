"""Loads configs/config.yaml into a dot-accessible object."""

from pathlib import Path
import yaml


class DotDict(dict):
    """Dictionary that allows dot notation access."""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e

        if isinstance(value, dict):
            value = DotDict(value)

        return value

    def __setattr__(self, key, value):
        self[key] = value


def load_config(config_path: str = None) -> DotDict:
    """Load the YAML configuration file."""

    if config_path is None:
        # config_loader.py -> src -> fer_project
        root = Path(__file__).resolve().parent.parent
        config_path = root / "configs" / "config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    return DotDict(raw)