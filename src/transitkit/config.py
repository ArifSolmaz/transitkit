"""Configuration defaults for TransitKit."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TransitKitConfig:
    """Base configuration for TransitKit."""

    cache_dir: str = "~/.cache/transitkit"
    default_mission: str = "TESS"
    default_timeout_s: int = 60
