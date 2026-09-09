"""Train archetype configuration (CLAUDE.md section 13).

Trailing load is kept separate from total train mass (trailing + locomotives); resistance
calculations that need total mass combine this with locomotive mass explicitly in simulation.py.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Hashable

import yaml


@dataclass(frozen=True)
class DavisCoefficients:
    a_n_per_t: float
    b_n_per_t_per_kmh: float
    c_n_per_t_per_kmh2: float


@dataclass(frozen=True)
class Train:
    name: str
    status: str
    trailing_mass_t: float
    davis: DavisCoefficients
    max_permitted_speed_kmh: float | None = None
    train_length_m: float | None = None


def load_train(path: str | Path) -> Train:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    davis_raw = raw["davis_resistance"]
    davis = DavisCoefficients(
        a_n_per_t=davis_raw["a_n_per_t"],
        b_n_per_t_per_kmh=davis_raw["b_n_per_t_per_kmh"],
        c_n_per_t_per_kmh2=davis_raw["c_n_per_t_per_kmh2"],
    )

    return Train(
        name=raw["name"],
        status=raw.get("status", "PLACEHOLDER"),
        trailing_mass_t=raw["trailing_mass_t"],
        davis=davis,
        max_permitted_speed_kmh=raw.get("max_permitted_speed_kmh"),
        train_length_m=raw.get("train_length_m"),
    )


def resolve_route_weight(
    base_train: Train, route_key: Hashable, overrides: dict[Hashable, float]
) -> Train:
    """Real per-route train weight (2026-09-06, CLAUDE.md's own "Unresolved" note on route-level
    train-weight distributions): returns `base_train` unchanged unless `route_key` (a segment_code,
    a hub-pair tuple, or whatever identity a caller's own route loop naturally uses) has a curated
    entry in `overrides`, in which case only `trailing_mass_t` is swapped -- same in-memory
    construction pattern `operating_concepts.run_mass_sweep` already uses per mass, just keyed by
    route identity instead of a mass list. Shared by both pan-European pipeline scripts
    (`compare_locomotives_on_routes.py`'s `SEGMENT_WEIGHT_OVERRIDES_T`,
    `build_country_od_pair_analysis.py`'s `HUB_PAIR_WEIGHT_OVERRIDES_T`) so the lookup/replace logic
    exists once, not duplicated per script."""
    if route_key not in overrides:
        return base_train
    return replace(base_train, trailing_mass_t=overrides[route_key])
