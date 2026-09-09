"""Locomotive configuration loading (CLAUDE.md section 8).

Locomotive parameters live in human-readable YAML, never as hidden constants in code. Every
locomotive carries an overall `status` plus a per-parameter `sources` register.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TeSpeedPoint:
    speed_kmh: float
    te_kn: float


@dataclass(frozen=True)
class Locomotive:
    name: str
    manufacturer: str
    status: str
    mass_t: float
    axles: int | None
    adhesive_weight_t: float
    max_speed_kmh: float
    electric_max_power_kw: float
    diesel_max_power_kw: float
    voltage_systems: tuple[str, ...]
    max_starting_te_kn: float
    te_speed_curve: tuple[TeSpeedPoint, ...]
    lease_eur_per_h: float
    maintenance_eur_per_km: float
    # Real coefficient of adhesion is itself speed-dependent (starting from rest genuinely exceeds
    # continuous/rolling adhesion) -- both None for a locomotive without a sourced split, meaning
    # "use the shared GlobalAssumptions.default_adhesion_coefficient for both" (old behaviour,
    # unchanged). See traction.py's module docstring for how these are actually applied.
    starting_adhesion_coefficient: float | None = None
    continuous_adhesion_coefficient: float | None = None

    def has_diesel(self) -> bool:
        return bool(self.diesel_max_power_kw) and self.diesel_max_power_kw > 0

    def axle_load_t(self) -> float | None:
        """Mass per axle (CLAUDE.md section 16: axle-load compatibility). None if axle count isn't
        known — an unset constraint, not a fabricated zero."""
        if not self.axles:
            return None
        return self.mass_t / self.axles

    def power_kw(self, electrified: bool) -> float:
        return self.electric_max_power_kw if electrified else self.diesel_max_power_kw


def load_locomotive(path: str | Path) -> Locomotive:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    electric = raw.get("electric") or {}
    diesel = raw.get("diesel") or {}
    traction = raw.get("traction") or {}
    economics = raw.get("economics") or {}

    curve_raw = traction.get("te_speed_curve_kn")
    curve = (
        tuple(TeSpeedPoint(p["speed_kmh"], p["te_kn"]) for p in curve_raw)
        if curve_raw
        else ()
    )

    return Locomotive(
        name=raw["name"],
        manufacturer=raw.get("manufacturer", "UNKNOWN"),
        status=raw.get("status", "PLACEHOLDER"),
        mass_t=raw.get("mass_t"),
        axles=raw.get("axles"),
        adhesive_weight_t=raw.get("adhesive_weight_t", raw.get("mass_t")),
        max_speed_kmh=raw.get("max_speed_kmh"),
        electric_max_power_kw=electric.get("max_power_kw") or 0,
        diesel_max_power_kw=diesel.get("max_power_kw") or 0,
        voltage_systems=tuple(electric.get("voltage_systems") or ()),
        max_starting_te_kn=traction.get("max_starting_te_kn"),
        te_speed_curve=curve,
        lease_eur_per_h=economics.get("lease_eur_per_h"),
        maintenance_eur_per_km=economics.get("maintenance_eur_per_km"),
        starting_adhesion_coefficient=traction.get("starting_adhesion_coefficient"),
        continuous_adhesion_coefficient=traction.get("continuous_adhesion_coefficient"),
    )


def load_locomotive_by_name(name: str, config_dir: str | Path) -> Locomotive:
    config_dir = Path(config_dir)
    slug = name.strip().lower().replace(" ", "_")
    path = config_dir / f"{slug}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No locomotive config found for '{name}' at {path}")
    return load_locomotive(path)
