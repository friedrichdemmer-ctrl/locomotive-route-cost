"""Cost model — kept separate from the physics engine (CLAUDE.md section 19).

Physics (simulation.py) outputs journey time, energy, distance, locomotive count, speed and
constraints. This module only converts those into cost; it must never call back into resistance,
traction or dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostBreakdown:
    energy_cost_eur: float
    lease_cost_eur: float
    maintenance_cost_eur: float
    driver_cost_eur: float
    operational_extras_cost_eur: float
    total_cost_eur: float
    eur_per_train_km: float
    eur_per_trailing_tonne_km: float


def driver_cost_eur(journey_time_h: float, driver_eur_per_h: float, num_drivers: int = 1) -> float:
    """Personnel cost group (CLAUDE.md section 19): driver time, second driver if required."""
    return driver_eur_per_h * num_drivers * journey_time_h


def energy_cost_eur(
    energy_kwh: float,
    electrified: bool,
    electricity_price_eur_per_kwh: float,
    diesel_price_eur_per_kwh: float,
) -> float:
    price = electricity_price_eur_per_kwh if electrified else diesel_price_eur_per_kwh
    return energy_kwh * price


def resolve_electricity_price_eur_per_kwh(
    country: str | None,
    price_by_country_eur_per_kwh: dict[str, float],
    default_price_eur_per_kwh: float,
) -> float:
    """Country-specific traction electricity price for one segment (2026-09-07, CLAUDE.md discipline:
    a real country-to-country cost difference must not be silently flattened). `country` is the
    segment's own ISO alpha-3 code (`Segment.country`, threaded from RINF's per-section country
    property via routing.hydrate_path_infrastructure) -- `None` for a segment with no known country
    (a manual cross-border bridge, a synthetic/test route) falls back to `default_price_eur_per_kwh`,
    same as any real country not yet in `price_by_country_eur_per_kwh`. Plain-argument signature
    (not `GlobalAssumptions` directly) to avoid a circular import between this module and
    simulation.py, which already imports `economics`.

    Deliberately electricity-only: diesel stays a single flat network-wide price (see
    `diesel_price_eur_per_kwh`'s notes in config/economics/global_assumptions.yaml) because diesel is
    a portable, storable commodity a real operator can buy wherever it's cheapest and carry across
    borders, unlike grid electricity, which is physically tied to wherever the pantograph draws it."""
    if country and country in price_by_country_eur_per_kwh:
        return price_by_country_eur_per_kwh[country]
    return default_price_eur_per_kwh


def compute_cost_breakdown(
    *,
    total_energy_cost_eur: float,
    journey_time_h: float,
    distance_km: float,
    trailing_mass_t: float,
    lease_eur_per_h: float,
    maintenance_eur_per_km: float,
    locomotive_count: int,
    driver_eur_per_h: float = 0.0,
    num_drivers: int = 1,
    operational_extras_cost_eur: float = 0.0,
) -> CostBreakdown:
    lease_cost = lease_eur_per_h * locomotive_count * journey_time_h
    maintenance_cost = maintenance_eur_per_km * locomotive_count * distance_km
    driver_cost = driver_cost_eur(journey_time_h, driver_eur_per_h, num_drivers)
    total_cost_eur = (
        total_energy_cost_eur + lease_cost + maintenance_cost + driver_cost + operational_extras_cost_eur
    )

    eur_per_train_km = total_cost_eur / distance_km if distance_km else float("nan")
    eur_per_trailing_tonne_km = (
        total_cost_eur / (distance_km * trailing_mass_t) if distance_km and trailing_mass_t else float("nan")
    )

    return CostBreakdown(
        energy_cost_eur=total_energy_cost_eur,
        lease_cost_eur=lease_cost,
        maintenance_cost_eur=maintenance_cost,
        driver_cost_eur=driver_cost,
        operational_extras_cost_eur=operational_extras_cost_eur,
        total_cost_eur=total_cost_eur,
        eur_per_train_km=eur_per_train_km,
        eur_per_trailing_tonne_km=eur_per_trailing_tonne_km,
    )
