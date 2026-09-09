"""Route/segment data model and infrastructure-compatibility checks (CLAUDE.md section 16, V0.5).

Trimmed vendored copy for the custom-locomotive API (2026-09-09): the real
src/elp_model/infrastructure.py also has CSV/Parquet ingestion (pandas-dependent); this API never
loads a route from a file -- it always receives already-resolved segments from the client (the same
segments the OD-pair map already fetched) -- so those loaders and the pandas dependency are dropped
here to keep the serverless function's cold start light. Source of truth for the real functionality
remains the main elp_locomotive_model project; keep this in sync with the dataclasses/compatibility
logic there if either changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class RouteDataError(ValueError):
    """Raised when a route file is malformed in a way that must not be silently skipped."""


@dataclass(frozen=True)
class Segment:
    segment_id: str
    distance_km: float
    gradient_pct: float
    electrified: bool
    speed_limit_kmh: float
    start_km: float | None = None
    end_km: float | None = None
    curve_radius_m: float | None = None
    electrification_system: str | None = None
    max_current_a: float | None = None
    max_power_kw: float | None = None
    axle_load_limit_t: float | None = None
    max_train_length_m: float | None = None
    country: str | None = None
    track_id: str | None = None
    direction: str | None = None
    source: str | None = None
    source_date: str | None = None
    quality_flag: str | None = None


@dataclass
class Route:
    route_id: str
    segments: list[Segment]
    warnings: list[str] = field(default_factory=list)

    def total_distance_km(self) -> float:
        return sum(s.distance_km for s in self.segments)


# ---------------------------------------------------------------------------
# V0.5: infrastructure-compatibility checks (CLAUDE.md section 16)
# ---------------------------------------------------------------------------

VOLTAGE_SYSTEM_NOMINAL_V = {
    "25kV_AC_50Hz": 25000.0,
    "15kV_AC_16.7Hz": 15000.0,
    "3kV_DC": 3000.0,
    "1.5kV_DC": 1500.0,
}


@dataclass(frozen=True)
class InfrastructureConstraintResult:
    compatible: bool
    reason: str | None
    usable_power_kw: float | None
    binding_source: str | None


def check_infrastructure_compatibility(
    segment: Segment,
    *,
    loco_voltage_systems: tuple[str, ...],
    loco_power_kw: float,
    axle_load_t: float | None,
    train_length_m: float | None,
) -> InfrastructureConstraintResult:
    if segment.electrified and segment.electrification_system is not None and loco_voltage_systems:
        if segment.electrification_system not in loco_voltage_systems:
            return InfrastructureConstraintResult(
                False,
                f"Locomotive not equipped for segment's electrification system ({segment.electrification_system})",
                None,
                "voltage_incompatible",
            )

    if segment.axle_load_limit_t is not None and axle_load_t is not None and axle_load_t > segment.axle_load_limit_t:
        return InfrastructureConstraintResult(
            False,
            f"Axle load {axle_load_t:.1f} t exceeds infrastructure limit {segment.axle_load_limit_t:.1f} t",
            None,
            "axle_load",
        )

    if (
        segment.max_train_length_m is not None
        and train_length_m is not None
        and train_length_m > segment.max_train_length_m
    ):
        return InfrastructureConstraintResult(
            False,
            f"Train length {train_length_m:.0f} m exceeds infrastructure limit {segment.max_train_length_m:.0f} m",
            None,
            "train_length",
        )

    usable_power_kw = loco_power_kw
    binding_source: str | None = None

    if segment.max_power_kw is not None and segment.max_power_kw < usable_power_kw:
        usable_power_kw = segment.max_power_kw
        binding_source = "infrastructure_power"

    if segment.max_current_a is not None and segment.electrification_system in VOLTAGE_SYSTEM_NOMINAL_V:
        voltage_v = VOLTAGE_SYSTEM_NOMINAL_V[segment.electrification_system]
        current_derived_kw = voltage_v * segment.max_current_a / 1000.0
        if current_derived_kw < usable_power_kw:
            usable_power_kw = current_derived_kw
            binding_source = "infrastructure_current"

    return InfrastructureConstraintResult(True, None, usable_power_kw, binding_source)
