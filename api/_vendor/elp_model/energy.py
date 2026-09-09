"""Traction energy calculation (CLAUDE.md sections 5 and 19).

Energy is charged only for positive traction work; negative (downhill) traction work is floored at
zero, matching the audited V0.2 behaviour. Regenerative/dynamic braking energy is explicitly
deferred to V0.4 (CLAUDE.md section 15) — a documented limitation, not a silent omission. Traction
and regenerated energy will be recorded separately once V0.4 lands.
"""
from __future__ import annotations


def traction_energy_kwh(required_resistance_kn: float, distance_km: float, traction_efficiency: float) -> float:
    """Energy = positive traction work / traction efficiency."""
    work_before_efficiency_kwh = max(required_resistance_kn, 0.0) * distance_km * 1000.0 * 1000.0 / 3_600_000.0
    return work_before_efficiency_kwh / traction_efficiency
