"""Train resistance models (CLAUDE.md sections 5 and 14).

Two resistance models are supported side by side:

- ``v02_legacy``: constant N/tonne rolling resistance, replicating the audited V0.2 Excel engine
  exactly (see docs/v02_model_audit.md). Retained only as a regression option per section 14.
- ``davis``: R = A + B*v + C*v^2 (Davis-type), the V0.3 default.

All forces are returned in newtons (N) unless the function name says otherwise.
"""
from __future__ import annotations


def gradient_force_n(total_mass_t: float, gradient_pct: float, gravity_mps2: float) -> float:
    """F_gradient = m * g * gradient. Positive gradient_pct (uphill) is a resisting force."""
    mass_kg = total_mass_t * 1000.0
    return mass_kg * gravity_mps2 * (gradient_pct / 100.0)


def legacy_rolling_resistance_n(total_mass_t: float, resistance_n_per_t: float) -> float:
    """V0.2 constant rolling resistance, independent of speed."""
    return total_mass_t * resistance_n_per_t


def davis_resistance_n(
    total_mass_t: float,
    speed_kmh: float,
    a_n_per_t: float,
    b_n_per_t_per_kmh: float,
    c_n_per_t_per_kmh2: float,
) -> float:
    """R = A + B*v + C*v^2. Coefficients are per tonne; speed is in km/h; result is in N."""
    per_tonne = a_n_per_t + b_n_per_t_per_kmh * speed_kmh + c_n_per_t_per_kmh2 * speed_kmh**2
    return total_mass_t * per_tonne
