"""Tractive-effort and power-limited traction models (CLAUDE.md sections 5 and 14).

Available TE = min(F_low_speed_max, F_power, F_adhesion):

- F_low_speed_max: sourced OEM TE/speed curve when available, else the constant nameplate max
  starting TE (CLAUDE.md: "Support actual OEM tractive-effort curve points ... when available").
- F_power: F = P / v, the power-limited tractive effort.
- F_adhesion: F = mu * adhesive_weight * g, with mu a scenario input (never an undocumented
  constant).

Real coefficient of adhesion is itself speed-dependent, not one flat constant (2026-08-26, added on
real EURO9000-specific figures: 0.41 starting from rest, 0.35 once already moving -- static/starting
friction genuinely exceeds rolling/continuous friction in real rail adhesion physics, and the gap
matters concretely for a route with a hard climb right at a station throat or after a signal stop).
`available_te_kn_with_binding`/`available_te_kn` therefore take TWO adhesion coefficients and select
between them by `speed_kmh`, at `STARTING_SPEED_THRESHOLD_KMH` -- below it, a segment is being
approached from (near) standstill and the higher starting coefficient applies; at or above it, the
train is already rolling and only the lower continuous coefficient applies. Locomotives without a
sourced starting/continuous split pass the same value for both (see `locomotives.Locomotive`), so
this collapses to the old single-coefficient behaviour unchanged."""
from __future__ import annotations

MIN_SPEED_KMH_FLOOR = 0.01  # avoids division by zero at standstill
STARTING_SPEED_THRESHOLD_KMH = 0.5  # below this, "starting from rest" adhesion physics applies


def power_limited_te_kn(power_kw: float, speed_kmh: float) -> float:
    """F_power = P / v."""
    speed_kmh = max(speed_kmh, MIN_SPEED_KMH_FLOOR)
    power_w = power_kw * 1000.0
    speed_mps = speed_kmh / 3.6
    force_n = power_w / speed_mps
    return force_n / 1000.0


def adhesion_limit_kn(adhesive_weight_t: float, adhesion_coefficient: float, gravity_mps2: float) -> float:
    """F_adhesion = mu * adhesive_weight * g."""
    adhesive_weight_kg = adhesive_weight_t * 1000.0
    force_n = adhesion_coefficient * adhesive_weight_kg * gravity_mps2
    return force_n / 1000.0


def te_curve_interpolate(curve: tuple, speed_kmh: float) -> float:
    """Linear interpolation over sourced OEM TE/speed points; flat extrapolation at the ends."""
    points = sorted(curve, key=lambda p: p.speed_kmh)
    if speed_kmh <= points[0].speed_kmh:
        return points[0].te_kn
    if speed_kmh >= points[-1].speed_kmh:
        return points[-1].te_kn
    for p0, p1 in zip(points, points[1:]):
        if p0.speed_kmh <= speed_kmh <= p1.speed_kmh:
            frac = (speed_kmh - p0.speed_kmh) / (p1.speed_kmh - p0.speed_kmh)
            return p0.te_kn + frac * (p1.te_kn - p0.te_kn)
    return points[-1].te_kn  # pragma: no cover - unreachable given the bounds checks above


def resolve_adhesion_coefficients(
    *, override: float | None, loco_starting: float | None, loco_continuous: float | None, default: float
) -> tuple[float, float]:
    """Resolves (starting_mu, continuous_mu) for one locomotive/scenario. An explicit `override`
    (e.g. a test's or OperatingConcept's caller-supplied what-if value) applies uniformly to both,
    preserving the old "one blunt value" semantics; only when no override is given does a
    locomotive's own sourced starting/continuous split (see `locomotives.Locomotive`) apply, each
    falling back to `default` (typically GlobalAssumptions.default_adhesion_coefficient)
    individually if that locomotive doesn't have a sourced value for it."""
    if override is not None:
        return override, override
    return (
        loco_starting if loco_starting is not None else default,
        loco_continuous if loco_continuous is not None else default,
    )


def _effective_adhesion_coefficient(
    starting_adhesion_coefficient: float, continuous_adhesion_coefficient: float, speed_kmh: float
) -> float:
    return (
        starting_adhesion_coefficient
        if speed_kmh < STARTING_SPEED_THRESHOLD_KMH
        else continuous_adhesion_coefficient
    )


def available_te_kn_with_binding(
    *,
    power_kw: float,
    max_starting_te_kn: float,
    te_speed_curve: tuple,
    adhesive_weight_t: float,
    starting_adhesion_coefficient: float,
    continuous_adhesion_coefficient: float,
    gravity_mps2: float,
    speed_kmh: float,
) -> tuple[float, str]:
    """Returns (available_te_kn, binding_constraint) where binding_constraint is whichever of
    'tractive_effort_curve', 'power' or 'adhesion' produced the minimum (CLAUDE.md section 27:
    every result must expose its binding constraint)."""
    f_low_speed_max = (
        te_curve_interpolate(te_speed_curve, speed_kmh) if te_speed_curve else max_starting_te_kn
    )
    f_power = power_limited_te_kn(power_kw, speed_kmh)
    mu = _effective_adhesion_coefficient(starting_adhesion_coefficient, continuous_adhesion_coefficient, speed_kmh)
    f_adhesion = adhesion_limit_kn(adhesive_weight_t, mu, gravity_mps2)

    candidates = {
        "tractive_effort_curve": f_low_speed_max,
        "power": f_power,
        "adhesion": f_adhesion,
    }
    binding = min(candidates, key=candidates.get)
    return candidates[binding], binding


def available_te_kn(
    *,
    power_kw: float,
    max_starting_te_kn: float,
    te_speed_curve: tuple,
    adhesive_weight_t: float,
    starting_adhesion_coefficient: float,
    continuous_adhesion_coefficient: float,
    gravity_mps2: float,
    speed_kmh: float,
) -> float:
    value, _binding = available_te_kn_with_binding(
        power_kw=power_kw,
        max_starting_te_kn=max_starting_te_kn,
        te_speed_curve=te_speed_curve,
        adhesive_weight_t=adhesive_weight_t,
        starting_adhesion_coefficient=starting_adhesion_coefficient,
        continuous_adhesion_coefficient=continuous_adhesion_coefficient,
        gravity_mps2=gravity_mps2,
        speed_kmh=speed_kmh,
    )
    return value
