"""Speed resolution (CLAUDE.md section 5): a steady-state solver (V0.3) and a step-by-step kinematic
integrator (V0.4).

V0.3's ``solve_*_speed_kmh`` functions solve for the maximum sustained speed at which available
tractive effort covers total resistance on a single segment, in isolation — a train is assumed to
instantly reach that equilibrium. This is exact for a sustained climb but cannot represent momentum:
CLAUDE.md's own Betuweroute validation (data/validation/betuweroute_corridor.md) found a short, steep
tunnel pinch inside an otherwise flat corridor scored as infeasible under V0.3, purely because the
steady-state solver has no notion of "already moving fast when the grade starts."

V0.4 (CLAUDE.md section 15) adds real F_net/effective_mass acceleration and deceleration, integrated
at a configurable time step, with a braking-distance lookahead so speed cannot jump instantaneously
at a lower speed limit, and explicit traction/coasting/regenerative-braking/mechanical-braking mode
selection with traction and regenerated energy recorded separately.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import resistance as res
from . import traction as trac
from .infrastructure import Segment
from .trains import DavisCoefficients

_BISECTION_ITERATIONS = 60
_MIN_SPEED_KMH = 0.1
_STALL_SPEED_THRESHOLD_MPS = 0.5 / 3.6  # 0.5 km/h — "stalled" only applies near genuine standstill


@dataclass(frozen=True)
class SpeedSolveResult:
    feasible: bool
    speed_kmh: float | None
    failure_reason: str | None
    required_resistance_kn: float
    available_te_kn: float | None
    binding_constraint: str | None


def solve_v02_legacy_speed_kmh(
    *,
    total_mass_t: float,
    gradient_pct: float,
    power_kw: float,
    max_te_kn: float,
    speed_cap_kmh: float,
    gravity_mps2: float,
    legacy_rolling_resistance_n_per_t: float,
    min_driving_resistance_n: float,
) -> SpeedSolveResult:
    """Exact replica of the V0.2 Excel Calc-sheet engine — see docs/v02_model_audit.md."""
    f_gradient_n = res.gradient_force_n(total_mass_t, gradient_pct, gravity_mps2)
    f_rolling_n = res.legacy_rolling_resistance_n(total_mass_t, legacy_rolling_resistance_n_per_t)
    resistance_kn = (f_gradient_n + f_rolling_n) / 1000.0

    if power_kw <= 0:
        return SpeedSolveResult(False, None, "No available traction power", resistance_kn, None, "power")
    if resistance_kn > max_te_kn:
        return SpeedSolveResult(
            False, None, "Insufficient tractive effort", resistance_kn, max_te_kn, "tractive_effort"
        )

    resistance_n_floored = max(resistance_kn * 1000.0, min_driving_resistance_n)
    theoretical_speed_kmh = (power_kw * 1000.0) / resistance_n_floored * 3.6
    actual_speed_kmh = max(min(theoretical_speed_kmh, speed_cap_kmh), _MIN_SPEED_KMH)
    binding = "speed_limit" if actual_speed_kmh >= speed_cap_kmh - 1e-9 else "power"

    return SpeedSolveResult(True, actual_speed_kmh, None, resistance_kn, max_te_kn, binding)


def solve_davis_speed_kmh(
    *,
    total_mass_t: float,
    gradient_pct: float,
    power_kw: float,
    max_starting_te_kn: float,
    te_speed_curve: tuple,
    adhesive_weight_t: float,
    adhesion_coefficient: float,
    davis: DavisCoefficients,
    speed_cap_kmh: float,
    gravity_mps2: float,
) -> SpeedSolveResult:
    """V0.3 default: Davis resistance + power-limited/adhesion-limited traction.

    Available TE(v) is non-increasing in v (power-limited term falls as 1/v; the OEM curve/nameplate
    term and the adhesion term are both independent of speed) and required resistance(v) is
    non-decreasing in v for non-negative Davis B, C coefficients. The steady-state speed is therefore
    the unique crossing point, found by bisection.

    Deliberately takes ONE adhesion coefficient, not a starting/continuous pair (2026-08-26): this
    solver has no notion of "already moving" at all (that is precisely why V0.4's dynamic engine
    exists — see this module's own top docstring) and, by the same reasoning, no notion of "just
    starting from rest" either. A first attempt threaded a real starting-vs-continuous split into
    this function's own initial near-zero-speed feasibility check and broke two things at once: the
    bisection's own non-increasing-TE(v) assumption above (a coefficient that steps down at a speed
    threshold is not monotonic), and, concretely, the Betuweroute regression test asserting the
    steady-state solver still correctly fails a short pinch — exactly the case the starting credit
    should NOT rescue here, since that credit belongs only to the dynamic engine's real per-timestep
    speed tracking (`simulate_step`), where "just starting from rest" is a real, well-defined
    instant, not a steady-state solver's own initial-guess artifact."""
    f_gradient_n = res.gradient_force_n(total_mass_t, gradient_pct, gravity_mps2)

    def required_resistance_kn(v_kmh: float) -> float:
        f_davis_n = res.davis_resistance_n(
            total_mass_t, v_kmh, davis.a_n_per_t, davis.b_n_per_t_per_kmh, davis.c_n_per_t_per_kmh2
        )
        return (f_gradient_n + f_davis_n) / 1000.0

    def available_with_binding(v_kmh: float) -> tuple[float, str]:
        return trac.available_te_kn_with_binding(
            power_kw=power_kw,
            max_starting_te_kn=max_starting_te_kn,
            te_speed_curve=te_speed_curve,
            adhesive_weight_t=adhesive_weight_t,
            starting_adhesion_coefficient=adhesion_coefficient,
            continuous_adhesion_coefficient=adhesion_coefficient,
            gravity_mps2=gravity_mps2,
            speed_kmh=v_kmh,
        )

    if power_kw <= 0:
        req = required_resistance_kn(_MIN_SPEED_KMH)
        return SpeedSolveResult(False, None, "No available traction power", req, None, "power")

    speed_cap_kmh = max(speed_cap_kmh, _MIN_SPEED_KMH)
    avail_at_start, binding_at_start = available_with_binding(_MIN_SPEED_KMH)
    req_at_start = required_resistance_kn(_MIN_SPEED_KMH)
    if avail_at_start < req_at_start:
        reason_by_binding = {
            "tractive_effort_curve": "Insufficient tractive effort (nameplate/curve-limited)",
            "power": "Insufficient tractive effort (power-limited)",
            "adhesion": "Insufficient tractive effort (adhesion-limited)",
        }
        return SpeedSolveResult(
            False,
            None,
            reason_by_binding[binding_at_start],
            req_at_start,
            avail_at_start,
            binding_at_start,
        )

    lo, hi = _MIN_SPEED_KMH, speed_cap_kmh
    for _ in range(_BISECTION_ITERATIONS):
        mid = (lo + hi) / 2.0
        avail_mid, _ = available_with_binding(mid)
        if avail_mid >= required_resistance_kn(mid):
            lo = mid
        else:
            hi = mid

    speed_kmh = lo
    available_kn, binding = available_with_binding(speed_kmh)
    if speed_kmh >= speed_cap_kmh - 1e-6:
        binding = "speed_limit"

    return SpeedSolveResult(
        True, speed_kmh, None, required_resistance_kn(speed_kmh), available_kn, binding
    )


# ---------------------------------------------------------------------------
# V0.4: braking-distance lookahead (CLAUDE.md section 15: "model transitions into lower speed
# limits so speed cannot change instantaneously")
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrakingProfile:
    """Backward-computed speed ceiling at the *start* of each segment: the fastest a train may be
    going there and still be able to decelerate, at ``braking_rate_mps2``, to satisfy every
    segment's own speed limit from that point onward."""

    ceiling_at_start_kmh: tuple[float, ...]


def _braking_ceiling_kmh(distance_km: float, target_speed_kmh: float, braking_rate_mps2: float) -> float:
    """The fastest speed from which a train can decelerate to ``target_speed_kmh`` at
    ``braking_rate_mps2`` within ``distance_km``: v = sqrt(v_target^2 + 2*b*d)."""
    target_mps = target_speed_kmh / 3.6
    distance_m = max(distance_km, 0.0) * 1000.0
    ceiling_mps = (target_mps**2 + 2.0 * braking_rate_mps2 * distance_m) ** 0.5
    return ceiling_mps * 3.6


def compute_braking_profile(
    segments: tuple[Segment, ...], locomotive_max_speed_kmh: float, braking_rate_mps2: float
) -> BrakingProfile:
    n = len(segments)
    ceiling_at_start = [0.0] * n
    # Floored at _MIN_SPEED_KMH, same defensive floor solve_davis_speed_kmh already applies to
    # speed_cap_kmh -- a real RINF segment can assert a literal 0 speed_limit_kmh (confirmed live
    # 2026-08-26 on a real Budapest-Szeged corridor section, almost certainly a data quirk -- e.g. an
    # unset/placeholder value or a headshunt/dead-end not meant for through running -- not a genuine
    # "trains must stop here forever" constraint). Without this floor, the ceiling here is exactly
    # 0 and the step simulator can never leave it (v=0, ceiling=0 forever), hanging the simulation
    # until max_time_h is exhausted rather than crawling through at a nominal floor speed the way the
    # steady-state solver already does.
    next_start_ceiling = max(min(segments[-1].speed_limit_kmh, locomotive_max_speed_kmh), _MIN_SPEED_KMH)
    for i in range(n - 1, -1, -1):
        local_limit = max(min(segments[i].speed_limit_kmh, locomotive_max_speed_kmh), _MIN_SPEED_KMH)
        reachable = _braking_ceiling_kmh(segments[i].distance_km, next_start_ceiling, braking_rate_mps2)
        ceiling_at_start[i] = min(local_limit, reachable)
        next_start_ceiling = ceiling_at_start[i]
    return BrakingProfile(tuple(ceiling_at_start))


def ceiling_speed_kmh(
    profile: BrakingProfile,
    segments: tuple[Segment, ...],
    segment_index: int,
    offset_km: float,
    locomotive_max_speed_kmh: float,
    braking_rate_mps2: float,
) -> float:
    """Max permitted speed at ``offset_km`` into ``segments[segment_index]``: respects that
    segment's own limit and the ability to brake, within the remaining distance, to whatever the
    next segment requires. Floored at _MIN_SPEED_KMH -- see compute_braking_profile's own comment."""
    seg = segments[segment_index]
    local_limit = max(min(seg.speed_limit_kmh, locomotive_max_speed_kmh), _MIN_SPEED_KMH)
    remaining_km = max(seg.distance_km - offset_km, 0.0)
    next_ceiling = (
        profile.ceiling_at_start_kmh[segment_index + 1] if segment_index + 1 < len(segments) else local_limit
    )
    reachable = _braking_ceiling_kmh(remaining_km, next_ceiling, braking_rate_mps2)
    return min(local_limit, reachable)


# ---------------------------------------------------------------------------
# V0.4: per-step kinematic integration (CLAUDE.md section 15)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DynamicStep:
    distance_km: float
    speed_kmh: float  # speed at the END of this step
    mode: str  # "coasting" | "accelerating" | "cruising" | "power_limited_cruise" | "holding_brake" | "braking" | "stalled"
    traction_force_kn: float
    brake_force_kn: float
    traction_energy_kwh: float
    regen_energy_kwh: float


def simulate_step(
    *,
    speed_kmh: float,
    ceiling_kmh: float,
    total_mass_t: float,
    gradient_pct: float,
    electrified: bool,
    power_kw: float,
    max_starting_te_kn: float,
    te_speed_curve: tuple,
    adhesive_weight_t: float,
    starting_adhesion_coefficient: float,
    continuous_adhesion_coefficient: float,
    davis: DavisCoefficients,
    gravity_mps2: float,
    rotating_mass_factor: float,
    braking_rate_mps2: float,
    regen_efficiency: float,
    traction_efficiency: float,
    dt_s: float,
) -> DynamicStep:
    """Advance one time step. Chooses a mode (coasting/accelerating/cruising/braking/holding_brake)
    based on current speed versus the speed ceiling for this position, applies F_net/effective_mass,
    and returns the resulting speed, distance covered and energy (traction and regenerated,
    recorded separately per CLAUDE.md section 15)."""
    v_mps = speed_kmh / 3.6
    ceiling_mps = ceiling_kmh / 3.6
    effective_mass_kg = total_mass_t * 1000.0 * rotating_mass_factor

    f_gradient_n = res.gradient_force_n(total_mass_t, gradient_pct, gravity_mps2)
    f_davis_n = res.davis_resistance_n(
        total_mass_t, speed_kmh, davis.a_n_per_t, davis.b_n_per_t_per_kmh, davis.c_n_per_t_per_kmh2
    )
    f_resistance_n = f_gradient_n + f_davis_n
    f_natural_n = -f_resistance_n  # net force with zero traction and zero brake

    def available_te_n() -> float:
        return (
            trac.available_te_kn(
                power_kw=power_kw,
                max_starting_te_kn=max_starting_te_kn,
                te_speed_curve=te_speed_curve,
                adhesive_weight_t=adhesive_weight_t,
                starting_adhesion_coefficient=starting_adhesion_coefficient,
                continuous_adhesion_coefficient=continuous_adhesion_coefficient,
                gravity_mps2=gravity_mps2,
                speed_kmh=speed_kmh,
            )
            * 1000.0
        )

    traction_force_n = 0.0
    brake_force_n = 0.0

    if v_mps < ceiling_mps - 1e-6:
        if f_natural_n > 0:
            mode = "coasting"  # gravity alone accelerates toward the ceiling; draw no power
        else:
            traction_force_n = available_te_n()
            mode = "accelerating"
        a_mps2 = (traction_force_n + f_natural_n) / effective_mass_kg
        v_new_mps = max(min(v_mps + a_mps2 * dt_s, ceiling_mps), 0.0)
        # "stalled" means genuinely stuck near standstill and unable to pull away — not merely
        # decelerating under load at speed, which is normal (and may settle at a lower but perfectly
        # sustainable cruising speed).
        if mode == "accelerating" and a_mps2 <= 1e-6 and v_new_mps <= _STALL_SPEED_THRESHOLD_MPS:
            mode = "stalled"

    elif v_mps > ceiling_mps + 1e-6:
        v_new_mps = max(v_mps - braking_rate_mps2 * dt_s, ceiling_mps)
        a_applied = (v_new_mps - v_mps) / dt_s if dt_s > 0 else -braking_rate_mps2
        brake_force_n = max(f_natural_n - effective_mass_kg * a_applied, 0.0)
        mode = "braking"

    else:
        f_needed_n = -f_natural_n  # force needed to hold speed exactly (a = 0)
        if f_needed_n >= 0:
            avail_n = available_te_n()
            traction_force_n = min(f_needed_n, avail_n)
            mode = "cruising" if traction_force_n >= f_needed_n - 1.0 else "power_limited_cruise"
            a_mps2 = (traction_force_n + f_natural_n) / effective_mass_kg
            v_new_mps = max(v_mps + a_mps2 * dt_s, 0.0)
        else:
            brake_force_n = f_natural_n  # positive: downhill assist alone would exceed the ceiling
            mode = "holding_brake"
            v_new_mps = v_mps

    distance_km = ((v_mps + v_new_mps) / 2.0) * dt_s / 1000.0

    traction_energy_kwh = 0.0
    if traction_force_n > 0:
        traction_energy_kwh = (
            (traction_force_n / 1000.0) * distance_km * 1_000_000.0 / 3_600_000.0 / traction_efficiency
        )

    regen_energy_kwh = 0.0
    if brake_force_n > 0 and electrified:
        regen_capacity_kn = available_te_n() / 1000.0  # proxy for max regen capacity — see docs
        regen_force_kn = min(brake_force_n / 1000.0, regen_capacity_kn)
        regen_energy_kwh = regen_force_kn * distance_km * 1_000_000.0 / 3_600_000.0 * regen_efficiency

    return DynamicStep(
        distance_km=distance_km,
        speed_kmh=v_new_mps * 3.6,
        mode=mode,
        traction_force_kn=traction_force_n / 1000.0,
        brake_force_kn=brake_force_n / 1000.0,
        traction_energy_kwh=traction_energy_kwh,
        regen_energy_kwh=regen_energy_kwh,
    )
