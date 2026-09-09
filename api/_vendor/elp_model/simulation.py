"""Segment-by-segment steady-state simulation (CLAUDE.md Part III).

This is the deterministic core: for every segment, resolve resistance, available tractive effort,
steady-state speed, journey time and traction energy, then record the first segment (if any) at
which the scenario becomes infeasible. No LLM is involved anywhere in this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from . import dynamics as dyn
from . import economics as eco
from . import energy as en
from . import traction as trac
from .infrastructure import Route
from .infrastructure import check_infrastructure_compatibility as infra_check
from .locomotives import Locomotive
from .trains import Train

ResistanceModel = Literal["v02_legacy", "davis"]
TractionModel = Literal["v02_legacy", "te_speed_curve"]


@dataclass(frozen=True)
class Consist:
    locomotive: Locomotive
    count: int

    def __post_init__(self) -> None:
        if self.locomotive.status == "PLACEHOLDER":
            raise ValueError(
                f"Locomotive '{self.locomotive.name}' has status PLACEHOLDER (no sourced data) and "
                "must not be used in a scenario."
            )
        if self.count < 1:
            raise ValueError("Consist count must be >= 1")


@dataclass(frozen=True)
class GlobalAssumptions:
    gravity_mps2: float
    legacy_rolling_resistance_n_per_t: float
    traction_efficiency: float
    electricity_price_eur_per_kwh: float
    diesel_price_eur_per_kwh: float
    min_driving_resistance_n: float
    default_adhesion_coefficient: float
    default_min_commercial_speed_kmh: float
    # V0.4 (CLAUDE.md section 15) — defaulted so existing V0.2/V0.3 call sites keep working unchanged.
    service_braking_rate_mps2: float = 0.5
    rotating_mass_factor: float = 1.08
    regen_efficiency: float = 0.9
    # V0.6 (CLAUDE.md section 19) — personnel cost.
    driver_eur_per_h: float = 0.0
    # Locomotive-equivalent demand translation (CLAUDE.md section 2/54) — same figure already baked
    # into every locomotive's maintenance_eur_per_km (see config/locomotives/*.yaml sources blocks),
    # now exposed as its own parameter so estimate_locomotive_demand.py can reuse it directly instead
    # of re-deriving it from a cost figure.
    annual_km_per_locomotive: float = 150000.0  # revised 2026-09-08, explicit user instruction (was 100000.0)
    # 2026-09-07 (CLAUDE.md discipline: never silently flatten a real country-to-country cost
    # difference) — country-specific traction electricity price overrides, ISO alpha-3 keyed (e.g.
    # "DEU": 0.21), sourced from a user-supplied 2026 assumptions table. Defaulted to an empty dict
    # so every existing hand-built `GlobalAssumptions(...)` (tests/conftest.py and friends) keeps
    # working unchanged -- an empty dict always falls back to `electricity_price_eur_per_kwh`, same
    # as before this field existed. Diesel deliberately has NO per-country equivalent: unlike grid
    # electricity (physically tied to where it's drawn), diesel is a portable, storable commodity --
    # a real operator fuels at the cheapest available point and carries it across borders, so
    # per-segment diesel attribution would misrepresent cost, not refine it (see
    # `diesel_price_eur_per_kwh`'s own notes in global_assumptions.yaml for the "smart fueling"
    # figure this project uses instead). Resolved per segment by
    # `economics.resolve_electricity_price_eur_per_kwh`.
    electricity_price_by_country_eur_per_kwh: dict[str, float] = field(default_factory=dict)
    # 2026-09-08 -- "diesel-relay" mechanism (Option A, operating_concepts.py): a pure-electric
    # locomotive on a segment with zero usable power gets a flat relay cost instead of being marked
    # infeasible outright. Defaulted to 0.0 so existing hand-built GlobalAssumptions(...) fixtures
    # keep working unchanged -- their synthetic routes are fully electrified/`country=None` anyway,
    # so `power_kw` never hits zero and this value is never actually applied for them. See
    # global_assumptions.yaml's own notes for the full derivation (no public commercial rate exists
    # for this vehicle class; the figure used is a caveated shunting-locomotive proxy).
    diesel_relay_eur_per_km: float = 0.0
    diesel_relay_attach_detach_cost_eur: float = 0.0
    diesel_relay_attach_detach_time_h: float = 0.0


_GLOBAL_ASSUMPTIONS_FIELDS = (
    "gravity_mps2",
    "legacy_rolling_resistance_n_per_t",
    "traction_efficiency",
    "electricity_price_eur_per_kwh",
    "diesel_price_eur_per_kwh",
    "min_driving_resistance_n",
    "default_adhesion_coefficient",
    "default_min_commercial_speed_kmh",
    "service_braking_rate_mps2",
    "rotating_mass_factor",
    "regen_efficiency",
    "driver_eur_per_h",
    "annual_km_per_locomotive",
    "electricity_price_by_country_eur_per_kwh",
    "diesel_relay_eur_per_km",
    "diesel_relay_attach_detach_cost_eur",
    "diesel_relay_attach_detach_time_h",
)


def load_global_assumptions(path: str | Path) -> GlobalAssumptions:
    """Loads config/economics/global_assumptions.yaml — the single source of truth for these
    parameters. Each YAML entry is a {value, unit, status, notes} block; only `value` is used here."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return GlobalAssumptions(**{name: raw[name]["value"] for name in _GLOBAL_ASSUMPTIONS_FIELDS})


@dataclass
class SegmentResult:
    scenario: str
    segment_id: str
    distance_km: float
    gradient_pct: float
    electrified: bool
    speed_limit_kmh: float
    feasible: bool
    failure_reason: str | None
    speed_kmh: float | None
    time_h: float | None
    energy_kwh: float | None
    energy_cost_eur: float | None
    required_resistance_kn: float
    available_te_kn: float | None
    binding_constraint: str | None


@dataclass
class ScenarioResult:
    scenario: str
    consist: Consist
    trailing_mass_t: float
    feasible: bool
    commercially_acceptable: bool
    first_failure_segment_id: str | None
    failure_reason: str | None
    distance_km: float
    journey_time_h: float | None
    energy_kwh: float | None
    min_speed_kmh: float | None
    critical_segment_id: str | None
    cost: eco.CostBreakdown | None
    segment_results: list[SegmentResult] = field(default_factory=list)


def run_scenario(
    *,
    scenario_name: str,
    route: Route,
    train: Train,
    consist: Consist,
    assumptions: GlobalAssumptions,
    resistance_model: ResistanceModel = "davis",
    traction_model: TractionModel = "te_speed_curve",
    adhesion_coefficient: float | None = None,
    min_commercial_speed_kmh: float | None = None,
) -> ScenarioResult:
    loco = consist.locomotive
    total_mass_t = train.trailing_mass_t + loco.mass_t * consist.count
    mu_starting, mu_continuous = trac.resolve_adhesion_coefficients(
        override=adhesion_coefficient,
        loco_starting=loco.starting_adhesion_coefficient,
        loco_continuous=loco.continuous_adhesion_coefficient,
        default=assumptions.default_adhesion_coefficient,
    )
    min_commercial_speed_kmh = (
        min_commercial_speed_kmh
        if min_commercial_speed_kmh is not None
        else assumptions.default_min_commercial_speed_kmh
    )
    if (resistance_model == "v02_legacy") != (traction_model == "v02_legacy"):
        raise ValueError(
            "v02_legacy must be used for both resistance_model and traction_model together, or not "
            "at all — mixing legacy and V0.3 physics is not a supported combination."
        )
    legacy_mode = resistance_model == "v02_legacy"

    segment_results: list[SegmentResult] = []
    first_failure_segment_id: str | None = None
    first_failure_reason: str | None = None
    total_time_h = 0.0
    total_energy_kwh = 0.0
    total_energy_cost_eur = 0.0
    any_infeasible = False
    min_speed_kmh: float | None = None
    critical_segment_id: str | None = None

    for seg in route.segments:
        power_kw = loco.power_kw(seg.electrified) * consist.count
        speed_cap_kmh = min(seg.speed_limit_kmh, loco.max_speed_kmh)
        infra_binding_source: str | None = None

        if not legacy_mode:
            infra = infra_check(
                seg,
                loco_voltage_systems=loco.voltage_systems,
                loco_power_kw=power_kw,
                axle_load_t=loco.axle_load_t(),
                train_length_m=train.train_length_m,
            )
            if not infra.compatible:
                result = dyn.SpeedSolveResult(False, None, infra.reason, 0.0, None, infra.binding_source)
                segment_results.append(
                    SegmentResult(
                        scenario=scenario_name,
                        segment_id=seg.segment_id,
                        distance_km=seg.distance_km,
                        gradient_pct=seg.gradient_pct,
                        electrified=seg.electrified,
                        speed_limit_kmh=seg.speed_limit_kmh,
                        feasible=False,
                        failure_reason=infra.reason,
                        speed_kmh=None,
                        time_h=None,
                        energy_kwh=None,
                        energy_cost_eur=None,
                        required_resistance_kn=0.0,
                        available_te_kn=None,
                        binding_constraint=infra.binding_source,
                    )
                )
                any_infeasible = True
                if first_failure_segment_id is None:
                    first_failure_segment_id = seg.segment_id
                    first_failure_reason = infra.reason
                continue
            power_kw = infra.usable_power_kw
            infra_binding_source = infra.binding_source

        if legacy_mode:
            result = dyn.solve_v02_legacy_speed_kmh(
                total_mass_t=total_mass_t,
                gradient_pct=seg.gradient_pct,
                power_kw=power_kw,
                max_te_kn=loco.max_starting_te_kn * consist.count,
                speed_cap_kmh=speed_cap_kmh,
                gravity_mps2=assumptions.gravity_mps2,
                legacy_rolling_resistance_n_per_t=assumptions.legacy_rolling_resistance_n_per_t,
                min_driving_resistance_n=assumptions.min_driving_resistance_n,
            )
        else:
            result = dyn.solve_davis_speed_kmh(
                total_mass_t=total_mass_t,
                gradient_pct=seg.gradient_pct,
                power_kw=power_kw,
                max_starting_te_kn=loco.max_starting_te_kn * consist.count,
                te_speed_curve=loco.te_speed_curve,
                adhesive_weight_t=loco.adhesive_weight_t * consist.count,
                # Continuous only -- see solve_davis_speed_kmh's own docstring for why the
                # steady-state solver never gets the starting-adhesion credit.
                adhesion_coefficient=mu_continuous,
                davis=train.davis,
                speed_cap_kmh=speed_cap_kmh,
                gravity_mps2=assumptions.gravity_mps2,
            )

        if result.feasible:
            time_h = seg.distance_km / result.speed_kmh
            energy_kwh = en.traction_energy_kwh(
                result.required_resistance_kn, seg.distance_km, assumptions.traction_efficiency
            )
            energy_cost = eco.energy_cost_eur(
                energy_kwh,
                seg.electrified,
                eco.resolve_electricity_price_eur_per_kwh(
                    seg.country, assumptions.electricity_price_by_country_eur_per_kwh,
                    assumptions.electricity_price_eur_per_kwh,
                ),
                assumptions.diesel_price_eur_per_kwh,
            )
            total_time_h += time_h
            total_energy_kwh += energy_kwh
            total_energy_cost_eur += energy_cost
            if min_speed_kmh is None or result.speed_kmh < min_speed_kmh:
                min_speed_kmh = result.speed_kmh
                critical_segment_id = seg.segment_id
        else:
            time_h = None
            energy_kwh = None
            energy_cost = None
            any_infeasible = True
            if first_failure_segment_id is None:
                first_failure_segment_id = seg.segment_id
                first_failure_reason = result.failure_reason

        binding_constraint = result.binding_constraint
        if binding_constraint == "power" and infra_binding_source is not None:
            # CLAUDE.md section 16: record when infrastructure power/current, not locomotive
            # capability, is what's actually limiting.
            binding_constraint = infra_binding_source

        segment_results.append(
            SegmentResult(
                scenario=scenario_name,
                segment_id=seg.segment_id,
                distance_km=seg.distance_km,
                gradient_pct=seg.gradient_pct,
                electrified=seg.electrified,
                speed_limit_kmh=seg.speed_limit_kmh,
                feasible=result.feasible,
                failure_reason=result.failure_reason,
                speed_kmh=result.speed_kmh,
                time_h=time_h,
                energy_kwh=energy_kwh,
                energy_cost_eur=energy_cost,
                required_resistance_kn=result.required_resistance_kn,
                available_te_kn=result.available_te_kn,
                binding_constraint=binding_constraint,
            )
        )

    feasible = not any_infeasible
    commercially_acceptable = feasible and (min_speed_kmh is None or min_speed_kmh >= min_commercial_speed_kmh)

    cost = None
    if feasible:
        cost = eco.compute_cost_breakdown(
            total_energy_cost_eur=total_energy_cost_eur,
            journey_time_h=total_time_h,
            distance_km=route.total_distance_km(),
            trailing_mass_t=train.trailing_mass_t,
            lease_eur_per_h=loco.lease_eur_per_h,
            maintenance_eur_per_km=loco.maintenance_eur_per_km,
            locomotive_count=consist.count,
            driver_eur_per_h=assumptions.driver_eur_per_h,
        )

    return ScenarioResult(
        scenario=scenario_name,
        consist=consist,
        trailing_mass_t=train.trailing_mass_t,
        feasible=feasible,
        commercially_acceptable=commercially_acceptable,
        first_failure_segment_id=first_failure_segment_id,
        failure_reason=first_failure_reason,
        distance_km=route.total_distance_km(),
        journey_time_h=total_time_h if feasible else None,
        energy_kwh=total_energy_kwh if feasible else None,
        min_speed_kmh=min_speed_kmh if feasible else None,
        critical_segment_id=critical_segment_id if feasible else None,
        cost=cost,
        segment_results=segment_results,
    )


# ---------------------------------------------------------------------------
# V0.4: dynamic (acceleration/braking) simulation (CLAUDE.md section 15)
# ---------------------------------------------------------------------------

_STALL_STEPS_LIMIT = 30  # consecutive zero-progress steps before declaring a segment infeasible

_FAILURE_REASON_BY_BINDING = {
    "power": "No available traction power",
    "tractive_effort_curve": "Insufficient tractive effort (nameplate/curve-limited)",
    "adhesion": "Insufficient tractive effort (adhesion-limited)",
}


@dataclass
class DynamicSegmentResult:
    scenario: str
    segment_id: str
    distance_km: float
    gradient_pct: float
    electrified: bool
    speed_limit_kmh: float
    feasible: bool
    failure_reason: str | None
    entry_speed_kmh: float
    exit_speed_kmh: float | None
    min_speed_kmh: float | None
    max_speed_kmh: float | None
    time_h: float | None
    traction_energy_kwh: float | None
    regen_energy_kwh: float | None
    net_energy_kwh: float | None


@dataclass
class DynamicScenarioResult:
    scenario: str
    consist: Consist
    trailing_mass_t: float
    feasible: bool
    commercially_acceptable: bool
    first_failure_segment_id: str | None
    failure_reason: str | None
    distance_km: float
    journey_time_h: float | None
    traction_energy_kwh: float | None
    regen_energy_kwh: float | None
    net_energy_kwh: float | None
    min_speed_kmh: float | None
    critical_segment_id: str | None
    cost: eco.CostBreakdown | None
    segment_results: list[DynamicSegmentResult] = field(default_factory=list)


def run_scenario_dynamic(
    *,
    scenario_name: str,
    route: Route,
    train: Train,
    consist: Consist,
    assumptions: GlobalAssumptions,
    adhesion_coefficient: float | None = None,
    min_commercial_speed_kmh: float | None = None,
    dt_s: float = 1.0,
    max_time_h: float = 24.0,
) -> DynamicScenarioResult:
    """V0.4 dynamic simulation: real acceleration/deceleration via F_net/effective_mass, a braking-
    distance lookahead so speed cannot jump instantaneously at a lower speed limit, and explicit
    coasting/regenerative/mechanical braking with traction and regenerated energy recorded
    separately (CLAUDE.md section 15).

    Unlike ``run_scenario`` (V0.3 steady state), this integrates position and speed continuously
    across segment boundaries, so a train can carry momentum into a short steep section it could not
    have started from rest — see data/validation/betuweroute_corridor.md for why this matters.

    ``dt_s`` should be small relative to typical segment length (at 100 km/h, 1.0 s covers ~28 m); on
    routes with very short segments — e.g. the 200 m synthetic V0.2/V0.3 regression route — a step
    may overshoot a segment boundary by a non-trivial fraction of that segment's length. The
    overshoot is simply attributed to the segment it started in rather than split proportionally;
    this is a deliberate simplification, acceptable given dt_s is user-configurable.
    """
    loco = consist.locomotive
    total_mass_t = train.trailing_mass_t + loco.mass_t * consist.count
    mu_starting, mu_continuous = trac.resolve_adhesion_coefficients(
        override=adhesion_coefficient,
        loco_starting=loco.starting_adhesion_coefficient,
        loco_continuous=loco.continuous_adhesion_coefficient,
        default=assumptions.default_adhesion_coefficient,
    )
    min_commercial_speed_kmh = (
        min_commercial_speed_kmh
        if min_commercial_speed_kmh is not None
        else assumptions.default_min_commercial_speed_kmh
    )

    segments = tuple(route.segments)
    if not segments:
        raise ValueError("Route has no segments")

    profile = dyn.compute_braking_profile(segments, loco.max_speed_kmh, assumptions.service_braking_rate_mps2)

    segment_results: list[DynamicSegmentResult] = []
    speed_kmh = 0.0
    total_time_s = 0.0
    total_traction_kwh = 0.0
    total_regen_kwh = 0.0
    total_energy_cost_eur = 0.0
    first_failure_segment_id: str | None = None
    first_failure_reason: str | None = None
    any_infeasible = False
    min_speed_kmh_overall: float | None = None
    critical_segment_id: str | None = None

    max_steps = int((max_time_h * 3600.0) / dt_s)
    steps_taken = 0

    for seg_index, seg in enumerate(segments):
        offset_km = 0.0
        seg_time_s = 0.0
        seg_traction_kwh = 0.0
        seg_regen_kwh = 0.0
        entry_speed_kmh = speed_kmh
        seg_min_speed = speed_kmh
        seg_max_speed = speed_kmh
        seg_infeasible = False
        seg_failure_reason: str | None = None
        stall_counter = 0
        power_kw = loco.power_kw(seg.electrified) * consist.count

        infra = infra_check(
            seg,
            loco_voltage_systems=loco.voltage_systems,
            loco_power_kw=power_kw,
            axle_load_t=loco.axle_load_t(),
            train_length_m=train.train_length_m,
        )
        if not infra.compatible:
            seg_infeasible = True
            seg_failure_reason = infra.reason
        else:
            power_kw = infra.usable_power_kw

        while not seg_infeasible and offset_km < seg.distance_km - 1e-9:
            if steps_taken >= max_steps:
                seg_infeasible = True
                seg_failure_reason = "Simulation exceeded max_time_h without completing the route"
                break

            # A lookahead-driven ceiling (an upcoming lower speed limit within braking distance)
            # keeps shrinking as the train approaches it — evaluating it only at the current position
            # would let "cruise at today's ceiling" lag a whole time step behind the tightening
            # constraint and overshoot the limit. Evaluating it a step further ahead too (using
            # current speed as the distance estimate) makes the train start braking early enough to
            # track it. This has no effect when the ceiling is flat (the local segment's own limit).
            lookahead_km = (speed_kmh / 3.6) * dt_s / 1000.0
            ceiling_kmh = min(
                dyn.ceiling_speed_kmh(
                    profile, segments, seg_index, offset_km, loco.max_speed_kmh, assumptions.service_braking_rate_mps2
                ),
                dyn.ceiling_speed_kmh(
                    profile,
                    segments,
                    seg_index,
                    min(offset_km + lookahead_km, seg.distance_km),
                    loco.max_speed_kmh,
                    assumptions.service_braking_rate_mps2,
                ),
            )
            step = dyn.simulate_step(
                speed_kmh=speed_kmh,
                ceiling_kmh=ceiling_kmh,
                total_mass_t=total_mass_t,
                gradient_pct=seg.gradient_pct,
                electrified=seg.electrified,
                power_kw=power_kw,
                max_starting_te_kn=loco.max_starting_te_kn * consist.count,
                te_speed_curve=loco.te_speed_curve,
                adhesive_weight_t=loco.adhesive_weight_t * consist.count,
                starting_adhesion_coefficient=mu_starting,
                continuous_adhesion_coefficient=mu_continuous,
                davis=train.davis,
                gravity_mps2=assumptions.gravity_mps2,
                rotating_mass_factor=assumptions.rotating_mass_factor,
                braking_rate_mps2=assumptions.service_braking_rate_mps2,
                regen_efficiency=assumptions.regen_efficiency,
                traction_efficiency=assumptions.traction_efficiency,
                dt_s=dt_s,
            )
            steps_taken += 1
            stall_counter = stall_counter + 1 if step.mode == "stalled" else 0

            offset_km += step.distance_km
            seg_time_s += dt_s
            seg_traction_kwh += step.traction_energy_kwh
            seg_regen_kwh += step.regen_energy_kwh
            speed_kmh = step.speed_kmh
            seg_min_speed = min(seg_min_speed, speed_kmh)
            seg_max_speed = max(seg_max_speed, speed_kmh)

            if stall_counter >= _STALL_STEPS_LIMIT:
                seg_infeasible = True
                if power_kw <= 0:
                    seg_failure_reason = _FAILURE_REASON_BY_BINDING["power"]
                else:
                    _, binding = trac.available_te_kn_with_binding(
                        power_kw=power_kw,
                        max_starting_te_kn=loco.max_starting_te_kn * consist.count,
                        te_speed_curve=loco.te_speed_curve,
                        adhesive_weight_t=loco.adhesive_weight_t * consist.count,
                        starting_adhesion_coefficient=mu_starting,
                        continuous_adhesion_coefficient=mu_continuous,
                        gravity_mps2=assumptions.gravity_mps2,
                        speed_kmh=max(speed_kmh, 0.1),
                    )
                    seg_failure_reason = _FAILURE_REASON_BY_BINDING.get(binding, "Insufficient tractive effort")
                break

        seg_net_kwh = seg_traction_kwh - seg_regen_kwh

        if seg_infeasible:
            any_infeasible = True
            if first_failure_segment_id is None:
                first_failure_segment_id = seg.segment_id
                first_failure_reason = seg_failure_reason
            segment_results.append(
                DynamicSegmentResult(
                    scenario=scenario_name,
                    segment_id=seg.segment_id,
                    distance_km=seg.distance_km,
                    gradient_pct=seg.gradient_pct,
                    electrified=seg.electrified,
                    speed_limit_kmh=seg.speed_limit_kmh,
                    feasible=False,
                    failure_reason=seg_failure_reason,
                    entry_speed_kmh=entry_speed_kmh,
                    exit_speed_kmh=None,
                    min_speed_kmh=None,
                    max_speed_kmh=None,
                    time_h=None,
                    traction_energy_kwh=None,
                    regen_energy_kwh=None,
                    net_energy_kwh=None,
                )
            )
            break

        energy_cost_this_seg = eco.energy_cost_eur(
            seg_net_kwh, seg.electrified,
            eco.resolve_electricity_price_eur_per_kwh(
                seg.country, assumptions.electricity_price_by_country_eur_per_kwh,
                assumptions.electricity_price_eur_per_kwh,
            ),
            assumptions.diesel_price_eur_per_kwh,
        )
        total_time_s += seg_time_s
        total_traction_kwh += seg_traction_kwh
        total_regen_kwh += seg_regen_kwh
        total_energy_cost_eur += energy_cost_this_seg

        if min_speed_kmh_overall is None or seg_min_speed < min_speed_kmh_overall:
            min_speed_kmh_overall = seg_min_speed
            critical_segment_id = seg.segment_id

        segment_results.append(
            DynamicSegmentResult(
                scenario=scenario_name,
                segment_id=seg.segment_id,
                distance_km=seg.distance_km,
                gradient_pct=seg.gradient_pct,
                electrified=seg.electrified,
                speed_limit_kmh=seg.speed_limit_kmh,
                feasible=True,
                failure_reason=None,
                entry_speed_kmh=entry_speed_kmh,
                exit_speed_kmh=speed_kmh,
                min_speed_kmh=seg_min_speed,
                max_speed_kmh=seg_max_speed,
                time_h=seg_time_s / 3600.0,
                traction_energy_kwh=seg_traction_kwh,
                regen_energy_kwh=seg_regen_kwh,
                net_energy_kwh=seg_net_kwh,
            )
        )

    feasible = not any_infeasible
    commercially_acceptable = feasible and (
        min_speed_kmh_overall is None or min_speed_kmh_overall >= min_commercial_speed_kmh
    )

    cost = None
    if feasible:
        cost = eco.compute_cost_breakdown(
            total_energy_cost_eur=total_energy_cost_eur,
            journey_time_h=total_time_s / 3600.0,
            distance_km=route.total_distance_km(),
            trailing_mass_t=train.trailing_mass_t,
            lease_eur_per_h=loco.lease_eur_per_h,
            maintenance_eur_per_km=loco.maintenance_eur_per_km,
            locomotive_count=consist.count,
            driver_eur_per_h=assumptions.driver_eur_per_h,
        )

    return DynamicScenarioResult(
        scenario=scenario_name,
        consist=consist,
        trailing_mass_t=train.trailing_mass_t,
        feasible=feasible,
        commercially_acceptable=commercially_acceptable,
        first_failure_segment_id=first_failure_segment_id,
        failure_reason=first_failure_reason,
        distance_km=route.total_distance_km(),
        journey_time_h=total_time_s / 3600.0 if feasible else None,
        traction_energy_kwh=total_traction_kwh if feasible else None,
        regen_energy_kwh=total_regen_kwh if feasible else None,
        net_energy_kwh=(total_traction_kwh - total_regen_kwh) if feasible else None,
        min_speed_kmh=min_speed_kmh_overall if feasible else None,
        critical_segment_id=critical_segment_id if feasible else None,
        cost=cost,
        segment_results=segment_results,
    )
