"""Operating concepts beyond a single repeated locomotive type (CLAUDE.md sections 3, 17, 19, 20).

CLAUDE.md section 3 is explicit that the relevant comparison is "EURO9000 versus the best realistic
alternative operating concept," which may include a mainline locomotive plus a banker/helper on the
critical gradient — attached only where needed, then detached, not hauled for the whole route. That
is the single most important discontinuity this module exists to price correctly: a helper attached
for 20 km of a 500 km route costs far less than doubling the mainline locomotive for the whole route
(CLAUDE.md section 20's "double-traction threshold" and "helper threshold").

This module evaluates such concepts using the V0.3 steady-state solver (``dynamics.solve_davis_speed_kmh``,
``run_operating_concept``), segment by segment, with the *combined* power/tractive-effort/adhesive-weight
of whichever units are attached on that segment. Since 2026-08-26, a second, opt-in engine
(``run_operating_concept_dynamic``) additionally integrates V0.4's momentum-aware dynamics
(``dynamics.simulate_step``) for the same multi-unit/helper concepts — a real short pinch a train
cannot climb from a standing start can still be crossed carrying speed from before it, the same
"already moving" credit CLAUDE.md's Betuweroute validation gave a single locomotive. It does not
(yet) integrate with V0.5's axle-load/train-length/infrastructure-power-ceiling checks
(``infrastructure.check_infrastructure_compatibility``) — a documented scope boundary, not a silent
gap. Voltage-system compatibility specifically *is* wired in (2026-08-26, see ``_usable_power_kw``):
a locomotive is only credited with electric power on segments whose electrification system it
actually supports, falling back to diesel power (or to infeasibility) otherwise. This was the first
locomotive-comparison bug this mattered for — a single-voltage locomotive (Vectron Dual Mode) being
wrongly treated as compatible everywhere a multi-system locomotive would be.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import dynamics as dyn
from . import economics as eco
from . import energy as en
from . import traction as trac
from .infrastructure import Route, Segment
from .locomotives import Locomotive
from .simulation import _FAILURE_REASON_BY_BINDING, _STALL_STEPS_LIMIT, GlobalAssumptions
from .trains import Train


@dataclass(frozen=True)
class HelperPolicy:
    """A banking/helper locomotive attached only on segments the PRIMARY consist genuinely cannot
    clear alone at this train's weight (CLAUDE.md section 20's "helper threshold" — a real
    capability question, not a fixed gradient rule). See ``_primary_alone_feasible`` for the actual
    trigger, called from both ``run_operating_concept`` and ``run_operating_concept_dynamic``.

    ``trigger_gradient_pct`` is now UNUSED by the trigger decision — kept only because removing it
    would break every existing call site's keyword construction (and ``scripts/run_mass_sweep.py``
    exposes it as a CLI flag). Real fix, 2026-09-06: this used to be compared directly against a
    segment's gradient (`gradient_pct >= trigger_gradient_pct`), a fixed number
    (`docs/assumptions_register.md`'s own note: reverse-engineered to fire on this project's four
    validation corridors' 2.2-2.6% ruling gradients, not derived from any capability calculation) —
    meaning a helper billed on every 2%+ segment regardless of whether THIS train's weight actually
    needed it there, and never billed below 2% even for a train too heavy to clear a gentler grade.
    All cost/time figures are ``ANALYST_ASSUMPTION`` — see config/economics/global_assumptions.yaml's
    helper defaults for sourcing notes.
    """

    locomotive: Locomotive
    count: int = 1
    trigger_gradient_pct: float = 2.0
    attach_detach_time_h: float = 0.25
    attach_detach_cost_eur: float = 150.0


@dataclass(frozen=True)
class OperatingConcept:
    name: str
    primary_locomotive: Locomotive
    primary_count: int = 1
    helper_policy: HelperPolicy | None = None
    adhesion_coefficient: float | None = None


@dataclass
class ConceptSegmentResult:
    segment_id: str
    distance_km: float
    gradient_pct: float
    helper_active: bool
    # 2026-09-08, "diesel-relay" mechanism (Option A): True when the primary (+ active helper, if
    # any) has zero usable power on this segment (see _combined_capability) -- a real electrification
    # gap a pure-electric locomotive cannot cross under its own power. The segment is priced via
    # diesel_relay_eur_per_km instead of being run through the physics solver at all; see
    # run_operating_concept's own per-segment loop for the exact branch. Mutually exclusive with
    # helper_active: a same-type helper achieves nothing on a segment with no usable power, so it is
    # forced False here rather than double-billed alongside a relay charge.
    relay_active: bool
    feasible: bool
    failure_reason: str | None
    speed_kmh: float | None
    time_h: float | None
    energy_kwh: float | None
    energy_cost_eur: float | None
    binding_constraint: str | None


@dataclass
class OperatingConceptResult:
    concept_name: str
    feasible: bool
    commercially_acceptable: bool
    first_failure_segment_id: str | None
    failure_reason: str | None
    distance_km: float
    journey_time_h: float | None
    energy_kwh: float | None
    helper_distance_km: float
    helper_attach_events: int
    # 2026-09-08, diesel-relay mechanism -- mirrors helper_distance_km/helper_attach_events exactly,
    # for relay-hauled segments instead of helper-assisted ones (see ConceptSegmentResult.relay_active).
    relay_distance_km: float
    relay_attach_events: int
    min_speed_kmh: float | None
    critical_segment_id: str | None
    cost: eco.CostBreakdown | None
    # ConceptSegmentResult for run_operating_concept (steady-state); DynamicConceptSegmentResult for
    # run_operating_concept_dynamic -- the two engines' segment-level shapes genuinely differ (a
    # single steady speed vs entry/exit/min/max, combined vs separately-tracked traction/regen
    # energy), so this is deliberately not narrowed to one type.
    segment_results: list[ConceptSegmentResult] | list["DynamicConceptSegmentResult"] = field(default_factory=list)


def _usable_power_kw(loco: Locomotive, seg: Segment) -> tuple[bool, float]:
    """Returns (used_electric, power_kw) for one locomotive on one segment, honouring voltage-system
    compatibility (CLAUDE.md section 16 / V0.5, wired in 2026-08-26): if the segment is electrified
    under a system the locomotive doesn't declare support for, it falls back to diesel power (if the
    locomotive has a diesel engine) rather than being silently credited with electric power it cannot
    actually draw. A locomotive with an empty voltage_systems list (not yet sourced for that param)
    is treated as unconstrained, preserving prior behaviour for those configs."""
    if seg.electrified and seg.electrification_system is not None and loco.voltage_systems:
        if seg.electrification_system not in loco.voltage_systems:
            return (False, loco.diesel_max_power_kw) if loco.has_diesel() else (False, 0.0)
    return seg.electrified, loco.power_kw(seg.electrified)


def _combined_capability(
    primary: Locomotive, primary_count: int, helper: Locomotive | None, helper_count: int, seg: Segment
) -> tuple[float, float, float, float, bool]:
    """Returns (power_kw, max_starting_te_kn, adhesive_weight_t, loco_mass_t, used_electric) summed
    across whichever units are active. used_electric reflects the actual energy mode used (see
    _usable_power_kw) for energy-cost pricing, which can differ from seg.electrified when the
    locomotive's voltage system doesn't cover this segment. Curve-based TE is not combined across
    heterogeneous units — CLAUDE.md's three locomotives have no sourced curve anyway (see
    docs/assumptions_register.md), so this only ever falls back to summed nameplate TE in practice."""
    used_electric, primary_power_kw = _usable_power_kw(primary, seg)
    power_kw = primary_power_kw * primary_count
    te_kn = primary.max_starting_te_kn * primary_count
    adhesive_t = primary.adhesive_weight_t * primary_count
    mass_t = primary.mass_t * primary_count
    if helper is not None:
        _, helper_power_kw = _usable_power_kw(helper, seg)
        power_kw += helper_power_kw * helper_count
        te_kn += helper.max_starting_te_kn * helper_count
        adhesive_t += helper.adhesive_weight_t * helper_count
        mass_t += helper.mass_t * helper_count
    return power_kw, te_kn, adhesive_t, mass_t, used_electric


def _primary_alone_feasible(
    primary: Locomotive,
    primary_count: int,
    seg: Segment,
    train: Train,
    mu_continuous: float,
    speed_cap_kmh: float,
    assumptions: GlobalAssumptions,
) -> bool:
    """The real helper trigger (CLAUDE.md section 20): can the PRIMARY consist alone — no helper —
    actually move this train's weight over this one segment? Reuses the exact same continuous-speed
    solver (`dyn.solve_davis_speed_kmh`) both engines already use to decide feasibility for whatever
    consist is currently attached, just asked about the primary-only consist first. If it can, no
    helper is needed here regardless of how steep the segment is; if it can't, a helper is needed
    here regardless of how gentle the segment is — weight-and-capability-driven, not a fixed gradient
    number."""
    power_kw, te_kn, adhesive_t, primary_mass_t, _used_electric = _combined_capability(
        primary, primary_count, None, 0, seg
    )
    result = dyn.solve_davis_speed_kmh(
        total_mass_t=train.trailing_mass_t + primary_mass_t,
        gradient_pct=seg.gradient_pct,
        power_kw=power_kw,
        max_starting_te_kn=te_kn,
        te_speed_curve=(),
        adhesive_weight_t=adhesive_t,
        # Continuous only -- same reasoning as the combined-consist solve below: the steady-state
        # solver never gets the starting-adhesion credit (see solve_davis_speed_kmh's own docstring).
        adhesion_coefficient=mu_continuous,
        davis=train.davis,
        speed_cap_kmh=speed_cap_kmh,
        gravity_mps2=assumptions.gravity_mps2,
    )
    return result.feasible


def run_operating_concept(
    *,
    concept: OperatingConcept,
    route: Route,
    train: Train,
    assumptions: GlobalAssumptions,
    min_commercial_speed_kmh: float | None = None,
) -> OperatingConceptResult:
    primary = concept.primary_locomotive
    helper_policy = concept.helper_policy
    # Resolved from the PRIMARY locomotive only -- a real simplification for a heterogeneous
    # primary+helper consist with genuinely different sourced coefficients, but every helper concept
    # in this project uses the same locomotive type as its own helper, matching real-world practice,
    # so this doesn't lose precision in the cases actually modelled.
    mu_starting, mu_continuous = trac.resolve_adhesion_coefficients(
        override=concept.adhesion_coefficient,
        loco_starting=primary.starting_adhesion_coefficient,
        loco_continuous=primary.continuous_adhesion_coefficient,
        default=assumptions.default_adhesion_coefficient,
    )
    min_commercial_speed_kmh = (
        min_commercial_speed_kmh
        if min_commercial_speed_kmh is not None
        else assumptions.default_min_commercial_speed_kmh
    )

    speed_cap_base_kmh = primary.max_speed_kmh
    if helper_policy is not None:
        speed_cap_base_kmh = min(speed_cap_base_kmh, helper_policy.locomotive.max_speed_kmh)

    segment_results: list[ConceptSegmentResult] = []
    first_failure_segment_id: str | None = None
    first_failure_reason: str | None = None
    total_time_h = 0.0
    total_energy_kwh = 0.0
    total_energy_cost_eur = 0.0
    helper_distance_km = 0.0
    helper_attach_events = 0
    relay_distance_km = 0.0
    relay_attach_events = 0
    any_infeasible = False
    min_speed_kmh: float | None = None
    critical_segment_id: str | None = None
    helper_was_active = False
    relay_was_active = False

    for seg in route.segments:
        speed_cap_kmh = min(seg.speed_limit_kmh, speed_cap_base_kmh)
        helper_active = helper_policy is not None and not _primary_alone_feasible(
            primary, concept.primary_count, seg, train, mu_continuous, speed_cap_kmh, assumptions
        )

        power_kw, te_kn, adhesive_t, loco_mass_t, used_electric = _combined_capability(
            primary,
            concept.primary_count,
            helper_policy.locomotive if helper_active else None,
            helper_policy.count if helper_active else 0,
            seg,
        )

        # 2026-09-08, diesel-relay mechanism (Option A): zero combined power means a real
        # electrification gap this consist cannot cross under its own power at all -- a same-type
        # helper (if any) achieves nothing here either, so it's forced off rather than billed
        # alongside a relay charge. See ConceptSegmentResult.relay_active's own docstring.
        relay_active = power_kw <= 0
        if relay_active:
            helper_active = False

        if helper_active and not helper_was_active:
            helper_attach_events += 1
        helper_was_active = helper_active
        if relay_active and not relay_was_active:
            relay_attach_events += 1
        relay_was_active = relay_active

        if relay_active:
            relay_distance_km += seg.distance_km
            # No new speed assumption invented -- assume the relay travels at this segment's own
            # effective speed cap, same as every other locomotive would on this segment. Floored to
            # dyn._MIN_SPEED_KMH, the same numerical floor solve_davis_speed_kmh itself applies to a
            # zero/missing speed_limit_kmh (confirmed live: a real, non-electrified 3.8km segment in
            # resolved_routes_pl_ten.parquet has speed_limit_kmh=0.0 -- a genuine RINF data gap, not
            # "must stop here" -- dividing by the raw value crashed with ZeroDivisionError before this
            # fix, a real bug this exposed rather than something the fix works around silently).
            relay_speed_kmh = max(speed_cap_kmh, dyn._MIN_SPEED_KMH)
            time_h = seg.distance_km / relay_speed_kmh
            energy_kwh = 0.0  # the primary provides no traction here, so it draws none
            energy_cost = 0.0
            total_time_h += time_h
            total_energy_kwh += energy_kwh
            total_energy_cost_eur += energy_cost
            if min_speed_kmh is None or relay_speed_kmh < min_speed_kmh:
                min_speed_kmh = relay_speed_kmh
                critical_segment_id = seg.segment_id
            segment_results.append(
                ConceptSegmentResult(
                    segment_id=seg.segment_id,
                    distance_km=seg.distance_km,
                    gradient_pct=seg.gradient_pct,
                    helper_active=False,
                    relay_active=True,
                    feasible=True,
                    failure_reason=None,
                    speed_kmh=relay_speed_kmh,
                    time_h=time_h,
                    energy_kwh=energy_kwh,
                    energy_cost_eur=energy_cost,
                    binding_constraint="diesel_relay",
                )
            )
            continue

        total_mass_t = train.trailing_mass_t + loco_mass_t

        result = dyn.solve_davis_speed_kmh(
            total_mass_t=total_mass_t,
            gradient_pct=seg.gradient_pct,
            power_kw=power_kw,
            max_starting_te_kn=te_kn,
            te_speed_curve=(),
            adhesive_weight_t=adhesive_t,
            # Continuous only -- see solve_davis_speed_kmh's own docstring for why the steady-state
            # solver never gets the starting-adhesion credit.
            adhesion_coefficient=mu_continuous,
            davis=train.davis,
            speed_cap_kmh=speed_cap_kmh,
            gravity_mps2=assumptions.gravity_mps2,
        )

        if result.feasible:
            time_h = seg.distance_km / result.speed_kmh
            if helper_active:
                helper_distance_km += seg.distance_km
            energy_kwh = en.traction_energy_kwh(
                result.required_resistance_kn, seg.distance_km, assumptions.traction_efficiency
            )
            energy_cost = eco.energy_cost_eur(
                energy_kwh, used_electric,
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

        segment_results.append(
            ConceptSegmentResult(
                segment_id=seg.segment_id,
                distance_km=seg.distance_km,
                gradient_pct=seg.gradient_pct,
                helper_active=helper_active,
                relay_active=False,
                feasible=result.feasible,
                failure_reason=result.failure_reason,
                speed_kmh=result.speed_kmh,
                time_h=time_h,
                energy_kwh=energy_kwh,
                energy_cost_eur=energy_cost,
                binding_constraint=result.binding_constraint,
            )
        )

    feasible = not any_infeasible
    commercially_acceptable = feasible and (min_speed_kmh is None or min_speed_kmh >= min_commercial_speed_kmh)

    cost = None
    if feasible:
        distance_km = route.total_distance_km()
        extras_cost_eur = 0.0
        total_time_h_with_dwell = total_time_h
        if helper_policy is not None and helper_attach_events > 0:
            extras_cost_eur += helper_attach_events * helper_policy.attach_detach_cost_eur * 2  # attach + detach
            total_time_h_with_dwell += helper_attach_events * helper_policy.attach_detach_time_h * 2
        if relay_attach_events > 0:
            # Diesel-relay mechanism (Option A): flat all-in EUR/km covers the relay unit's own
            # lease+maintenance+energy for the distance it hauls (see diesel_relay_eur_per_km's own
            # notes in global_assumptions.yaml) -- plus an attach/detach fee and dwell time per
            # transition, same pattern as the helper's own attach/detach cost just above.
            extras_cost_eur += relay_distance_km * assumptions.diesel_relay_eur_per_km
            extras_cost_eur += relay_attach_events * assumptions.diesel_relay_attach_detach_cost_eur * 2
            total_time_h_with_dwell += relay_attach_events * assumptions.diesel_relay_attach_detach_time_h * 2

        # Helper lease/maintenance apply only for the distance/time it is actually attached.
        helper_time_h = (
            sum(sr.time_h or 0.0 for sr in segment_results if sr.helper_active)
            if helper_policy is not None
            else 0.0
        )
        helper_lease_cost = (
            helper_policy.locomotive.lease_eur_per_h * helper_policy.count * helper_time_h
            if helper_policy is not None
            else 0.0
        )
        helper_maintenance_cost = (
            helper_policy.locomotive.maintenance_eur_per_km * helper_policy.count * helper_distance_km
            if helper_policy is not None
            else 0.0
        )
        # A second driver rides the helper for the time it is attached.
        helper_driver_cost = (
            eco.driver_cost_eur(helper_time_h, assumptions.driver_eur_per_h, num_drivers=1)
            if helper_policy is not None
            else 0.0
        )
        extras_cost_eur += helper_lease_cost + helper_maintenance_cost + helper_driver_cost

        cost = eco.compute_cost_breakdown(
            total_energy_cost_eur=total_energy_cost_eur,
            journey_time_h=total_time_h_with_dwell,
            distance_km=distance_km,
            trailing_mass_t=train.trailing_mass_t,
            lease_eur_per_h=primary.lease_eur_per_h,
            maintenance_eur_per_km=primary.maintenance_eur_per_km,
            locomotive_count=concept.primary_count,
            driver_eur_per_h=assumptions.driver_eur_per_h,
            num_drivers=1,
            operational_extras_cost_eur=extras_cost_eur,
        )

    return OperatingConceptResult(
        concept_name=concept.name,
        feasible=feasible,
        commercially_acceptable=commercially_acceptable,
        first_failure_segment_id=first_failure_segment_id,
        failure_reason=first_failure_reason,
        distance_km=route.total_distance_km(),
        journey_time_h=total_time_h if feasible else None,
        energy_kwh=total_energy_kwh if feasible else None,
        helper_distance_km=helper_distance_km,
        helper_attach_events=helper_attach_events,
        relay_distance_km=relay_distance_km,
        relay_attach_events=relay_attach_events,
        min_speed_kmh=min_speed_kmh if feasible else None,
        critical_segment_id=critical_segment_id if feasible else None,
        cost=cost,
        segment_results=segment_results,
    )


@dataclass
class DynamicConceptSegmentResult:
    segment_id: str
    distance_km: float
    gradient_pct: float
    helper_active: bool
    # 2026-09-08, diesel-relay mechanism -- see ConceptSegmentResult.relay_active's own docstring.
    relay_active: bool
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
    # 2026-09-08: mirrors ConceptSegmentResult.energy_cost_eur (the static engine already exposed
    # this; the dynamic engine computed it locally per segment but never stored it) -- added for a
    # user-requested cumulative-cost-by-distance analysis across real corridors, and generally useful
    # any time a per-segment monetary breakdown is needed from the dynamic engine, not just this once.
    energy_cost_eur: float | None


def run_operating_concept_dynamic(
    *,
    concept: OperatingConcept,
    route: Route,
    train: Train,
    assumptions: GlobalAssumptions,
    min_commercial_speed_kmh: float | None = None,
    dt_s: float = 1.0,
    max_time_h: float = 24.0,
) -> OperatingConceptResult:
    """V0.4-integrated counterpart to run_operating_concept (2026-08-26, CLAUDE.md section 15):
    real F_net/effective_mass acceleration/deceleration, a braking-distance lookahead, and explicit
    coasting/regenerative/mechanical-braking mode selection -- the same physics
    `simulation.run_scenario_dynamic` already gives a single locomotive, now available for
    multi-unit/helper operating concepts too.

    Motivating real result, not a hypothetical: the OD-pair map's Zurich-Basel corridor showed every
    modelled locomotive as infeasible under the steady-state solver above, traced to one real short
    2.73% pinch near Zürich HB -- a textbook case of what CLAUDE.md's own Betuweroute validation
    already found (data/validation/betuweroute_corridor.md): a train carrying real speed into a brief
    steep section can cross it even though it could never have started climbing it from rest, and the
    V0.3 steady-state solver has no notion of "already moving." This function gives operating
    concepts (helper/multi-unit included) the same momentum credit `run_scenario_dynamic` already
    gives a single locomotive.

    Deliberately NOT the new default: this is an additional, opt-in engine alongside
    run_operating_concept, the same "two selectable models" pattern as v02_legacy/davis and
    run_scenario/run_scenario_dynamic -- every headline number this project has reported so far
    (the pooled addressable-share comparisons, the locomotive-demand translation) was computed with
    the steady-state engine; switching that would move real, already-reported numbers, not just fix
    Zurich-Basel, and hasn't been asked for.

    Per-step combined capability (power/TE/adhesive weight, and whether the segment is actually being
    drawn electrically once voltage compatibility is honoured) is recomputed via the same
    `_combined_capability`/`_usable_power_kw` helpers `run_operating_concept` uses, since a helper can
    attach or detach mid-route and a locomotive's usable power can differ hop to hop -- unlike
    `simulation.run_scenario_dynamic`, which only ever has one fixed consist for the whole route.
    Regenerative-braking eligibility is keyed to `used_electric` (whether this concept is actually
    drawing catenary power on this segment), not the segment's own raw `electrified` flag -- a
    locomotive running in diesel fallback on an electrified segment (voltage mismatch) cannot feed
    power back into a wire its pantograph isn't even compatible with.

    Does not (yet) integrate V0.5's axle-load/train-length/infrastructure-power-ceiling checks, same
    documented scope boundary as run_operating_concept above -- only the dynamics changed here.
    """
    mu_starting, mu_continuous = trac.resolve_adhesion_coefficients(
        override=concept.adhesion_coefficient,
        loco_starting=concept.primary_locomotive.starting_adhesion_coefficient,
        loco_continuous=concept.primary_locomotive.continuous_adhesion_coefficient,
        default=assumptions.default_adhesion_coefficient,
    )
    min_commercial_speed_kmh = (
        min_commercial_speed_kmh
        if min_commercial_speed_kmh is not None
        else assumptions.default_min_commercial_speed_kmh
    )

    primary = concept.primary_locomotive
    helper_policy = concept.helper_policy
    speed_cap_base_kmh = primary.max_speed_kmh
    if helper_policy is not None:
        speed_cap_base_kmh = min(speed_cap_base_kmh, helper_policy.locomotive.max_speed_kmh)

    segments = tuple(route.segments)
    if not segments:
        raise ValueError("Route has no segments")
    profile = dyn.compute_braking_profile(segments, speed_cap_base_kmh, assumptions.service_braking_rate_mps2)

    segment_results: list[DynamicConceptSegmentResult] = []
    first_failure_segment_id: str | None = None
    first_failure_reason: str | None = None
    speed_kmh = 0.0
    total_time_h = 0.0
    total_traction_kwh = 0.0
    total_regen_kwh = 0.0
    total_energy_cost_eur = 0.0
    helper_distance_km = 0.0
    helper_attach_events = 0
    relay_distance_km = 0.0
    relay_attach_events = 0
    any_infeasible = False
    min_speed_kmh_overall: float | None = None
    critical_segment_id: str | None = None
    helper_was_active = False
    relay_was_active = False

    max_steps = int((max_time_h * 3600.0) / dt_s)
    steps_taken = 0

    for seg_index, seg in enumerate(segments):
        seg_speed_cap_kmh = min(seg.speed_limit_kmh, speed_cap_base_kmh)
        helper_active = helper_policy is not None and not _primary_alone_feasible(
            primary, concept.primary_count, seg, train, mu_continuous, seg_speed_cap_kmh, assumptions
        )

        power_kw, te_kn, adhesive_t, loco_mass_t, used_electric = _combined_capability(
            primary,
            concept.primary_count,
            helper_policy.locomotive if helper_active else None,
            helper_policy.count if helper_active else 0,
            seg,
        )

        # 2026-09-08, diesel-relay mechanism (Option A): see run_operating_concept's identical branch
        # for the full rationale -- zero combined power means a real electrification gap this consist
        # cannot cross under its own power at all, so a same-type helper is forced off (it would
        # achieve nothing here) and the segment is relay-hauled instead of run through the physics
        # step loop at all.
        relay_active = power_kw <= 0
        if relay_active:
            helper_active = False

        if helper_active and not helper_was_active:
            helper_attach_events += 1
        helper_was_active = helper_active
        if relay_active and not relay_was_active:
            relay_attach_events += 1
        relay_was_active = relay_active

        if relay_active:
            relay_distance_km += seg.distance_km
            # No new speed assumption invented -- assume the relay travels at this segment's own
            # effective speed cap, same as every other locomotive would on this segment. Floored to
            # dyn._MIN_SPEED_KMH, the same numerical floor solve_davis_speed_kmh/simulate_step apply
            # to a zero/missing speed_limit_kmh (confirmed live: a real, non-electrified 3.8km segment
            # in resolved_routes_pl_ten.parquet has speed_limit_kmh=0.0 -- a genuine RINF data gap,
            # not "must stop here" -- dividing by the raw value crashed with ZeroDivisionError before
            # this fix, in the sibling static-engine branch above; applied here too for consistency,
            # not because this exact crash was reproduced in the dynamic engine).
            relay_speed_kmh = max(seg_speed_cap_kmh, dyn._MIN_SPEED_KMH)
            # Carried forward as both entry and exit speed so the next segment's momentum continuity
            # isn't disrupted by an artificial discontinuity.
            entry_speed_kmh = speed_kmh
            seg_time_h = seg.distance_km / relay_speed_kmh
            speed_kmh = relay_speed_kmh
            total_time_h += seg_time_h
            # The primary provides no traction here, so it draws/regenerates none.
            segment_results.append(
                DynamicConceptSegmentResult(
                    segment_id=seg.segment_id, distance_km=seg.distance_km, gradient_pct=seg.gradient_pct,
                    helper_active=False, relay_active=True, feasible=True, failure_reason=None,
                    entry_speed_kmh=entry_speed_kmh, exit_speed_kmh=speed_kmh,
                    min_speed_kmh=relay_speed_kmh, max_speed_kmh=relay_speed_kmh,
                    time_h=seg_time_h, traction_energy_kwh=0.0, regen_energy_kwh=0.0, net_energy_kwh=0.0,
                    energy_cost_eur=0.0,
                )
            )
            if min_speed_kmh_overall is None or relay_speed_kmh < min_speed_kmh_overall:
                min_speed_kmh_overall = relay_speed_kmh
                critical_segment_id = seg.segment_id
            continue

        total_mass_t = train.trailing_mass_t + loco_mass_t

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

        while not seg_infeasible and offset_km < seg.distance_km - 1e-9:
            if steps_taken >= max_steps:
                seg_infeasible = True
                seg_failure_reason = "Simulation exceeded max_time_h without completing the route"
                break

            # Same two-point lookahead as simulation.run_scenario_dynamic: evaluating the ceiling one
            # step ahead too (using current speed as the distance estimate) keeps a tightening
            # upcoming limit from lagging a full step behind and being overshot.
            lookahead_km = (speed_kmh / 3.6) * dt_s / 1000.0
            ceiling_kmh = min(
                dyn.ceiling_speed_kmh(
                    profile, segments, seg_index, offset_km, speed_cap_base_kmh, assumptions.service_braking_rate_mps2
                ),
                dyn.ceiling_speed_kmh(
                    profile,
                    segments,
                    seg_index,
                    min(offset_km + lookahead_km, seg.distance_km),
                    speed_cap_base_kmh,
                    assumptions.service_braking_rate_mps2,
                ),
            )
            step = dyn.simulate_step(
                speed_kmh=speed_kmh,
                ceiling_kmh=ceiling_kmh,
                total_mass_t=total_mass_t,
                gradient_pct=seg.gradient_pct,
                electrified=used_electric,
                power_kw=power_kw,
                max_starting_te_kn=te_kn,
                te_speed_curve=(),
                adhesive_weight_t=adhesive_t,
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
                        max_starting_te_kn=te_kn,
                        te_speed_curve=(),
                        adhesive_weight_t=adhesive_t,
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
                DynamicConceptSegmentResult(
                    segment_id=seg.segment_id, distance_km=seg.distance_km, gradient_pct=seg.gradient_pct,
                    helper_active=helper_active, relay_active=False, feasible=False, failure_reason=seg_failure_reason,
                    entry_speed_kmh=entry_speed_kmh, exit_speed_kmh=None, min_speed_kmh=None, max_speed_kmh=None,
                    time_h=None, traction_energy_kwh=None, regen_energy_kwh=None, net_energy_kwh=None,
                    energy_cost_eur=None,
                )
            )
            break

        if helper_active:
            helper_distance_km += seg.distance_km
        energy_cost_this_seg = eco.energy_cost_eur(
            seg_net_kwh, used_electric,
            eco.resolve_electricity_price_eur_per_kwh(
                seg.country, assumptions.electricity_price_by_country_eur_per_kwh,
                assumptions.electricity_price_eur_per_kwh,
            ),
            assumptions.diesel_price_eur_per_kwh,
        )
        total_time_h += seg_time_s / 3600.0
        total_traction_kwh += seg_traction_kwh
        total_regen_kwh += seg_regen_kwh
        total_energy_cost_eur += energy_cost_this_seg

        if min_speed_kmh_overall is None or seg_min_speed < min_speed_kmh_overall:
            min_speed_kmh_overall = seg_min_speed
            critical_segment_id = seg.segment_id

        segment_results.append(
            DynamicConceptSegmentResult(
                segment_id=seg.segment_id, distance_km=seg.distance_km, gradient_pct=seg.gradient_pct,
                helper_active=helper_active, relay_active=False, feasible=True, failure_reason=None,
                entry_speed_kmh=entry_speed_kmh, exit_speed_kmh=speed_kmh, min_speed_kmh=seg_min_speed,
                max_speed_kmh=seg_max_speed, time_h=seg_time_s / 3600.0, traction_energy_kwh=seg_traction_kwh,
                regen_energy_kwh=seg_regen_kwh, net_energy_kwh=seg_net_kwh,
                energy_cost_eur=energy_cost_this_seg,
            )
        )

    feasible = not any_infeasible
    commercially_acceptable = feasible and (
        min_speed_kmh_overall is None or min_speed_kmh_overall >= min_commercial_speed_kmh
    )

    cost = None
    if feasible:
        distance_km = route.total_distance_km()
        extras_cost_eur = 0.0
        total_time_h_with_dwell = total_time_h
        if helper_policy is not None and helper_attach_events > 0:
            extras_cost_eur += helper_attach_events * helper_policy.attach_detach_cost_eur * 2  # attach + detach
            total_time_h_with_dwell += helper_attach_events * helper_policy.attach_detach_time_h * 2
        if relay_attach_events > 0:
            # Diesel-relay mechanism (Option A) -- see run_operating_concept's identical block for
            # the full rationale.
            extras_cost_eur += relay_distance_km * assumptions.diesel_relay_eur_per_km
            extras_cost_eur += relay_attach_events * assumptions.diesel_relay_attach_detach_cost_eur * 2
            total_time_h_with_dwell += relay_attach_events * assumptions.diesel_relay_attach_detach_time_h * 2

        helper_time_h = (
            sum(sr.time_h or 0.0 for sr in segment_results if sr.helper_active)
            if helper_policy is not None
            else 0.0
        )
        helper_lease_cost = (
            helper_policy.locomotive.lease_eur_per_h * helper_policy.count * helper_time_h
            if helper_policy is not None
            else 0.0
        )
        helper_maintenance_cost = (
            helper_policy.locomotive.maintenance_eur_per_km * helper_policy.count * helper_distance_km
            if helper_policy is not None
            else 0.0
        )
        helper_driver_cost = (
            eco.driver_cost_eur(helper_time_h, assumptions.driver_eur_per_h, num_drivers=1)
            if helper_policy is not None
            else 0.0
        )
        extras_cost_eur += helper_lease_cost + helper_maintenance_cost + helper_driver_cost

        cost = eco.compute_cost_breakdown(
            total_energy_cost_eur=total_energy_cost_eur,
            journey_time_h=total_time_h_with_dwell,
            distance_km=distance_km,
            trailing_mass_t=train.trailing_mass_t,
            lease_eur_per_h=primary.lease_eur_per_h,
            maintenance_eur_per_km=primary.maintenance_eur_per_km,
            locomotive_count=concept.primary_count,
            driver_eur_per_h=assumptions.driver_eur_per_h,
            num_drivers=1,
            operational_extras_cost_eur=extras_cost_eur,
        )

    return OperatingConceptResult(
        concept_name=concept.name,
        feasible=feasible,
        commercially_acceptable=commercially_acceptable,
        first_failure_segment_id=first_failure_segment_id,
        failure_reason=first_failure_reason,
        distance_km=route.total_distance_km(),
        journey_time_h=total_time_h if feasible else None,
        energy_kwh=(total_traction_kwh - total_regen_kwh) if feasible else None,
        helper_distance_km=helper_distance_km,
        helper_attach_events=helper_attach_events,
        relay_distance_km=relay_distance_km,
        relay_attach_events=relay_attach_events,
        min_speed_kmh=min_speed_kmh_overall if feasible else None,
        critical_segment_id=critical_segment_id if feasible else None,
        cost=cost,
        segment_results=segment_results,
    )


def compare_operating_concepts(
    concepts: list[OperatingConcept], route: Route, train: Train, assumptions: GlobalAssumptions
) -> list[OperatingConceptResult]:
    """Runs every concept and returns results sorted feasible-and-cheapest first (CLAUDE.md section
    21: "cost-optimal operating concept")."""
    results = [
        run_operating_concept(concept=c, route=route, train=train, assumptions=assumptions) for c in concepts
    ]
    return sorted(
        results,
        key=lambda r: (not r.feasible, r.cost.total_cost_eur if r.cost else float("inf")),
    )


@dataclass
class MassSweepRow:
    """One row of a CLAUDE.md section 21 scenario sweep: for a given trailing mass, the cost-optimal
    operating concept and how the named "subject" concept (typically EURO9000) compares to the best
    of everything else."""

    trailing_mass_t: float
    best_concept_name: str | None
    best_feasible: bool
    best_min_speed_kmh: float | None
    best_journey_time_h: float | None
    best_total_cost_eur: float | None
    best_eur_per_tonne_km: float | None
    subject_feasible: bool
    subject_cost_eur: float | None
    best_alternative_name: str | None  # cheapest feasible concept excluding the subject
    best_alternative_cost_eur: float | None
    subject_saving_eur: float | None  # best_alternative_cost - subject_cost; positive = subject cheaper
    subject_saving_pct: float | None


def run_mass_sweep(
    *,
    concepts: list[OperatingConcept],
    subject_concept_name: str,
    route: Route,
    base_train: Train,
    masses_t: list[float],
    assumptions: GlobalAssumptions,
) -> list[MassSweepRow]:
    """CLAUDE.md section 21: sweeps train mass across ``masses_t``, running every concept in
    ``concepts`` at each mass and reporting the cost-optimal concept plus how ``subject_concept_name``
    (typically the EURO9000 concept) compares to the best of the rest."""
    rows: list[MassSweepRow] = []
    for mass_t in masses_t:
        train_at_mass = Train(
            name=base_train.name,
            status=base_train.status,
            trailing_mass_t=mass_t,
            davis=base_train.davis,
            max_permitted_speed_kmh=base_train.max_permitted_speed_kmh,
            train_length_m=base_train.train_length_m,
        )
        results = compare_operating_concepts(concepts, route, train_at_mass, assumptions)
        best = results[0] if results else None
        subject = next((r for r in results if r.concept_name == subject_concept_name), None)
        alternatives = [r for r in results if r.concept_name != subject_concept_name and r.feasible]
        best_alternative = alternatives[0] if alternatives else None

        subject_saving_eur = None
        subject_saving_pct = None
        if subject is not None and subject.feasible and best_alternative is not None:
            subject_saving_eur = best_alternative.cost.total_cost_eur - subject.cost.total_cost_eur
            subject_saving_pct = subject_saving_eur / best_alternative.cost.total_cost_eur * 100.0

        rows.append(
            MassSweepRow(
                trailing_mass_t=mass_t,
                best_concept_name=best.concept_name if best and best.feasible else None,
                best_feasible=bool(best and best.feasible),
                best_min_speed_kmh=best.min_speed_kmh if best and best.feasible else None,
                best_journey_time_h=best.journey_time_h if best and best.feasible else None,
                best_total_cost_eur=best.cost.total_cost_eur if best and best.feasible else None,
                best_eur_per_tonne_km=best.cost.eur_per_trailing_tonne_km if best and best.feasible else None,
                subject_feasible=bool(subject and subject.feasible),
                subject_cost_eur=subject.cost.total_cost_eur if subject and subject.feasible else None,
                best_alternative_name=best_alternative.concept_name if best_alternative else None,
                best_alternative_cost_eur=best_alternative.cost.total_cost_eur if best_alternative else None,
                subject_saving_eur=subject_saving_eur,
                subject_saving_pct=subject_saving_pct,
            )
        )
    return rows
