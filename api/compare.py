"""Vercel Python serverless function: runs the real elp_model physics/economics engine against a
user-supplied custom locomotive spec, on a route the client already has (its segments, fetched from
one of the existing od_routes/*.json files). Reuses the actual deterministic model -- see
api/_vendor/elp_model/ -- not a reimplementation, per this project's own "one engine, no drift"
discipline.

POST body:
{
  "segments": [{"segment_id", "distance_km", "gradient_pct", "electrified", "speed_limit_kmh",
                "electrification_system", "country"}, ...],   // from the client's already-loaded route
  "locomotive": {
    "name": str,
    "mass_t": float, "axles": int,
    "electric_max_power_kw": float, "diesel_max_power_kw": float,
    "max_speed_kmh": float, "max_starting_te_kn": float,
    "capex_eur": float,
    "voltage_systems": [str, ...]   // optional, omit/empty = unrestricted
  }
}

Response: {"feasible": bool, "concept": str, "total_cost_eur": float|null, "journey_time_h": float|null,
           "helper_distance_km", "helper_attach_events", "relay_distance_km", "relay_attach_events",
           "failure_reason": str|null, "seg_states": str}  -- seg_states uses the SAME compact
encoding as od_routes/*.json ('.'=OK, 'H'=helper, 'R'=relay, 'X'=infeasible) so the client can reuse
its existing rendering code unchanged.
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "_vendor"))

from elp_model.infrastructure import Route, Segment  # noqa: E402
from elp_model.locomotives import Locomotive  # noqa: E402
from elp_model.operating_concepts import (  # noqa: E402
    HelperPolicy,
    OperatingConcept,
    run_operating_concept_dynamic,
)
from elp_model.simulation import load_global_assumptions  # noqa: E402
from elp_model.trains import load_train  # noqa: E402

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "_vendor", "config")
ASSUMPTIONS = load_global_assumptions(os.path.join(CONFIG_DIR, "global_assumptions.yaml"))
TRAIN = load_train(os.path.join(CONFIG_DIR, "generic_freight_1600t.yaml"))
MAX_TIME_H = 72.0
MAX_SEGMENTS = 2000  # sanity bound -- real routes top out around 1050


class ValidationError(ValueError):
    pass


def _require_positive_number(d: dict, key: str) -> float:
    if key not in d or d[key] is None:
        raise ValidationError(f"'{key}' is required")
    try:
        v = float(d[key])
    except (TypeError, ValueError):
        raise ValidationError(f"'{key}' must be a number")
    if v <= 0:
        raise ValidationError(f"'{key}' must be greater than 0")
    return v


def _build_locomotive(spec: dict) -> Locomotive:
    mass_t = _require_positive_number(spec, "mass_t")
    axles = spec.get("axles")
    if axles is None or int(axles) <= 0:
        raise ValidationError("'axles' is required and must be a positive integer")
    axles = int(axles)

    electric_kw = float(spec.get("electric_max_power_kw") or 0)
    diesel_kw = float(spec.get("diesel_max_power_kw") or 0)
    if electric_kw <= 0 and diesel_kw <= 0:
        raise ValidationError("at least one of 'electric_max_power_kw' / 'diesel_max_power_kw' must be > 0")

    max_speed_kmh = _require_positive_number(spec, "max_speed_kmh")
    max_starting_te_kn = _require_positive_number(spec, "max_starting_te_kn")
    capex_eur = _require_positive_number(spec, "capex_eur")

    voltage_systems = tuple(spec.get("voltage_systems") or ())
    valid_systems = {"25kV_AC_50Hz", "15kV_AC_16.7Hz", "3kV_DC", "1.5kV_DC"}
    bad = [v for v in voltage_systems if v not in valid_systems]
    if bad:
        raise ValidationError(f"unknown voltage system(s): {bad} -- must be from {sorted(valid_systems)}")

    # Same 12.5%-of-capex / annual_km_per_locomotive convention every locomotive in this project
    # uses (config/economics/global_assumptions.yaml) -- applied identically here so a custom
    # locomotive is compared on the same basis, not given an invented cost advantage or penalty.
    maintenance_eur_per_km = capex_eur * 0.125 / ASSUMPTIONS.annual_km_per_locomotive

    return Locomotive(
        name=str(spec.get("name") or "Custom locomotive"),
        manufacturer="User-supplied",
        status="USER_SUPPLIED",
        mass_t=mass_t,
        axles=axles,
        adhesive_weight_t=mass_t,  # assume all axles powered -- same convention as every locomotive
        max_speed_kmh=max_speed_kmh,
        electric_max_power_kw=electric_kw,
        diesel_max_power_kw=diesel_kw,
        voltage_systems=voltage_systems,
        max_starting_te_kn=max_starting_te_kn,
        te_speed_curve=(),
        lease_eur_per_h=0.0,
        maintenance_eur_per_km=maintenance_eur_per_km,
        starting_adhesion_coefficient=None,
        continuous_adhesion_coefficient=None,
    )


def _build_route(raw_segments: list) -> Route:
    if not raw_segments:
        raise ValidationError("'segments' must be a non-empty list")
    if len(raw_segments) > MAX_SEGMENTS:
        raise ValidationError(f"route has {len(raw_segments)} segments, exceeding the {MAX_SEGMENTS} limit")
    segments = []
    for i, s in enumerate(raw_segments):
        try:
            segments.append(Segment(
                segment_id=str(s.get("segment_id", f"seg-{i}")),
                distance_km=float(s["distance_km"]),
                gradient_pct=float(s["gradient_pct"]),
                electrified=bool(s["electrified"]),
                speed_limit_kmh=float(s.get("speed_limit_kmh") or 100.0),
                electrification_system=s.get("electrification_system"),
                country=s.get("country"),
            ))
        except (KeyError, TypeError, ValueError) as e:
            raise ValidationError(f"segment {i} is malformed: {e}")
    return Route(route_id="custom-comparison", segments=segments)


def _run(locomotive: Locomotive, route: Route) -> dict:
    candidates = [
        ("1x", OperatingConcept(f"1x {locomotive.name}", locomotive, 1)),
        ("1x+helper", OperatingConcept(
            f"1x {locomotive.name} + helper", locomotive, 1,
            helper_policy=HelperPolicy(locomotive, count=1),
        )),
    ]
    best = None
    best_label = None
    first_failure = None
    for label, concept in candidates:
        result = run_operating_concept_dynamic(
            concept=concept, route=route, train=TRAIN, assumptions=ASSUMPTIONS, max_time_h=MAX_TIME_H,
        )
        if result.feasible and (best is None or result.cost.total_cost_eur < best.cost.total_cost_eur):
            best, best_label = result, label
        elif not result.feasible and first_failure is None:
            first_failure = result

    if best is None:
        r = first_failure
        seg_states = "".join(
            "X" if not sr.feasible else ("R" if sr.relay_active else ("H" if sr.helper_active else "."))
            for sr in r.segment_results
        ) if r else ""
        return {
            "feasible": False, "concept": None, "total_cost_eur": None, "journey_time_h": None,
            "helper_distance_km": 0.0, "helper_attach_events": 0,
            "relay_distance_km": 0.0, "relay_attach_events": 0,
            "failure_reason": r.failure_reason if r else "infeasible",
            "seg_states": seg_states,
        }

    seg_states = "".join(
        "R" if sr.relay_active else ("H" if sr.helper_active else ".")
        for sr in best.segment_results
    )
    return {
        "feasible": True, "concept": best_label,
        "total_cost_eur": best.cost.total_cost_eur, "journey_time_h": best.journey_time_h,
        "helper_distance_km": best.helper_distance_km, "helper_attach_events": best.helper_attach_events,
        "relay_distance_km": best.relay_distance_km, "relay_attach_events": best.relay_attach_events,
        "failure_reason": None, "seg_states": seg_states,
    }


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "malformed JSON body"})
            return

        try:
            locomotive = _build_locomotive(body.get("locomotive") or {})
            route = _build_route(body.get("segments") or [])
            result = _run(locomotive, route)
            self._send_json(200, result)
        except ValidationError as e:
            self._send_json(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001 -- deliberately broad: never leak a bare 500 with no message
            self._send_json(500, {"error": f"internal error: {e}"})
