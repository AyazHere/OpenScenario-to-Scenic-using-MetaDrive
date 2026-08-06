#!/usr/bin/env python3
"""
xosc2scenicrules.py
===================
Converter: OpenSCENARIO (.xosc, SCTrans dataset) -> Scenic 3 scenario in
the ScenicRules file layout, runnable on MetaDrive.

This version converts PLACEMENT + DRIVING BEHAVIORS, combining the good
properties of both earlier converters:
  - exact recorded spawn positions and headings (never moved or rotated)
  - ego destination (from AcquirePositionAction) and goal-directed ego
  - NPCs drive their RECORDED routes: waypoints -> lane sequence via
    heading-aware matching -> FollowTrajectoryBehavior; the recorded
    heading picks the right lane among overlapping ones, and routes are
    stitched across junctions with broken connectivity (common in these
    LGSVL maps). Lane matching steers the car - it never relocates it.
  - recorded speeds (initial speed exact; cruise speed = mean of the
    recorded moving profile), parked detection
  - pedestrians walk toward the end of their recorded route

Deliberately NOT included yet (later steps, on request):
  action triggers, rule recording (bench monitor), falsification
  parameters (VerifaiRange), map repair. SUMO maps: xodr2sumo.py.

Heading conversion (fixes the 90-degree bug in convNew.py):
  OpenSCENARIO h : radians, 0 = east (+x), counter-clockwise
  Scenic heading : 0 = north (+y), counter-clockwise
  => scenic_deg = degrees(h) - 90

Usage:
    python xosc2scenicrules.py input.xosc                # -> input.scenic
    python xosc2scenicrules.py input.xosc output.scenic
    python xosc2scenicrules.py folder/ -d out_dir/       # batch convert
    python xosc2scenicrules.py folder/ -d out_dir/ --limit 20

Run a converted scenario on MetaDrive (needs <map>.net.xml next to the
.xodr):
    scenic --2d --model scenic.simulators.metadrive.model --simulate file.scenic
"""

import argparse
import math
import os
import statistics
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────────────────────

PARKED_SPEED_THRESHOLD = 0.5   # m/s — below this an entity is "parked"
MAX_ROUTE_WAYPOINTS    = 60    # per-entity cap after dedup/downsampling
WAYPOINT_MIN_SPACING   = 0.5   # m — dedup tolerance
DEFAULT_DURATION       = 30.0  # s — when no profile/StopTrigger info
DURATION_BUFFER        = 5.0   # s — extra time past longest speed profile

# behavior-layer tunables (converter-added driving parameters; the
# recorded data has no equivalents for these)
MIN_TARGET_SPEED       = 1.0   # m/s — floor for cruise speeds
DEFAULT_EGO_SPEED      = 8.0   # m/s — when the recorded ego speed is ~0
DEFAULT_PED_SPEED      = 1.4   # m/s — when a pedestrian profile is empty
EGO_SAFETY_DIST        = 8.0   # m — ego brakes when anything is closer
EGO_BRAKE              = 0.8   # brake intensity for the safety interrupt
GOAL_RADIUS            = 5.0   # m — ego counts as arrived at its goal
MIN_GOAL_TRIP          = 20.0  # m — below this, skip goal-based terminate


# ─────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────

@dataclass
class Position:
    """WorldPosition from the .xosc (x, y in metres, h in radians east-CCW)."""
    x: float = 0.0
    y: float = 0.0
    heading_rad: float = 0.0

    @property
    def scenic_heading_rad(self) -> float:
        """Scenic heading (radians, 0 = north/+y, counter-clockwise)."""
        rad = self.heading_rad - math.pi / 2.0
        while rad <= -math.pi:
            rad += 2.0 * math.pi
        while rad > math.pi:
            rad -= 2.0 * math.pi
        return rad

    @property
    def scenic_heading_deg(self) -> float:
        """Scenic heading (degrees, 0 = north/+y, counter-clockwise)."""
        return math.degrees(self.scenic_heading_rad)


@dataclass
class SpeedEntry:
    speed: float      # m/s
    duration: float   # s


@dataclass
class Entity:
    name: str
    is_ego: bool = False
    category: str = "car"            # xosc vehicleCategory / "pedestrian"
    blueprint: str = ""
    length: float = 0.0              # from BoundingBox (0 = unknown)
    width: float = 0.0
    spawn: Optional[Position] = None      # TeleportAction
    goal: Optional[Position] = None       # AcquirePositionAction (ego dest)
    init_speed: float = 0.0
    controller_module: str = ""
    waypoints: List[Position] = field(default_factory=list)
    speed_profile: List[SpeedEntry] = field(default_factory=list)

    @property
    def is_parked(self) -> bool:
        """True when the recorded data shows the entity never really moves."""
        peak = max([self.init_speed] + [e.speed for e in self.speed_profile],
                   default=self.init_speed)
        return peak < PARKED_SPEED_THRESHOLD

    @property
    def target_speed(self) -> float:
        """Cruise speed for behaviors: mean of the recorded moving speeds."""
        moving = [e.speed for e in self.speed_profile
                  if e.speed > PARKED_SPEED_THRESHOLD]
        if moving:
            return max(MIN_TARGET_SPEED, statistics.mean(moving))
        if self.init_speed > PARKED_SPEED_THRESHOLD:
            return max(MIN_TARGET_SPEED, self.init_speed)
        return DEFAULT_PED_SPEED if self.is_pedestrian else DEFAULT_EGO_SPEED

    @property
    def profile_duration(self) -> float:
        return sum(e.duration for e in self.speed_profile)

    @property
    def is_pedestrian(self) -> bool:
        return self.category == "pedestrian"


@dataclass
class Weather:
    precipitation: str = "none"
    rain_intensity: float = 0.0
    fog_range: float = 100000.0
    sun_elevation_rad: float = 1.31


@dataclass
class Scenario:
    source_file: str = ""
    description: str = ""
    map_file: str = ""
    map_name: str = ""
    entities: List[Entity] = field(default_factory=list)
    weather: Weather = field(default_factory=Weather)
    tod_hour: int = 12
    duration_sec: float = DEFAULT_DURATION

    @property
    def ego(self) -> Optional[Entity]:
        return next((e for e in self.entities if e.is_ego), None)

    @property
    def npcs(self) -> List[Entity]:
        return [e for e in self.entities if not e.is_ego]


# ─────────────────────────────────────────────────────────────────
# XML helpers (tag-name based, namespace-safe)
# ─────────────────────────────────────────────────────────────────

def tag(element) -> str:
    t = element.tag
    return t.split("}")[-1] if "}" in t else t


def find_all(root, tag_name: str) -> list:
    return [el for el in root.iter() if tag(el) == tag_name]


def find_first(root, tag_name: str):
    results = find_all(root, tag_name)
    return results[0] if results else None


def attr_float(el, name: str, default: float = 0.0) -> float:
    try:
        return float(el.get(name, default))
    except (TypeError, ValueError):
        return default


def attr_str(el, name: str, default: str = "") -> str:
    return el.get(name, default)


def safe_name(name: str) -> str:
    """Convert any entity name to a valid Python identifier."""
    result = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    if result and result[0].isdigit():
        result = "e_" + result
    return result or "entity"


# ─────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────

class XOSCParser:
    """Reads a .xosc file into a Scenario object."""

    # MetaDrive (stock Scenic interface) implements only Car and Pedestrian.
    SCENIC_CLASS = {
        "car":        "Car",
        "van":        "Car",
        "truck":      "Car",
        "bus":        "Car",
        "bicycle":    "Car",
        "motorbike":  "Car",
        "trailer":    "Car",
        "pedestrian": "Pedestrian",
    }

    def __init__(self, path: str):
        self.tree = ET.parse(path)
        self.root = self.tree.getroot()
        self.scenario = Scenario(source_file=path)
        self._entities: dict = {}

    def parse(self) -> Scenario:
        self._parse_header()
        self._parse_map()
        self._parse_entities()
        self._parse_init()
        self._parse_stories()
        self._parse_environment()
        self._parse_end_condition()
        self.scenario.entities = list(self._entities.values())
        return self.scenario

    def _parse_header(self):
        h = find_first(self.root, "FileHeader")
        if h is not None:
            desc = attr_str(h, "description")
            author = attr_str(h, "author")
            self.scenario.description = (
                f"{desc} (author: {author})" if author else desc)

    def _parse_map(self):
        lf = find_first(self.root, "LogicFile")
        if lf is not None:
            path = attr_str(lf, "filepath")
            self.scenario.map_file = path
            self.scenario.map_name = os.path.splitext(os.path.basename(path))[0]

    def _parse_entities(self):
        ent_root = find_first(self.root, "Entities")
        if ent_root is None:
            return

        for so in find_all(ent_root, "ScenarioObject"):
            name = attr_str(so, "name", "entity")
            e = Entity(name=name)

            veh = find_first(so, "Vehicle")
            ped = find_first(so, "Pedestrian")
            if veh is not None:
                e.category = attr_str(veh, "vehicleCategory", "car").lower()
                e.blueprint = attr_str(veh, "name")
                bbox_src = veh
            elif ped is not None:
                e.category = "pedestrian"
                e.blueprint = attr_str(ped, "model") or attr_str(ped, "name")
                bbox_src = ped
            else:
                bbox_src = so

            dims = find_first(bbox_src, "Dimensions")
            if dims is not None:
                e.length = attr_float(dims, "length")
                e.width = attr_float(dims, "width")

            for prop in find_all(so, "Property"):
                if (attr_str(prop, "name") == "type"
                        and attr_str(prop, "value") == "ego_vehicle"):
                    e.is_ego = True

            self._entities[name] = e

        if not any(e.is_ego for e in self._entities.values()):
            for e in self._entities.values():
                if "ego" in e.name.lower() or "hero" in e.name.lower():
                    e.is_ego = True
                    break

    def _parse_init(self):
        init = find_first(self.root, "Init")
        if init is None:
            return

        for private in find_all(init, "Private"):
            name = attr_str(private, "entityRef")
            e = self._entities.get(name)
            if e is None:
                continue

            # spawn = TeleportAction; goal = AcquirePositionAction.
            # (convNew.py collapsed both into "spawn", losing the ego goal.)
            tele = find_first(private, "TeleportAction")
            if tele is not None:
                e.spawn = self._read_world_position(tele)
            acq = find_first(private, "AcquirePositionAction")
            if acq is not None:
                e.goal = self._read_world_position(acq)
            if e.spawn is None and e.goal is not None:
                e.spawn, e.goal = e.goal, None

            speed_node = find_first(private, "AbsoluteTargetSpeed")
            if speed_node is not None:
                e.init_speed = attr_float(speed_node, "value")

            ctrl = find_first(private, "Controller")
            if ctrl is not None:
                for prop in find_all(ctrl, "Property"):
                    if attr_str(prop, "name") == "module":
                        e.controller_module = attr_str(prop, "value")

            sp = find_first(private, "SpeedProfileAction")
            if sp is not None:
                for entry in find_all(sp, "SpeedProfileEntry"):
                    e.speed_profile.append(SpeedEntry(
                        speed=attr_float(entry, "speed"),
                        duration=attr_float(entry, "time", 1.0)))

    def _parse_stories(self):
        for mg in find_all(self.root, "ManeuverGroup"):
            actors = [attr_str(er, "entityRef")
                      for er in find_all(mg, "EntityRef")]
            for ara in find_all(mg, "AssignRouteAction"):
                for wp in find_all(ara, "Waypoint"):
                    pos = self._read_world_position(wp)
                    if pos is None:
                        continue
                    for name in actors:
                        e = self._entities.get(name)
                        if e is not None:
                            e.waypoints.append(pos)

    def _parse_environment(self):
        for ea in find_all(self.root, "EnvironmentAction"):
            tod = find_first(ea, "TimeOfDay")
            if tod is not None:
                dt = attr_str(tod, "dateTime", "")
                if "T" in dt:
                    try:
                        self.scenario.tod_hour = int(
                            dt.split("T")[1].split(":")[0])
                    except (IndexError, ValueError):
                        pass

            wn = find_first(ea, "Weather")
            if wn is None:
                continue
            w = self.scenario.weather
            sun = find_first(wn, "Sun")
            if sun is not None:
                w.sun_elevation_rad = attr_float(sun, "elevation", 1.31)
            fog = find_first(wn, "Fog")
            if fog is not None:
                w.fog_range = attr_float(fog, "visualRange", 100000.0)
            precip = find_first(wn, "Precipitation")
            if precip is not None:
                w.precipitation = attr_str(
                    precip, "precipitationType", "none").lower()
                w.rain_intensity = attr_float(precip, "intensity")

    def _parse_end_condition(self):
        stop = find_first(self.root, "StopTrigger")
        if stop is not None:
            stc = find_first(stop, "SimulationTimeCondition")
            if stc is not None:
                self.scenario.duration_sec = attr_float(stc, "value",
                                                        DEFAULT_DURATION)
                return

        # SCTrans stop triggers fire on storyboard completion; the actual
        # duration is the longest recorded speed profile.
        max_profile = max((e.profile_duration
                           for e in self._entities.values()), default=0.0)
        if max_profile > 0:
            self.scenario.duration_sec = max_profile + DURATION_BUFFER

    def _read_world_position(self, node) -> Optional[Position]:
        wp = find_first(node, "WorldPosition")
        if wp is None:
            return None
        return Position(x=attr_float(wp, "x"), y=attr_float(wp, "y"),
                        heading_rad=attr_float(wp, "h"))


# ─────────────────────────────────────────────────────────────────
# Waypoint utilities
# ─────────────────────────────────────────────────────────────────

def dedupe_waypoints(points: List[Position],
                     tol: float = WAYPOINT_MIN_SPACING) -> List[Position]:
    """Drop consecutive points closer than `tol` metres."""
    if not points:
        return []
    out = [points[0]]
    for p in points[1:]:
        prev = out[-1]
        if math.hypot(p.x - prev.x, p.y - prev.y) >= tol:
            out.append(p)
    return out


def downsample(points: List[Position],
               max_count: int = MAX_ROUTE_WAYPOINTS) -> List[Position]:
    """Keep at most max_count points, always retaining first and last."""
    if len(points) <= max_count:
        return points
    step = (len(points) - 1) / (max_count - 1)
    return [points[round(i * step)] for i in range(max_count)]


def route_points(e: Entity) -> List[Position]:
    return downsample(dedupe_waypoints(e.waypoints))


def waypoint_triples(pts: List[Position]) -> List[tuple]:
    """(x, y, scenic_heading_rad) triples for heading-aware lane matching.
    When the source left every heading at 0.0 (unset), headings are derived
    from the motion direction between neighbouring waypoints instead."""
    all_zero = all(abs(p.heading_rad) < 1e-9 for p in pts)
    out = []
    for i, p in enumerate(pts):
        pos = p
        if all_zero and len(pts) > 1:
            a = pts[max(i - 1, 0)]
            b = pts[min(i + 1, len(pts) - 1)]
            if a is not b and (a.x != b.x or a.y != b.y):
                pos = Position(p.x, p.y, math.atan2(b.y - a.y, b.x - a.x))
        out.append((pos.x, pos.y, pos.scenic_heading_rad))
    return out


# ─────────────────────────────────────────────────────────────────
# ScenicRules-style code generator (placement only)
# ─────────────────────────────────────────────────────────────────

BANNER_WIDTH = 33


class ScenicRulesWriter:
    """Emits a Scenic 3 program in the ScenicRules section layout,
    containing only entity placement and recorded data."""

    def __init__(self, scenario: Scenario, output_path: str):
        self.s = scenario
        self.output_path = output_path
        self._lines: List[str] = []
        self.var: dict = {}
        for e in self.s.entities:
            self.var[e.name] = "ego" if e.is_ego else safe_name(e.name).lower()

    def _w(self, line: str = ""):
        self._lines.append(line)

    def _banner(self, title: str):
        self._w("#" * BANNER_WIDTH)
        self._w(f"# {title:<{BANNER_WIDTH - 4}} #")
        self._w("#" * BANNER_WIDTH)
        self._w()

    def write(self) -> str:
        self._lines = []
        self._docstring()
        self._map_and_model()
        self._constants()
        self._spatial_relations()
        self._agent_behaviors()
        self._entities()
        self._termination()
        return "\n".join(self._lines) + "\n"

    def _speed_const(self, e: Entity) -> str:
        return ("EGO_SPEED" if e.is_ego
                else f"{safe_name(e.name).upper()}_SPEED")

    # ── sections ────────────────────────────────────────────────

    def _docstring(self):
        s = self.s
        name = os.path.basename(self.output_path)
        moving = sum(1 for e in s.npcs
                     if not e.is_parked and not e.is_pedestrian)
        parked = sum(1 for e in s.npcs
                     if e.is_parked and not e.is_pedestrian)
        peds = sum(1 for e in s.npcs if e.is_pedestrian)
        self._w('"""')
        self._w(f"SCENARIO: {name}")
        self._w(f"SOURCE FILE: {os.path.basename(s.source_file)} (SCTrans)")
        if s.description:
            self._w(f"SOURCE HEADER: {s.description}")
        self._w(f"ENTITIES: ego + {moving} moving NPC(s), {parked} parked, "
                f"{peds} pedestrian(s)")
        self._w("NOTE: placement is the exact recorded data (positions, "
                "headings, speeds);")
        self._w("      moving NPCs drive their recorded routes, the ego "
                "drives toward its")
        self._w("      recorded destination. Triggers and rule recording "
                "come in later steps.")
        self._w('"""')
        self._w()

    def _map_and_model(self):
        self._banner("MAP AND MODEL")
        self._w(f"param map = localPath('{self._relative_map_path()}')")
        self._w("param use2DMap = True  # 2D map without needing --2d or "
                "-p use2DMap on the CLI")
        self._w("model scenic.simulators.metadrive.model")
        self._w()
        self._w("# Run (needs <map>.net.xml beside the .xodr):")
        self._w("#   scenic <this file> --simulate")
        self._w("# The ScenicRules pipeline can still override the model:")
        self._w("#   scenic.scenarioFromFile(..., model='...', mode2D=True) "
                "or scenic --model ...")
        self._w()

    def _relative_map_path(self) -> str:
        out_dir = os.path.dirname(os.path.abspath(self.output_path)) or "."
        xodr = self.s.map_file
        if not os.path.isabs(xodr):
            xodr = os.path.join(
                os.path.dirname(os.path.abspath(self.s.source_file)), xodr)
        rel = os.path.relpath(xodr, out_dir)
        return rel.replace("\\", "/")

    def _constants(self):
        self._banner("CONSTANTS")
        self._w(f"SIM_DURATION = {int(round(self.s.duration_sec))}  "
                "# s, from the recorded speed profiles")
        self._w()
        self._w("# Cruise speeds (m/s) — mean of each recorded moving "
                "profile")
        for e in self.s.entities:
            if e.spawn is None or (e.is_parked and not e.is_ego):
                continue
            src = ("recorded" if not e.is_parked
                   else "recorded ego speed ~0; nominal urban speed")
            self._w(f"{self._speed_const(e)} = {e.target_speed:.2f}  # {src}")
        ego = self.s.ego
        if ego is not None:
            self._w()
            self._w("# Converter-added driving parameters (no equivalent "
                    "in the recording)")
            self._w(f"EGO_SAFETY_DIST = {EGO_SAFETY_DIST:.0f}  "
                    "# m, ego brakes when anything is closer")
            self._w(f"EGO_BRAKE = {EGO_BRAKE}")
            if ego.goal is not None:
                self._w(f"EGO_GOAL_RADIUS = {GOAL_RADIUS:.0f}  "
                        "# m, counts as arrival at the destination")
        w = self.s.weather
        self._w()
        self._w("# Environment from the source file (kept for reference)")
        self._w(f"param time_of_day = {self.s.tod_hour}")
        self._w(f"param weather_precipitation = '{w.precipitation}'")
        self._w(f"param weather_rain_intensity = {w.rain_intensity:.2f}")
        self._w(f"param weather_fog_range = {w.fog_range:.0f}")
        self._w()

    def _spatial_relations(self):
        self._banner("SPATIAL RELATIONS")
        self._w("# Original SCTrans world coordinates and headings.")
        self._w("# Scenic heading 0 deg = +y axis (north), counter-clockwise;")
        self._w("# converted from OpenSCENARIO h (0 = +x axis / east).")
        self._w()

        ego = self.s.ego
        if ego is not None and ego.spawn is not None:
            self._w(self._point_line("egoSpawnPt", ego.spawn))
            if ego.goal is not None:
                self._w(self._point_line("egoGoalPt", ego.goal)
                        + "  # recorded destination")
        for e in self.s.npcs:
            if e.spawn is None:
                continue
            var = self.var[e.name]
            self._w(self._point_line(f"{var}SpawnPt", e.spawn))
            if e.is_pedestrian and not e.is_parked:
                end = e.waypoints[-1] if e.waypoints else None
                if end is not None and (abs(end.x - e.spawn.x) > 1.0
                                        or abs(end.y - e.spawn.y) > 1.0):
                    self._w(f"{var}EndPt = new OrientedPoint at "
                            f"({end.x:.4f}, {end.y:.4f})"
                            f"  # end of recorded walk")
                else:
                    self._w(f"{var}EndPt = new OrientedPoint at {var}SpawnPt "
                            f"offset along {var}SpawnPt.heading by (0, 10)")
        self._w()

        # Recorded routes: (x, y, heading) — heading picks the right lane
        # among overlapping ones when the route is mapped onto the network.
        any_route = False
        for e in self.s.entities:
            if e.is_pedestrian or e.is_parked:
                continue
            pts = route_points(e)
            if len(pts) < 2:
                continue
            if not any_route:
                self._w("# Recorded route waypoints (x, y, heading_rad)")
                any_route = True
            self._emit_waypoints(f"{self.var[e.name]}Waypoints", pts)
        if any_route:
            self._w()
        self._route_helpers()

    def _point_line(self, var: str, pos: Position) -> str:
        return (f"{var} = new OrientedPoint at ({pos.x:.4f}, {pos.y:.4f}), "
                f"facing {pos.scenic_heading_deg:.2f} deg")

    def _emit_waypoints(self, var: str, pts: List[Position]):
        triples = waypoint_triples(pts)
        self._w(f"{var} = [")
        for i in range(0, len(triples), 3):
            chunk = triples[i:i + 3]
            row = ", ".join(f"({x:.2f}, {y:.2f}, {h:.3f})"
                            for x, y, h in chunk)
            self._w(f"    {row},")
        self._w("]")

    def _has_vehicle_route(self, e: Entity) -> bool:
        return (not e.is_pedestrian and not e.is_parked
                and len(route_points(e)) >= 2)

    def _route_helpers(self):
        """Route derivation emitted into the scenario. Lane matching is only
        used for STEERING — placement keeps the exact recorded coordinates."""
        ego = self.s.ego
        need_goal_router = (ego is not None and ego.spawn is not None
                            and ego.goal is not None
                            and not self._has_vehicle_route(ego))
        have_routes = any(self._has_vehicle_route(e)
                          for e in self.s.entities)
        if not (have_routes or need_goal_router):
            return
        w = self._w
        w("import math as _math")
        w()
        w("def bestLaneAt(x, y, heading):")
        w('    """Containing lane whose direction best matches the recorded '
          'heading')
        w('    (several lanes can overlap near junctions)."""')
        w("    pt = Vector(x, y)")
        w("    candidates = [l for l in network.lanes if l.containsPoint(pt)]")
        w("    if not candidates:")
        w("        return network.laneAt(pt)")
        w("    def _hdiff(l):")
        w("        value = l.orientation.value(pt)")
        w("        yaw = value.yaw if hasattr(value, 'yaw') else float(value)")
        w("        d = abs(yaw - heading) % (2 * _math.pi)")
        w("        return min(d, 2 * _math.pi - d)")
        w("    return min(candidates, key=_hdiff)")
        w()
        if have_routes:
            w("def lanesFromWaypoints(waypoints):")
            w('    """Ordered lane sequence traversed by a recorded '
              '(x, y, heading) route."""')
            w("    lanes = []")
            w("    for x, y, h in waypoints:")
            w("        lane = bestLaneAt(x, y, h)")
            w("        if lane is not None and (not lanes or "
              "lanes[-1] is not lane):")
            w("            lanes.append(lane)")
            w("    return lanes")
            w()
        w("def connectedSegments(lanes, gap=1.0):")
        w('    """Split a lane route wherever junction connectivity is '
          'missing,')
        w('    so each piece can be followed with '
          'FollowTrajectoryBehavior."""')
        w("    segments = []")
        w("    for lane in lanes:")
        w("        if segments:")
        w("            pe = segments[-1][-1].centerline.points[-1]")
        w("            ls = lane.centerline.points[0]")
        w("            if _math.hypot(ls[0] - pe[0], ls[1] - pe[1]) <= gap:")
        w("                segments[-1].append(lane)")
        w("                continue")
        w("        segments.append([lane])")
        w("    return segments")
        w()
        if need_goal_router:
            w("def lanesTowardGoal(spawnPt, spawnHeading, goalPt, "
              "maxHops=14):")
            w('    """Greedy lane chaining toward the recorded destination; '
              'stitches across')
            w('    junctions with missing connectivity."""')
            w("    from shapely.geometry import Point as _ShPoint")
            w("    goal = _ShPoint(goalPt.position.x, goalPt.position.y)")
            w("    lane = bestLaneAt(spawnPt.position.x, "
              "spawnPt.position.y, spawnHeading)")
            w("    if lane is None:")
            w("        return []")
            w("    route = [lane]")
            w("    visited = {lane.uid}")
            w("    for _ in range(maxHops):")
            w("        cur = route[-1]")
            w("        if cur.polygon.distance(goal) <= 1.0:")
            w("            break")
            w("        nxt = []")
            w("        for m in cur.maneuvers:")
            w("            step = m.connectingLane if m.connectingLane "
              "is not None else m.endLane")
            w("            if step is not None and step.uid not in visited:")
            w("                nxt.append(step)")
            w("        succ = getattr(cur, '_successor', None)")
            w("        if not nxt and succ is not None and "
              "succ.uid not in visited:")
            w("            nxt.append(succ)")
            w("        if not nxt:")
            w("            ex, ey = cur.centerline.points[-1][0], "
              "cur.centerline.points[-1][1]")
            w("            for l in network.lanes:")
            w("                if l.uid in visited:")
            w("                    continue")
            w("                ls = l.centerline.points[0]")
            w("                if _math.hypot(ls[0] - ex, ls[1] - ey) "
              "<= 4.0:")
            w("                    nxt.append(l)")
            w("        if not nxt:")
            w("            break")
            w("        best = min(nxt, key=lambda l: "
              "l.polygon.distance(goal))")
            w("        route.append(best)")
            w("        visited.add(best.uid)")
            w("    return route")
            w()
        if ego is not None and ego.spawn is not None:
            if self._has_vehicle_route(ego):
                w("egoRouteLanes = lanesFromWaypoints(egoWaypoints)")
            elif ego.goal is not None:
                w("egoRouteLanes = lanesTowardGoal(egoSpawnPt, "
                  "egoSpawnPt.heading, egoGoalPt)")
        for e in self.s.npcs:
            if self._has_vehicle_route(e) and e.spawn is not None:
                var = self.var[e.name]
                w(f"{var}RouteLanes = lanesFromWaypoints({var}Waypoints)")
        self._w()

    def _agent_behaviors(self):
        self._banner("AGENT BEHAVIORS")
        ego = self.s.ego

        if ego is not None and ego.spawn is not None:
            has_route = (self._has_vehicle_route(ego)
                         or ego.goal is not None)
            self._w("behavior EgoBehavior():")
            self._w("    try:")
            if has_route:
                self._w("        if egoRouteLanes:")
                self._w("            for seg in "
                        "connectedSegments(egoRouteLanes):")
                self._w("                do FollowTrajectoryBehavior("
                        "target_speed=EGO_SPEED, trajectory=seg)")
                self._w("            # route complete — stop")
                self._w("            while True:")
                self._w("                take SetBrakeAction(1.0)")
                self._w("        else:")
                self._w("            do FollowLaneBehavior("
                        "target_speed=EGO_SPEED)")
            else:
                self._w("        do FollowLaneBehavior("
                        "target_speed=EGO_SPEED)")
            self._w("    interrupt when withinDistanceToAnyObjs(self, "
                    "EGO_SAFETY_DIST):")
            self._w("        take SetBrakeAction(EGO_BRAKE)")
            self._w()

        for e in self.s.npcs:
            if e.spawn is None or e.is_parked:
                continue
            var = self.var[e.name]
            bname = f"{safe_name(e.name).capitalize()}Behavior"
            sconst = self._speed_const(e)
            if e.is_pedestrian:
                self._w(f"behavior {bname}():")
                self._w(f"    take SetWalkingSpeedAction(speed={sconst})")
                self._w()
                continue
            self._w(f"behavior {bname}():")
            if self._has_vehicle_route(e):
                self._w(f"    if {var}RouteLanes:")
                self._w(f"        for seg in connectedSegments("
                        f"{var}RouteLanes):")
                self._w(f"            do FollowTrajectoryBehavior("
                        f"target_speed={sconst}, trajectory=seg)")
                self._w("        # recorded route complete — stop")
                self._w("        while True:")
                self._w("            take SetBrakeAction(1.0)")
                self._w("    else:")
                self._w("        # route did not map onto network lanes")
                self._w(f"        do FollowLaneBehavior("
                        f"target_speed={sconst})")
            else:
                self._w(f"    do FollowLaneBehavior(target_speed={sconst})")
            self._w()

    def _entities(self):
        self._banner("ENTITIES")
        self._w("# 'with regionContainedIn None' keeps the exact recorded")
        self._w("# positions (default containment would reject off-road/"
                "junction spawns).")
        self._w("# 'with allowCollisions True' tolerates LGSVL's inflated")
        self._w("# bounding boxes, which overlap in dense recorded traffic.")
        self._w()

        ego = self.s.ego
        if ego is not None and ego.spawn is not None:
            self._declare(ego, "ego")
        for e in self.s.npcs:
            if e.spawn is None:
                self._w(f"# {e.name}: skipped (no spawn position in source)")
                self._w()
                continue
            self._declare(e, self.var[e.name])

    def _declare(self, e: Entity, var: str):
        cls = XOSCParser.SCENIC_CLASS.get(e.category, "Car")
        note = ""
        if e.category not in ("car", "pedestrian"):
            note = f"  # source category: {e.category}"
        if e.is_parked:
            self._w(f"# {e.name}: parked in the recorded data "
                    f"(peak speed < {PARKED_SPEED_THRESHOLD} m/s)")
        if e.is_pedestrian and not e.is_parked:
            head = (f"{var} = new {cls} at {var}SpawnPt, "
                    f"facing toward {var}EndPt,{note}")
        else:
            head = (f"{var} = new {cls} at {var}SpawnPt, "
                    f"facing {var}SpawnPt.heading,{note}")
        self._w(head)
        if e.blueprint:
            self._w(f"      with blueprint '{e.blueprint}',")
        if e.length > 0 and e.width > 0 and not e.is_pedestrian:
            self._w(f"      with length {e.length:.2f}, "
                    f"with width {e.width:.2f},")
        self._w("      with regionContainedIn None,")
        if e.is_parked:
            self._w("      with allowCollisions True")
        else:
            self._w("      with allowCollisions True,")
            if not e.is_pedestrian and e.init_speed > PARKED_SPEED_THRESHOLD:
                self._w(f"      with speed {e.init_speed:.4f},  "
                        f"# recorded initial speed (m/s)")
            bname = ("EgoBehavior" if e.is_ego
                     else f"{safe_name(e.name).capitalize()}Behavior")
            self._w(f"      with behavior {bname}()")
        self._w()

    def _termination(self):
        self._banner("TERMINATION")
        ego = self.s.ego
        if ego is not None and ego.goal is not None and ego.spawn is not None:
            trip = math.hypot(ego.goal.x - ego.spawn.x,
                              ego.goal.y - ego.spawn.y)
            if trip > MIN_GOAL_TRIP:
                self._w("terminate when (distance from ego to egoGoalPt) "
                        "<= EGO_GOAL_RADIUS")
            else:
                self._w(f"# recorded trip is only {trip:.1f} m; goal-based "
                        "termination omitted (would fire immediately)")
        self._w("terminate after SIM_DURATION seconds")


# ─────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────

def convert_file(input_path: str, output_path: Optional[str] = None) -> str:
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".scenic"

    scenario = XOSCParser(input_path).parse()
    code = ScenicRulesWriter(scenario, output_path).write()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)

    ego = scenario.ego
    moving = sum(1 for e in scenario.npcs
                 if not e.is_parked and not e.is_pedestrian)
    parked = sum(1 for e in scenario.npcs
                 if e.is_parked and not e.is_pedestrian)
    peds = sum(1 for e in scenario.npcs if e.is_pedestrian)
    print(f"  map={scenario.map_name}  ego={'yes' if ego else 'NO'}  "
          f"moving={moving} parked={parked} peds={peds}  "
          f"duration={scenario.duration_sec:.0f}s")
    return output_path


def main():
    ap = argparse.ArgumentParser(
        description="Convert SCTrans OpenSCENARIO (.xosc) to minimal "
                    "ScenicRules-style Scenic 3 scenarios (placement only).")
    ap.add_argument("input", help=".xosc file or a directory of .xosc files")
    ap.add_argument("output", nargs="?",
                    help="output .scenic path (single file mode)")
    ap.add_argument("-d", "--outdir", help="output directory")
    ap.add_argument("--limit", type=int, default=0,
                    help="batch: convert at most N files")
    args = ap.parse_args()

    if os.path.isdir(args.input):
        files = sorted(f for f in os.listdir(args.input)
                       if f.lower().endswith(".xosc"))
        if args.limit:
            files = files[:args.limit]
        outdir = args.outdir or args.input
        os.makedirs(outdir, exist_ok=True)
        inputs = [(os.path.join(args.input, f),
                   os.path.join(outdir, os.path.splitext(f)[0] + ".scenic"))
                  for f in files]
    else:
        out = args.output
        if out is None and args.outdir:
            os.makedirs(args.outdir, exist_ok=True)
            base = os.path.splitext(os.path.basename(args.input))[0]
            out = os.path.join(args.outdir, base + ".scenic")
        inputs = [(args.input, out)]

    ok = 0
    for src, dst in inputs:
        print(f"Converting {os.path.basename(src)}")
        try:
            convert_file(src, dst)
            ok += 1
        except ET.ParseError as ex:
            print(f"  XML error: {ex}")
        except Exception as ex:  # noqa: BLE001 — keep batch going
            print(f"  ERROR: {type(ex).__name__}: {ex}")

    print(f"\nDone: {ok}/{len(inputs)} converted.")


if __name__ == "__main__":
    main()
