#!/usr/bin/env python3
"""
xosc_to_scenic.py
=================
Converts any OpenSCENARIO (.xosc) file into a Scenic 3 scenario
that runs on the MetaDrive simulator.

Usage:
    python xosc_to_scenic.py input.xosc              # saves input.scenic
    python xosc_to_scenic.py input.xosc output.scenic

How it works:
    1. Parse  – reads the .xosc XML and extracts every relevant piece
                of data into plain Python objects (no hard-coded paths).
    2. Generate – turns those objects into a valid .scenic file that
                  follows the Scenic 3 / MetaDrive model conventions.
"""

import xml.etree.ElementTree as ET
import math, os, sys, argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────
# STEP 1: Simple data containers
# ─────────────────────────────────────────────────────────────────

@dataclass
class Position:
    """World-space position and heading extracted from WorldPosition."""
    x: float = 0.0
    y: float = 0.0
    heading_rad: float = 0.0   # heading in radians (from xosc attribute h)

    @property
    def heading_deg(self):
        return math.degrees(self.heading_rad)


@dataclass
class SpeedEntry:
    """One entry in a SpeedProfileAction: drive at `speed` m/s for `duration` seconds."""
    speed: float   # m/s
    duration: float  # seconds


@dataclass
class Entity:
    """Everything Scenic needs to know about one scenario object."""
    name: str
    is_ego: bool = False
    scenic_class: str = "Car"        # Car, Pedestrian, etc.
    blueprint: str = ""              # vehicle model name from xosc
    spawn: Optional[Position] = None # where it starts
    init_speed: float = 0.0          # starting speed in m/s
    controller_module: str = ""      # e.g. "external_control", "waypoint_vehicle_control"
    waypoints: List[Position] = field(default_factory=list)   # route to follow
    speed_profile: List[SpeedEntry] = field(default_factory=list)  # timed speed changes


@dataclass
class Weather:
    """Weather conditions extracted from EnvironmentAction."""
    precipitation: str = "none"      # "rain", "snow", "none"
    rain_intensity: float = 0.0      # 0.0 – 1.0
    fog_range: float = 100000.0      # visibility in metres
    sun_elevation_rad: float = 1.31  # radians


@dataclass
class Scenario:
    """Top-level container for everything parsed from the .xosc file."""
    description: str = ""
    map_file: str = ""               # path to .xodr file
    map_name: str = ""               # short name (used as carla_map / sumo_map key)
    entities: List[Entity] = field(default_factory=list)
    weather: Weather = field(default_factory=Weather)
    tod_hour: int = 12               # time of day hour
    duration_sec: float = 60.0      # how long to run the scenario


# ─────────────────────────────────────────────────────────────────
# STEP 2: XML helpers  (tag-name based, namespace-safe)
# ─────────────────────────────────────────────────────────────────

def tag(element) -> str:
    """Return the local tag name, stripping any XML namespace."""
    t = element.tag
    return t.split("}")[-1] if "}" in t else t


def find_all(root, tag_name: str) -> list:
    """Return every descendant element whose local tag equals tag_name."""
    return [el for el in root.iter() if tag(el) == tag_name]


def find_first(root, tag_name: str):
    """Return the first matching descendant, or None."""
    results = find_all(root, tag_name)
    return results[0] if results else None


def attr_float(el, name: str, default: float = 0.0) -> float:
    try:
        return float(el.get(name, default))
    except (TypeError, ValueError):
        return default


def attr_str(el, name: str, default: str = "") -> str:
    return el.get(name, default)


# ─────────────────────────────────────────────────────────────────
# STEP 3: Parser
# ─────────────────────────────────────────────────────────────────

class XOSCParser:
    """
    Reads a .xosc file and fills a Scenario object.
    Works on any valid OpenSCENARIO file regardless of tag capitalisation,
    nesting depth, or which optional elements are present.
    """

    # Map xosc vehicleCategory → Scenic class name.
    # MetaDrive only supports Car, NPCCar, and Pedestrian.
    # Everything else (truck, bus, bicycle, motorbike, etc.) falls back to Car.
    SCENIC_CLASS = {
        "car":        "Car",
        "van":        "Car",
        "truck":      "Car",       # no Truck in MetaDrive → Car
        "bus":        "Car",       # no Bus in MetaDrive → Car
        "bicycle":    "Car",       # no Bicycle in MetaDrive → Car
        "motorbike":  "Car",       # no Motorcycle in MetaDrive → Car
        "trailer":    "Car",       # no Trailer in MetaDrive → Car
        "pedestrian": "Pedestrian",
    }

    def __init__(self, path: str):
        self.tree = ET.parse(path)
        self.root = self.tree.getroot()
        self.scenario = Scenario()
        self._entities: dict = {}   # name → Entity  (built incrementally)

    # ── public ──────────────────────────────────────────────────

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

    # ── header / map ────────────────────────────────────────────

    def _parse_header(self):
        h = find_first(self.root, "FileHeader")
        if h is not None:
            desc   = attr_str(h, "description")
            author = attr_str(h, "author")
            self.scenario.description = f"{desc} (author: {author})" if author else desc

    def _parse_map(self):
        lf = find_first(self.root, "LogicFile")
        if lf is not None:
            path = attr_str(lf, "filepath")
            self.scenario.map_file = path
            # derive short name from filename without extension
            self.scenario.map_name = os.path.splitext(os.path.basename(path))[0]

    # ── entities ────────────────────────────────────────────────

    def _parse_entities(self):
        """Create an Entity for every ScenarioObject in <Entities>."""
        ent_root = find_first(self.root, "Entities")
        if ent_root is None:
            return

        for so in find_all(ent_root, "ScenarioObject"):
            name = attr_str(so, "name", "entity")
            e = Entity(name=name)

            # ── vehicle ──
            veh = find_first(so, "Vehicle")
            if veh is not None:
                category = attr_str(veh, "vehicleCategory", "car").lower()
                e.scenic_class = self.SCENIC_CLASS.get(category, "Car")
                e.blueprint    = attr_str(veh, "name")

            # ── pedestrian ──
            ped = find_first(so, "Pedestrian")
            if ped is not None:
                e.scenic_class = "Pedestrian"
                e.blueprint    = attr_str(ped, "model")

            # ── ego detection via <Property name="type" value="ego_vehicle"> ──
            for prop in find_all(so, "Property"):
                if attr_str(prop, "name") == "type" and attr_str(prop, "value") == "ego_vehicle":
                    e.is_ego = True

            self._entities[name] = e

        # fallback: detect ego by name if Property tag was not used
        if not any(e.is_ego for e in self._entities.values()):
            for e in self._entities.values():
                if "ego" in e.name.lower():
                    e.is_ego = True
                    break

    # ── init ────────────────────────────────────────────────────

    def _parse_init(self):
        """
        Read <Init><Actions><Private> blocks to get each entity's
        spawn position, initial speed, controller, and speed profile.
        """
        init = find_first(self.root, "Init")
        if init is None:
            return

        for private in find_all(init, "Private"):
            name = attr_str(private, "entityRef")
            e = self._entities.get(name)
            if e is None:
                continue

            # spawn position  (TeleportAction takes priority over AcquirePositionAction)
            for action_tag in ("TeleportAction", "AcquirePositionAction"):
                node = find_first(private, action_tag)
                if node is not None:
                    pos = self._read_world_position(node)
                    if pos:
                        e.spawn = pos
                        break

            # initial speed
            speed_node = find_first(private, "AbsoluteTargetSpeed")
            if speed_node is not None:
                e.init_speed = attr_float(speed_node, "value")

            # controller module  (tells us if this is ego or an NPC autopilot)
            ctrl = find_first(private, "Controller")
            if ctrl is not None:
                for prop in find_all(ctrl, "Property"):
                    if attr_str(prop, "name") == "module":
                        e.controller_module = attr_str(prop, "value")

            # per-second speed profile
            sp = find_first(private, "SpeedProfileAction")
            if sp is not None:
                for entry in find_all(sp, "SpeedProfileEntry"):
                    e.speed_profile.append(SpeedEntry(
                        speed    = attr_float(entry, "speed"),
                        duration = attr_float(entry, "time", 1.0),
                    ))

    # ── stories (NPC route waypoints) ───────────────────────────

    def _parse_stories(self):
        """
        Read AssignRouteAction waypoints from each ManeuverGroup.
        These define the path each NPC follows during the scenario.
        """
        for mg in find_all(self.root, "ManeuverGroup"):
            # collect actor names
            actors = [attr_str(er, "entityRef") for er in find_all(mg, "EntityRef")]

            # each AssignRouteAction inside this group = waypoint list
            for ara in find_all(mg, "AssignRouteAction"):
                for wp in find_all(ara, "Waypoint"):
                    pos = self._read_world_position(wp)
                    if pos is None:
                        continue
                    for name in actors:
                        e = self._entities.get(name)
                        if e:
                            e.waypoints.append(pos)

    # ── environment ─────────────────────────────────────────────

    def _parse_environment(self):
        """Extract weather and time-of-day from any EnvironmentAction."""
        for ea in find_all(self.root, "EnvironmentAction"):
            # time of day
            tod = find_first(ea, "TimeOfDay")
            if tod is not None:
                dt = attr_str(tod, "dateTime", "")
                if "T" in dt:
                    try:
                        self.scenario.tod_hour = int(dt.split("T")[1].split(":")[0])
                    except (IndexError, ValueError):
                        pass

            # weather
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
                w.precipitation  = attr_str(precip, "precipitationType", "none").lower()
                w.rain_intensity = attr_float(precip, "intensity")

    # ── end condition ───────────────────────────────────────────

    def _parse_end_condition(self):
        """
        Estimate scenario duration:
        - If SimulationTimeCondition is in StopTrigger → use that value.
        - Otherwise estimate from speed profile length (1 entry = 1 second).
        """
        stop = find_first(self.root, "StopTrigger")
        if stop is not None:
            stc = find_first(stop, "SimulationTimeCondition")
            if stc is not None:
                self.scenario.duration_sec = attr_float(stc, "value", 60.0)
                return

        # estimate: longest speed profile among all entities
        max_profile_time = 0.0
        for e in self._entities.values():
            total = sum(entry.duration for entry in e.speed_profile)
            max_profile_time = max(max_profile_time, total)

        if max_profile_time > 0:
            self.scenario.duration_sec = max_profile_time + 5.0  # small buffer

    # ── helper ──────────────────────────────────────────────────

    def _read_world_position(self, node) -> Optional[Position]:
        """Find the first WorldPosition inside `node` and return a Position."""
        wp = find_first(node, "WorldPosition")
        if wp is None:
            return None
        return Position(
            x           = attr_float(wp, "x"),
            y           = attr_float(wp, "y"),
            heading_rad = attr_float(wp, "h"),
        )


# ─────────────────────────────────────────────────────────────────
# STEP 4: Scenic 3 / MetaDrive code generator
# ─────────────────────────────────────────────────────────────────

class ScenicWriter:
    """
    Turns a Scenario object into a Scenic 3 script targeting MetaDrive.

    MetaDrive Scenic 3 conventions used here:
      - model scenic.simulators.metadrive.model
      - param map  = path to .xodr
      - param sumo_map = path to .net.xml  (must exist alongside .xodr)
      - Classes: Car, Pedestrian (from driving domain)
      - Behaviors: FollowLaneBehavior, FollowTrajectoryBehavior
      - Actions: SetSpeedAction
      - Positions: given as (x, y) tuples
      - Headings: degrees, clockwise from north  (MetaDrive convention)
    """

    def __init__(self, scenario: Scenario, source_file: str = ""):
        self.s = scenario
        self.source = source_file
        self._lines: List[str] = []

    def write(self) -> str:
        self._lines = []
        self._file_header()
        self._map_and_model()
        self._weather_params()
        self._behavior_definitions()   # behaviors MUST come before entities that use them
        self._entity_declarations()
        self._termination()
        return "\n".join(self._lines)

    # ── internal helpers ─────────────────────────────────────────

    def _w(self, line: str = ""):
        self._lines.append(line)

    def _section(self, title: str):
        self._w()
        self._w(f"# {'─' * 55}")
        self._w(f"# {title}")
        self._w(f"# {'─' * 55}")

    # ── sections ─────────────────────────────────────────────────

    def _file_header(self):
        """Comment block at top of file."""
        self._w(f"# Scenic 3 scenario — MetaDrive compatible")
        if self.source:
            self._w(f"# Converted from: {os.path.basename(self.source)}")
        if self.s.description:
            self._w(f"# Description   : {self.s.description}")
        self._w()

    def _map_and_model(self):
        """
        param map        → path to the OpenDRIVE (.xodr) road network file
        param sumo_map   → path to the SUMO (.net.xml) file MetaDrive needs
                           (must be in the same folder as the .xodr, same base name)
        param use2DMap   → required by Scenic 3; 3D maps not yet supported
        model            → loads MetaDrive classes + behaviors
        """
        self._section("Map and simulator")
        xodr = self.s.map_file or "maps/your_map.xodr"
        # MetaDrive needs a SUMO .net.xml alongside the .xodr
        sumo = os.path.splitext(xodr)[0] + ".net.xml"
        self._w(f'param use2DMap = True   # required — Scenic 3 does not support 3D maps yet')
        self._w()
        self._w(f'param map      = "{xodr}"')
        self._w(f'param sumo_map = "{sumo}"')
        self._w()
        self._w("model scenic.simulators.metadrive.model")

    def _weather_params(self):
        """
        MetaDrive does not have a single weather object like CARLA,
        so we express weather as scenario parameters that can be read
        by the simulation wrapper if needed.
        """
        self._section("Weather / environment parameters")
        w = self.s.weather
        self._w(f"param time_of_day         = {self.s.tod_hour}   # hour (0-23)")
        self._w(f"param weather_precipitation = '{w.precipitation}'  # none / rain / snow")
        self._w(f"param weather_rain_intensity = {w.rain_intensity:.2f}  # 0.0 – 1.0")
        self._w(f"param weather_fog_range      = {w.fog_range:.0f}     # metres")
        self._w(f"param sun_elevation_deg      = {math.degrees(w.sun_elevation_rad):.1f}")
        self._w()
        self._w("from scenic.domains.driving.behaviors import FollowLaneBehavior")
        self._emit_snap_helper()

    def _emit_snap_helper(self):
        """
        Emit the snap_to_lane() helper used by all entity declarations.

        WHY A HELPER INSTEAD OF 'on network.laneRegion':
          Scenic's 'on' projection onto a PolygonalRegion is NOT implemented
          in all versions:
            NotImplementedError: PolygonalRegion does not yet support
                                 projection using "on"
          So we cannot rely on Scenic to project a coordinate onto the lanes.
          snap_to_lane() does the projection ourselves with shapely.

        WHAT IT RETURNS: (snapped_position, heading)
          1. network.laneAt(pt) → the lane containing the point (safe: a plain
             point-in-polygon test, never touches junction maneuvers).
          2. if None (point a few cm off-lane) → nearest lane by centerline
             distance.
          3. project the point onto that lane's centerline → on-lane position.
          4. if that lands within min_gap m of an already-placed car, slide
             forward along the centerline until clear → prevents
             'InvalidScenarioError: ... Car ... intersects ... Car' when two
             xosc entities snap to nearly the same spot.
          5. heading from the centerline TANGENT (sample just behind/ahead and
             atan2). Computing heading ourselves means we never evaluate
             Scenic's default '(roadDirection at position)', which crashes on
             maps whose junctions have no maneuvers
             (ValueError: min() arg is an empty sequence).
        """
        self._section("Helper: snap xosc positions onto lane centerlines (+ heading)")
        self._w("import math as _math")
        self._w("from shapely.geometry import Point as _SPoint")
        self._w()
        self._w("# Set SNAP_DEBUG = False to silence the per-car snap log below.")
        self._w("SNAP_DEBUG  = True")
        self._w("_snap_used  = []   # positions already taken, to avoid overlaps")
        self._w("_snap_n     = 0    # how many cars have been placed")
        self._w("_snap_moved = 0    # how many were moved >0.5 m (off-lane / anti-overlap)")
        self._w()
        self._w("def snap_to_lane(x, y, min_gap=6.0):")
        self._w('    """Return (point_on_centerline, heading) for the lane nearest (x, y).')
        self._w("")
        self._w("    If the snapped point lands within min_gap metres of an already-")
        self._w("    placed car, slide it forward along the lane centerline until it")
        self._w("    clears them. Prevents Scenic's")
        self._w("    'InvalidScenarioError: ... Car ... intersects ... Car'.")
        self._w('    """')
        self._w("    lane = network.laneAt(Vector(x, y))")
        self._w("    if lane is None:")
        self._w("        # point is slightly off every lane polygon — take nearest lane")
        self._w("        lane = min(network.lanes,")
        self._w("                   key=lambda l: l.centerline.lineString.distance(_SPoint(x, y)))")
        self._w("    ls = lane.centerline.lineString")
        self._w("    d  = ls.project(_SPoint(x, y))          # distance along centerline")
        self._w("")
        self._w("    # slide forward in 1 m steps until clear of previously placed cars")
        self._w("    for _ in range(int(ls.length) + 1):")
        self._w("        p = ls.interpolate(d)")
        self._w("        if all((p.x - ux) ** 2 + (p.y - uy) ** 2 >= min_gap ** 2")
        self._w("               for ux, uy in _snap_used):")
        self._w("            break")
        self._w("        d = min(d + 1.0, ls.length)")
        self._w("    p = ls.interpolate(d)")
        self._w("    _snap_used.append((p.x, p.y))")
        self._w("")
        self._w("    # ── permanent diagnostic log ──")
        self._w("    global _snap_n, _snap_moved")
        self._w("    _snap_n += 1")
        self._w("    moved = ((p.x - x) ** 2 + (p.y - y) ** 2) ** 0.5")
        self._w("    if moved > 0.5:")
        self._w("        _snap_moved += 1")
        self._w("    if SNAP_DEBUG:")
        self._w("        flag = '  <-- moved' if moved > 0.5 else ''")
        self._w("        print(f'  snap #{_snap_n:<2} original=({x:8.2f},{y:8.2f}) '")
        self._w("              f'-> ({p.x:8.2f},{p.y:8.2f})  moved {moved:5.2f} m{flag}')")
        self._w("")
        self._w("    # tangent direction: sample just behind and just ahead of p")
        self._w("    a = ls.interpolate(max(d - 0.5, 0.0))")
        self._w("    b = ls.interpolate(min(d + 0.5, ls.length))")
        self._w("    # Scenic heading: radians, 0 = +y axis (north), counterclockwise")
        self._w("    heading = _math.atan2(b.y - a.y, b.x - a.x) - _math.pi / 2")
        self._w("    return Vector(p.x, p.y), heading")
        self._w()
        self._w("def snap_summary():")
        self._w('    """Print a one-line summary after all cars are placed."""')
        self._w("    if SNAP_DEBUG:")
        self._w("        print(f'  snap summary: {_snap_n} cars placed, '")
        self._w("              f'{_snap_moved} adjusted >0.5 m, '")
        self._w("              f'{_snap_n - _snap_moved} essentially unchanged')")

    def _entity_declarations(self):
        """Declare ego and all NPCs."""
        self._section("Entities")

        ego    = next((e for e in self.s.entities if e.is_ego), None)
        others = [e for e in self.s.entities if not e.is_ego]

        # ── ego ──
        if ego:
            self._w()
            self._w(f"# Ego vehicle: {ego.name}")
            self._declare_entity(ego, var_name="ego")

        # ── NPCs ──
        for e in others:
            self._w()
            self._w(f"# NPC: {e.name}  ({e.scenic_class})")
            self._declare_entity(e, var_name=safe_name(e.name))

        # one-line summary after every car has been snapped
        self._w()
        self._w("snap_summary()")

    def _declare_entity(self, e: Entity, var_name: str):
        """
        Emit one Scenic 3 object declaration — all on a single line.

        Why single line?
          Different Scenic versions have different tolerance for multi-line
          specifier continuations. A single line is always safe across all
          Scenic 3 versions.

        Scenic 3 specifier syntax:
          new ClassName at (x,y), facing N deg, with speed V, with behavior B()
          NO trailing comma after the last specifier.
          NO inline comments inside the declaration (put them on the line above).
        """
        spawn = e.spawn
        if spawn:
            pos_str = f"({spawn.x:.4f}, {spawn.y:.4f})"
            hdg_str = f"{spawn.heading_deg:.2f} deg"
        else:
            pos_str = "(0, 0)"
            self._w(f"# NOTE: spawn position not found in source file for {e.name}")

        has_behavior = bool(e.waypoints or e.speed_profile)

        # Build the specifier list — order matters in Scenic 3.
        #
        # PLACEMENT (snap-to-lane via helper) — see _emit_snap_helper():
        #   'at <snapped pos>' + 'with parentOrientation <computed heading>'.
        #   We do NOT use 'on network.laneRegion' because Scenic raises
        #   'NotImplementedError: PolygonalRegion does not yet support
        #   projection using "on"'. snap_to_lane() projects ourselves and also
        #   computes the heading, so Scenic's crashing default parentOrientation
        #   '(roadDirection at self.position)' is never evaluated.
        #
        # Fallback (no spawn position in xosc): random lane via
        # Uniform(*network.lanes). network.lanes is a Python LIST, so it
        # cannot be used directly with 'on' — Uniform(*...) picks one lane.
        if spawn:
            self._w(f"_{var_name}_pos, _{var_name}_hdg = snap_to_lane({spawn.x:.4f}, {spawn.y:.4f})")
            specs = [
                f"at _{var_name}_pos",
                f"with parentOrientation _{var_name}_hdg",
            ]
        else:
            specs = ["on Uniform(*network.lanes)", "with roadDeviation 0 deg"]

        if e.init_speed > 0:
            specs.append(f"with speed {e.init_speed:.4f}")

        # Behavior: every MOVING vehicle must steer, otherwise it travels in
        # a straight line and leaves the road at the first curve.
        #   - entities with xosc actions get their custom behavior
        #   - any other vehicle with speed > 0 gets FollowLaneBehavior so it
        #     follows the lane geometry instead of driving off the road
        if has_behavior:
            specs.append(f"with behavior {safe_name(e.name)}_Behavior()")
        elif e.init_speed > 0 and e.scenic_class != "Pedestrian":
            specs.append(f"with behavior FollowLaneBehavior({e.init_speed:.4f})")

        # Comments go ABOVE the declaration, never inside it
        comments = []
        if e.blueprint:
            comments.append(f"# blueprint: {e.blueprint}")
        if e.init_speed > 0:
            comments.append(f"# initial speed: {e.init_speed:.4f} m/s")
        for c in comments:
            self._w(c)

        # Emit everything on ONE line — safest across all Scenic 3 versions
        self._w(f"{var_name} = new {e.scenic_class} {', '.join(specs)}")
        self._w()

    def _behavior_definitions(self):
        """Write one behavior function for every entity that has actions."""
        for e in self.s.entities:
            if not (e.waypoints or e.speed_profile):
                continue   # no behavior needed

            self._section(f"Behavior for {e.name}")
            bname = safe_name(e.name)
            self._w(f"behavior {bname}_Behavior():")

            # ── route / lane following ──
            # Use FollowLaneBehavior — raw xosc waypoint coordinates do not map
            # 1-to-1 to MetaDrive lane positions, so FollowTrajectoryBehavior fails.
            if e.waypoints:
                self._w(f"    do FollowLaneBehavior()")

            # ── timed speed profile ──
            if e.speed_profile:
                total = sum(entry.duration for entry in e.speed_profile)
                self._w(f"    # Speed profile: {len(e.speed_profile)} steps,"
                        f" total {total:.0f} s")
                self._w(f"    for target_speed, duration in [")
                for entry in e.speed_profile:
                    self._w(f"        ({entry.speed:.4f}, {entry.duration:.1f}),  # m/s, s")
                self._w(f"    ]:")
                self._w(f"        take SetSpeedAction(target_speed)")
                self._w(f"        wait for duration seconds")

    def _termination(self):
        self._section("Termination")
        dur = int(self.s.duration_sec)
        self._w(f"terminate after {dur} seconds")


# ─────────────────────────────────────────────────────────────────
# STEP 5: Small utilities
# ─────────────────────────────────────────────────────────────────

def safe_name(name: str) -> str:
    """Convert any string to a valid Python identifier."""
    result = name.replace(" ", "_").replace("-", "_").replace(".", "_")
    return "e_" + result if result and result[0].isdigit() else result


def deduplicate(positions: List[Position], tol: float = 0.05) -> List[Position]:
    """Remove consecutive duplicate positions (same x,y within tolerance)."""
    out = [positions[0]]
    for p in positions[1:]:
        prev = out[-1]
        if abs(p.x - prev.x) > tol or abs(p.y - prev.y) > tol:
            out.append(p)
    return out


# ─────────────────────────────────────────────────────────────────
# STEP 6: Main entry point
# ─────────────────────────────────────────────────────────────────

def convert(input_path: str, output_path: str = None) -> str:
    """Parse `input_path` and write a .scenic file. Returns output path."""

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".scenic"

    # ── parse ──
    print(f"Parsing : {input_path}")
    scenario = XOSCParser(input_path).parse()

    # ── summary ──
    ego = next((e for e in scenario.entities if e.is_ego), None)
    print(f"Map     : {scenario.map_name or '(not found)'}")
    print(f"Ego     : {ego.name if ego else 'none detected'}")
    print(f"Entities: {len(scenario.entities)}")
    for e in scenario.entities:
        wps = len(deduplicate(e.waypoints)) if e.waypoints else 0
        sp  = len(e.speed_profile)
        tag = "[EGO]" if e.is_ego else "     "
        print(f"  {tag} {e.name:<18} {e.scenic_class:<12} "
              f"v0={e.init_speed:.2f} m/s  waypoints={wps}  speed_steps={sp}")
    print(f"Weather : {scenario.weather.precipitation}  fog={scenario.weather.fog_range:.0f}m")
    print(f"Duration: {scenario.duration_sec:.0f}s")

    # ── generate ──
    code = ScenicWriter(scenario, source_file=input_path).write()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"Written : {output_path}")
    return output_path


def main():
    ap = argparse.ArgumentParser(
        description="Convert OpenSCENARIO (.xosc) → Scenic 3 / MetaDrive (.scenic)",
        epilog="Example: python xosc_to_scenic.py my_scenario.xosc"
    )
    ap.add_argument("input",  help="Path to the .xosc file")
    ap.add_argument("output", nargs="?", help="Output .scenic path (optional)")
    args = ap.parse_args()

    try:
        out = convert(args.input, args.output)
        print(f"\nDone → {out}")
    except FileNotFoundError as e:
        print(f"Error: {e}"); sys.exit(1)
    except ET.ParseError as e:
        print(f"XML error: {e}"); sys.exit(2)


if __name__ == "__main__":
    main()