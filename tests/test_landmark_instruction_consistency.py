#!/usr/bin/env python3
"""Tests for landmark instruction consistency with path geometry.

Validates that instruction text (turn direction, turn angle, distances)
matches the actual waypoint geometry for 3-waypoint Landmark/LandmarkSet episodes.

Usage:
    pytest tests/test_landmark_instruction_consistency.py -v
    pytest tests/test_landmark_instruction_consistency.py -v --json-path /path/to/episodes.json
"""

import json
import math
import re
from pathlib import Path
from typing import List, Tuple, Optional

import pytest


# ---------------------------------------------------------------------------
# Configurable thresholds
# ---------------------------------------------------------------------------
DISTANCE_ERROR_THRESHOLD_PCT = 30  # percent
TURN_ANGLE_ERROR_THRESHOLD_DEG = 30  # degrees


# ---------------------------------------------------------------------------
# Geo helpers
# ---------------------------------------------------------------------------
def haversine_distance(coord1: List[float], coord2: List[float]) -> float:
    """Return distance in meters between two [lon, lat, alt] points."""
    lon1, lat1 = math.radians(coord1[0]), math.radians(coord1[1])
    lon2, lat2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 6_371_000 * 2 * math.asin(math.sqrt(a))


def bearing(coord1: List[float], coord2: List[float]) -> float:
    """Return bearing in degrees [0, 360) from *coord1* to *coord2*."""
    lon1, lat1 = math.radians(coord1[0]), math.radians(coord1[1])
    lon2, lat2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    )
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def turn_angle(bearing_before: float, bearing_after: float) -> float:
    """Signed turn angle.  Positive → right, negative → left."""
    d = (bearing_after - bearing_before + 360) % 360
    return d - 360 if d > 180 else d


# ---------------------------------------------------------------------------
# Instruction extraction helpers
# ---------------------------------------------------------------------------
# Matches patterns like:
#   "90-degree right", "120° left", "60 degree left", "75° right"
_TURN_RE = re.compile(
    r"(\d+)\s*[°\-]?\s*(?:degree)?\s*(left|right)", re.IGNORECASE
)
# Also matches "75 degree turn right" where 'turn' sits between degree and direction
_TURN_RE2 = re.compile(
    r"(\d+)\s*[°\-]?\s*(?:degree)?\s*turn\s*(left|right)", re.IGNORECASE
)
# Distance pattern: "140 meters", "220 meter"
_DIST_RE = re.compile(r"(\d+)\s*meters?", re.IGNORECASE)


def extract_turn_patterns(text: str) -> List[Tuple[int, str]]:
    """Return list of (degrees, 'left'|'right') found in *text*."""
    matches = _TURN_RE.findall(text)
    for m in _TURN_RE2.findall(text):
        if m not in matches:
            matches.append(m)
    return [(int(deg), direction.lower()) for deg, direction in matches]


def extract_distances(text: str) -> List[int]:
    """Return list of distances (meters) mentioned in *text*."""
    return [int(d) for d in _DIST_RE.findall(text)]


# ---------------------------------------------------------------------------
# Episode-level checks
# ---------------------------------------------------------------------------
class InstructionIssue:
    """One detected inconsistency between instruction and geometry."""

    def __init__(self, category: str, message: str):
        self.category = category  # 'turn_direction' | 'turn_angle' | 'distance'
        self.message = message

    def __repr__(self):
        return f"[{self.category}] {self.message}"


def check_episode(episode: dict) -> List[InstructionIssue]:
    """Check a single 3-waypoint Landmark/LandmarkSet episode.

    Returns a (possibly empty) list of issues found.
    """
    waypoints = episode["waypoints"]
    instruction_data = episode["instruction"]
    instruction_text = (
        instruction_data["instruction_text"]
        if isinstance(instruction_data, dict)
        else instruction_data
    )

    # ---------- geometry ----------
    seg_distances = [
        haversine_distance(waypoints[i], waypoints[i + 1])
        for i in range(len(waypoints) - 1)
    ]
    bearings = [
        bearing(waypoints[i], waypoints[i + 1])
        for i in range(len(waypoints) - 1)
    ]
    # For 3 waypoints there is exactly 1 turn (between segment 0 and segment 1)
    actual_turn = turn_angle(bearings[0], bearings[1])
    actual_dir = "right" if actual_turn > 0 else "left"
    actual_deg = abs(actual_turn)

    issues: List[InstructionIssue] = []

    # ---------- 1. Turn direction & angle ----------
    turn_patterns = extract_turn_patterns(instruction_text)
    for idx, (instr_deg, instr_dir) in enumerate(turn_patterns):
        if idx >= 1:
            # 3-waypoint → only 1 turn; skip extra matches
            break
        if instr_dir != actual_dir:
            issues.append(
                InstructionIssue(
                    "turn_direction",
                    f"instruction says '{instr_dir} {instr_deg}°' "
                    f"but actual turn is '{actual_dir} {actual_deg:.1f}°'",
                )
            )
        else:
            deg_error = abs(instr_deg - actual_deg)
            if deg_error > TURN_ANGLE_ERROR_THRESHOLD_DEG:
                issues.append(
                    InstructionIssue(
                        "turn_angle",
                        f"instruction says '{instr_dir} {instr_deg}°' "
                        f"but actual is '{actual_dir} {actual_deg:.1f}°' "
                        f"(error {deg_error:.1f}°)",
                    )
                )

    # ---------- 2. Segment distances ----------
    instr_distances = extract_distances(instruction_text)
    for i, instr_m in enumerate(instr_distances):
        if i >= len(seg_distances):
            break
        actual_m = seg_distances[i]
        error_pct = abs(instr_m - actual_m) / max(actual_m, 1) * 100
        if error_pct > DISTANCE_ERROR_THRESHOLD_PCT:
            issues.append(
                InstructionIssue(
                    "distance",
                    f"segment {i}: instruction says {instr_m}m "
                    f"but actual is {actual_m:.1f}m "
                    f"(error {error_pct:.1f}%)",
                )
            )

    return issues


# ---------------------------------------------------------------------------
# Fixtures & parametrization
# ---------------------------------------------------------------------------
_LANDMARK_TYPES = ("Landmark", "LandmarkSet")


def _load_episodes(json_path: str) -> List[dict]:
    """Load episodes from a JSON file and filter for 3-waypoint Landmark(Set)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    episodes = data.get("episodes", [])
    filtered = [
        ep
        for ep in episodes
        if ep.get("trajectory_type", "") in _LANDMARK_TYPES
        and len(ep.get("waypoints", [])) == 3
    ]
    return filtered


def _episode_id_label(ep: dict) -> str:
    """Human-readable label for parametrize ids."""
    eid = ep.get("episode_id", "?")
    traj = ep.get("aux_info", {}).get("trajectory_id", "")
    if not traj:
        traj = ep.get("aux_info", {}).get("path_id", "")
    return f"ep{eid}_traj{traj}"


# Default test file shipped with the repo
_DEFAULT_TEST_FILE = (
    Path(__file__).parent
    / "test_data"
    / "satnav_landmark_direction_inconsistent.json"
)


def _resolve_json_path() -> str:
    """Return JSON path from env var ``LANDMARK_JSON`` or fall back to default."""
    import os

    env_path = os.environ.get("LANDMARK_JSON")
    if env_path:
        return env_path
    if _DEFAULT_TEST_FILE.exists():
        return str(_DEFAULT_TEST_FILE)
    pytest.skip(
        "Set LANDMARK_JSON env var or place default test file at "
        f"{_DEFAULT_TEST_FILE}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestLandmarkInstructionConsistency:
    """Verify instruction text matches path geometry for 3-wp Landmark episodes."""

    # ---- summary test: report overall stats ----
    def test_summary(self):
        """Print an overall summary of instruction consistency."""
        json_path = _resolve_json_path()
        episodes = _load_episodes(json_path)
        if not episodes:
            pytest.skip("No 3-waypoint Landmark/LandmarkSet episodes found")

        total = len(episodes)
        bad_direction = 0
        bad_angle = 0
        bad_distance = 0
        total_bad = 0

        for ep in episodes:
            issues = check_episode(ep)
            if issues:
                total_bad += 1
            for iss in issues:
                if iss.category == "turn_direction":
                    bad_direction += 1
                elif iss.category == "turn_angle":
                    bad_angle += 1
                elif iss.category == "distance":
                    bad_distance += 1

        print(f"\n{'='*60}")
        print(f"Instruction consistency summary for: {json_path}")
        print(f"{'='*60}")
        print(f"  Total 3-wp Landmark episodes : {total}")
        print(f"  Episodes with issues          : {total_bad} ({total_bad/total*100:.1f}%)")
        print(f"  Turn direction errors          : {bad_direction}")
        print(f"  Turn angle errors (>{TURN_ANGLE_ERROR_THRESHOLD_DEG}°) : {bad_angle}")
        print(f"  Distance errors (>{DISTANCE_ERROR_THRESHOLD_PCT}%)     : {bad_distance}")
        print(f"{'='*60}")

        # This test always passes – it's just a report
        assert True

    # ---- per-episode parametrized test ----
    def test_turn_direction(self):
        """Every episode's instruction turn direction must match the path."""
        json_path = _resolve_json_path()
        episodes = _load_episodes(json_path)

        failures = []
        for ep in episodes:
            issues = [i for i in check_episode(ep) if i.category == "turn_direction"]
            if issues:
                label = _episode_id_label(ep)
                failures.append(f"  {label}: {issues[0].message}")

        if failures:
            msg = (
                f"{len(failures)} episode(s) have turn direction errors:\n"
                + "\n".join(failures)
            )
            pytest.fail(msg)

    def test_turn_angle(self):
        """Every episode's instruction turn angle must be within threshold."""
        json_path = _resolve_json_path()
        episodes = _load_episodes(json_path)

        failures = []
        for ep in episodes:
            issues = [i for i in check_episode(ep) if i.category == "turn_angle"]
            if issues:
                label = _episode_id_label(ep)
                failures.append(f"  {label}: {issues[0].message}")

        if failures:
            msg = (
                f"{len(failures)} episode(s) have turn angle errors "
                f"(threshold {TURN_ANGLE_ERROR_THRESHOLD_DEG}°):\n"
                + "\n".join(failures)
            )
            pytest.fail(msg)

    def test_segment_distances(self):
        """Every episode's instruction distances must match actual path."""
        json_path = _resolve_json_path()
        episodes = _load_episodes(json_path)

        failures = []
        for ep in episodes:
            issues = [i for i in check_episode(ep) if i.category == "distance"]
            if issues:
                label = _episode_id_label(ep)
                for iss in issues:
                    failures.append(f"  {label}: {iss.message}")

        if failures:
            msg = (
                f"{len(failures)} distance error(s) found "
                f"(threshold {DISTANCE_ERROR_THRESHOLD_PCT}%):\n"
                + "\n".join(failures)
            )
            pytest.fail(msg)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Check landmark instruction consistency with path geometry"
    )
    parser.add_argument(
        "json_path",
        help="Path to episodes JSON file",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=DISTANCE_ERROR_THRESHOLD_PCT,
        help=f"Distance error threshold in %% (default {DISTANCE_ERROR_THRESHOLD_PCT})",
    )
    parser.add_argument(
        "--angle-threshold",
        type=float,
        default=TURN_ANGLE_ERROR_THRESHOLD_DEG,
        help=f"Turn angle error threshold in degrees (default {TURN_ANGLE_ERROR_THRESHOLD_DEG})",
    )
    args = parser.parse_args()

    # Override globals with CLI args
    DISTANCE_ERROR_THRESHOLD_PCT = args.distance_threshold
    TURN_ANGLE_ERROR_THRESHOLD_DEG = args.angle_threshold

    json_path = args.json_path
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_episodes = data.get("episodes", [])
    landmark_types = ("Landmark", "LandmarkSet")
    episodes = [
        ep
        for ep in all_episodes
        if ep.get("trajectory_type", "") in landmark_types
        and len(ep.get("waypoints", [])) == 3
    ]

    print(f"Loaded {len(all_episodes)} total episodes, "
          f"{len(episodes)} are 3-waypoint Landmark/LandmarkSet")
    print(f"Thresholds: distance > {DISTANCE_ERROR_THRESHOLD_PCT}%, "
          f"angle > {TURN_ANGLE_ERROR_THRESHOLD_DEG}°\n")

    dir_errors = []
    angle_errors = []
    dist_errors = []

    for ep in episodes:
        issues = check_episode(ep)
        label = _episode_id_label(ep)
        for iss in issues:
            if iss.category == "turn_direction":
                dir_errors.append((label, iss))
            elif iss.category == "turn_angle":
                angle_errors.append((label, iss))
            elif iss.category == "distance":
                dist_errors.append((label, iss))

    total_bad = len({
        label for label, _ in dir_errors + angle_errors + dist_errors
    })

    # --- Report ---
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Total checked       : {len(episodes)}")
    print(f"  Episodes with issues: {total_bad} ({total_bad/max(len(episodes),1)*100:.1f}%)")
    print(f"  Turn direction errors: {len(dir_errors)}")
    print(f"  Turn angle errors    : {len(angle_errors)}")
    print(f"  Distance errors      : {len(dist_errors)}")

    if dir_errors:
        print(f"\n--- Turn Direction Errors ({len(dir_errors)}) ---")
        for label, iss in dir_errors:
            print(f"  {label}: {iss.message}")

    if angle_errors:
        print(f"\n--- Turn Angle Errors ({len(angle_errors)}) ---")
        for label, iss in angle_errors:
            print(f"  {label}: {iss.message}")

    if dist_errors:
        print(f"\n--- Distance Errors ({len(dist_errors)}) ---")
        for label, iss in dist_errors:
            print(f"  {label}: {iss.message}")

    if dir_errors or angle_errors or dist_errors:
        print(f"\nFAILED: {total_bad} episode(s) with issues")
        sys.exit(1)
    else:
        print("\nPASSED: All episodes are consistent")
        sys.exit(0)
