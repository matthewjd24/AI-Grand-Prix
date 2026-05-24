"""
route_planner.py — decides where the drone should go.

Exposes current_target(t_seconds) which returns (x, y, z, yaw_or_None) in
MAVLink NED frame. yaw is None when the active mode doesn't care.
"""

import numpy as np

# --- Active mode -------------------------------------------------------------
MODE = "spin"   # "spin" | "circle" | "waypoints"

def set_mode(mode):
    global MODE
    MODE = mode

def _spin(t):
    SPIN_PERIOD   = 10.0    # seconds per full rotation
    yaw = 2 * np.pi * (t / SPIN_PERIOD)
    return 0.0, 0.0, -1.0, yaw

# --- Waypoints ---------------------------------------------------------------
WAYPOINTS = [
    ( 0.0,  0.0, -1.0),
    ( 0.0,  0.0, -1.0),
    ( 1.0,  0.0, -1.0),
    ( 1.0,  1.0, -1.0),
    ( 0.0,  1.0, -1.0),
    ( 0.0,  0.0, -1.0),
    ( 0.0,  0.0, -2.0),
    ( 2.0,  0.0, -2.0),
]
SECONDS_PER_WAYPOINT = 3.5
LOOP_WAYPOINTS       = True


def _waypoints(t):
    total_index = t / SECONDS_PER_WAYPOINT
    if LOOP_WAYPOINTS:
        i      = int(total_index) % len(WAYPOINTS)
        next_i = (i + 1) % len(WAYPOINTS)
    else:
        i      = min(int(total_index), len(WAYPOINTS) - 1)
        next_i = min(i + 1, len(WAYPOINTS) - 1)
    alpha = total_index - int(total_index)
    a, b  = WAYPOINTS[i], WAYPOINTS[next_i]
    return a[0] + (b[0]-a[0])*alpha, a[1] + (b[1]-a[1])*alpha, a[2] + (b[2]-a[2])*alpha, None


# --- Dispatch ----------------------------------------------------------------
_MODES = {
    "spin":      _spin,
    "waypoints": _waypoints,
}


def current_target(t_seconds):
    """Returns (x, y, z, yaw_or_None) in NED for the active mode."""
    return _MODES.get(MODE, _spin)(t_seconds)
