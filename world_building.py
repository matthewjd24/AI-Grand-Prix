"""
world_building.py — turns per-frame gate detections from vision.py into a
persistent list of tracked gates.

Each cycle:
  1. Get the newest gate detections from vision.py
  2. Convert each detection from camera coordinates into world coordinates
     (using the drone's current attitude)
  3. Match each detection to the nearest existing track
  4. Create new tracks for any detections that didn't match
  5. Drop old samples and tracks we haven't seen lately

Other modules read the results by calling get_smoothed_tracks().
"""

import json
import threading
import time
from collections import deque

import cv2
import numpy as np

import flight_control
import log_dir
import vision

# --- Tuning constants -------------------------------------------------------
UPDATE_HZ        = 30     # how many times per second we update the world
HISTORY_SECONDS  = 0.5    # samples older than this are dropped
MATCH_DISTANCE   = 2.5    # detections within this many metres = same gate
SMOOTH_LAMBDA    = 3      # higher = older samples count less in averages

# --- Camera → body rotation matrix (spec §3.8) ------------------------------
# The camera sits on the drone tilted 20° upward.
# OpenCV camera frame: X-right, Y-down, Z-forward.
# Body NED frame:      X-forward, Y-right, Z-down.
# This matrix rotates a vector from camera frame into body frame.
_TILT_RAD = np.radians(20)
_ct = np.cos(_TILT_RAD)
_st = np.sin(_TILT_RAD)
_R_BODY_FROM_CAM = np.array([
    [0,  _st,  _ct],
    [1,  0,    0  ],
    [0,  _ct, -_st],
], dtype=np.float64)

# --- Module state -----------------------------------------------------------
# Each track is a dict:
#   "id"        : int, unique per track
#   "history"   : deque of sample dicts (oldest first)
#   "last_seen" : timestamp of the most recent matching detection
#
# Each sample dict:
#   "t"       : timestamp the sample was added
#   "tvec"    : 3-element numpy array, gate position in world NED metres
#   "rvec"    : 3x1 numpy array, gate rotation as a Rodrigues vector
#   "partial" : True if pose came from <4 corners (rotation is unreliable)
_tracks             = []
_next_track_id      = 0
_last_seen_frame_id = -1
_lock               = threading.Lock()

# --- Logging setup ----------------------------------------------------------
# Two files per run:
#   world_building.jsonl — full track history per gate (existing)
#   world_state.jsonl    — current smoothed pose per gate, quaternion form
_log_file       = log_dir.open_log()                                           # → world_building.jsonl
_state_log_file = open(log_dir.log_path("world_state.jsonl"), "w", buffering=1)

# ============================================================================
# Public API
# ============================================================================
def start():
    """Launch the background thread that updates the world model."""
    t = threading.Thread(target=_thread, daemon=True, name="WorldBuilding")
    t.start()

def get_tracks():
    """Return a shallow copy of the current list of tracks (raw history)."""
    with _lock:
        copy = []
        for track in _tracks:
            copy.append({
                "id":        track["id"],
                "history":   list(track["history"]),
                "last_seen": track["last_seen"],
            })
        return copy

def get_smoothed_tracks():
    """Return tracks with smoothed positions and rotations.
    Tracks with only one sample are skipped — the first detection of a new
    gate is often a noisy outlier."""
    now = time.time()
    smoothed = []
    with _lock:
        for track in _tracks:
            if len(track["history"]) < 2:
                continue
            smoothed_position = _smooth_position(track["history"], now)
            smoothed_rotation = _smooth_rotation(track["history"], now)
            smoothed.append({
                "id":        track["id"],
                "tvec":      smoothed_position.reshape(3, 1),
                "rvec":      smoothed_rotation,
                "last_seen": track["last_seen"],
            })
    return smoothed

# ============================================================================
# Background update loop
# ============================================================================
def _thread():
    """Calls update_model() at UPDATE_HZ until the program ends."""
    period = 1.0 / UPDATE_HZ
    print(f"[world] update_model() running at {UPDATE_HZ} Hz")
    while True:
        t0 = time.time()
        update_model()
        elapsed = time.time() - t0
        time.sleep(max(0.0, period - elapsed))

def update_model():
    """One pass of the update pipeline."""
    global _last_seen_frame_id

    # Only do work if vision has produced a NEW frame since last time.
    gates, frame_id = vision.get_detections()
    if frame_id == _last_seen_frame_id:
        return
    _last_seen_frame_id = frame_id

    # We can only transform detections into world frame if we know the
    # drone's attitude. Skip the frame otherwise.
    attitude = flight_control.get_attitude()
    if attitude is None:
        return
    roll  = attitude[0]
    pitch = attitude[1]
    yaw   = attitude[2]

    # Convert every gate detection from camera frame into world frame.
    detections = _build_detections(gates, roll, pitch, yaw)

    # Fold them into the persistent track list.
    now = time.time()
    with _lock:
        _add_new_gate_positions_to_tracks(detections, now)
        _log_tracks(now)
        _log_current_state(now)

# ============================================================================
# Camera frame → world frame
# ============================================================================
def _build_detections(gates, roll, pitch, yaw):
    """Build detection dicts from raw gate detections, with each pose
    transformed into world NED coordinates."""
    R_world_from_camera = _R_world_from_body(roll, pitch, yaw) @ _R_BODY_FROM_CAM

    detections = []
    for gate in gates:
        if gate["pose"] is None:
            continue   # skip gates the detector couldn't get a pose for
        tvec_world, rvec_world = _apply_world_transform(
            gate["pose"]["tvec"],
            gate["pose"]["rvec"],
            R_world_from_camera,
        )
        detections.append({
            "tvec":    tvec_world,
            "rvec":    rvec_world,
            "partial": bool(gate["pose"].get("partial", False)),
        })
    return detections

def _R_world_from_body(roll, pitch, yaw):
    """Build the 3x3 rotation matrix that turns body-NED vectors into
    world-NED vectors. Standard aerospace ZYX (yaw, pitch, roll) convention."""
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    return np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,             cp*cr           ],
    ], dtype=np.float64)

def _apply_world_transform(tvec_cam, rvec_cam, R_world_from_camera):
    """Rotate one detection's (tvec, rvec) from camera frame into world frame."""
    # Position: just multiply by the rotation matrix.
    tvec_world = R_world_from_camera @ np.asarray(tvec_cam).flatten()

    # Rotation: convert rvec → matrix, compose, convert back.
    R_gate_in_camera, _ = cv2.Rodrigues(rvec_cam)
    R_gate_in_world     = R_world_from_camera @ R_gate_in_camera
    rvec_world, _       = cv2.Rodrigues(R_gate_in_world)
    return tvec_world, rvec_world

# ============================================================================
# Track management — match, create, prune, drop
# ============================================================================
def _add_new_gate_positions_to_tracks(detections, now):
    """Update the track list with this frame's detections."""
    matched_pairs, unmatched_indices = _match_new_gate_positions_to_tracks(detections)

    # Track all the IDs that got a detection this frame (new or existing)
    fresh_track_ids = set()

    # For each detection that matched an existing track, append
    for detection_index, track_index in matched_pairs:
        _append_sample(_tracks[track_index], detections[detection_index], now)
        fresh_track_ids.add(_tracks[track_index]["id"])

    # For each detection that didn't match anything, start a new track
    for detection_index in unmatched_indices:
        new_id = _create_track(detections[detection_index], now)
        fresh_track_ids.add(new_id)

    _prune_old_samples(now)
    _drop_unseen_tracks(fresh_track_ids)

def _match_new_gate_positions_to_tracks(detections):
    """Pair each detection with the closest existing track within
    MATCH_DISTANCE metres. Each track can match at most one detection.
    Returns (matched_pairs, unmatched_detection_indices)."""
    matched_pairs      = []
    claimed_tracks     = set()
    claimed_detections = set()

    for detection_index, detection in enumerate(detections):
        track_index = _find_closest_unclaimed_track(detection["tvec"], claimed_tracks)
        if track_index is None:
            continue
        matched_pairs.append((detection_index, track_index))
        claimed_tracks.add(track_index)
        claimed_detections.add(detection_index)

    unmatched_indices = []
    for i in range(len(detections)):
        if i not in claimed_detections:
            unmatched_indices.append(i)
    return matched_pairs, unmatched_indices

def _find_closest_unclaimed_track(tvec, claimed_track_indices):
    """Find the index of the closest unclaimed track. Returns None if no
    track is within MATCH_DISTANCE metres."""
    best_index    = None
    best_distance = MATCH_DISTANCE
    for i, track in enumerate(_tracks):
        if i in claimed_track_indices:
            continue
        last_known_tvec = track["history"][-1]["tvec"]
        distance        = float(np.linalg.norm(tvec - last_known_tvec))
        if distance < best_distance:
            best_index    = i
            best_distance = distance
    return best_index

def _append_sample(track, detection, now):
    """Add a new sample built from `detection` onto an existing track."""
    track["history"].append({
        "t":       now,
        "tvec":    detection["tvec"],
        "rvec":    detection["rvec"],
        "partial": detection["partial"],
    })
    track["last_seen"] = now

def _create_track(detection, now):
    """Start a new track from a detection that didn't match anything.
    Returns the new track's id."""
    global _next_track_id
    track_id = _next_track_id
    _next_track_id += 1

    first_sample = {
        "t":       now,
        "tvec":    detection["tvec"],
        "rvec":    detection["rvec"],
        "partial": detection["partial"],
    }
    _tracks.append({
        "id":        track_id,
        "history":   deque([first_sample]),
        "last_seen": now,
    })
    return track_id

def _prune_old_samples(now):
    """Drop samples older than HISTORY_SECONDS from every track."""
    cutoff = now - HISTORY_SECONDS
    for track in _tracks:
        while track["history"] and track["history"][0]["t"] < cutoff:
            track["history"].popleft()

def _drop_unseen_tracks(fresh_track_ids):
    """Remove tracks that didn't get a fresh detection this frame.
    If a gate isn't being detected anymore, it's gone from the world model."""
    surviving = []
    for track in _tracks:
        if track["id"] in fresh_track_ids and track["history"]:
            surviving.append(track)
    _tracks[:] = surviving   # in-place replace so other code keeps the same list

# ============================================================================
# Smoothing — weighted averages over each track's history
# ============================================================================
def _smooth_position(history, now):
    """Weighted average of the position vectors. Recent samples weigh more.
    Partial poses are included (the position part is still useful)."""
    total_weight = 0.0
    weighted_sum = np.zeros(3)
    for sample in history:
        age    = now - sample["t"]
        weight = np.exp(-SMOOTH_LAMBDA * age)
        weighted_sum += weight * sample["tvec"]
        total_weight += weight
    return weighted_sum / total_weight

def _smooth_rotation(history, now):
    """Weighted average rotation, using only full-pose samples.
    Partial samples have rvec=0 (no real rotation info) — including them
    would pull the average toward the identity rotation."""
    # Keep only the samples with a real rotation estimate.
    full_samples = []
    for sample in history:
        if not sample["partial"]:
            full_samples.append(sample)
    if not full_samples:
        return history[-1]["rvec"]   # fall back to the most recent rvec

    # Convert every rvec to a quaternion so we can average them safely.
    quats = []
    for sample in full_samples:
        quats.append(_rvec_to_quat(sample["rvec"]))

    # Hemisphere fix: q and -q describe the same rotation, but averaging
    # them naively cancels out. Flip any quaternion that points away from
    # the reference so they all sit in the same hemisphere.
    reference = quats[-1]
    for i in range(len(quats)):
        if np.dot(quats[i], reference) < 0:
            quats[i] = -quats[i]

    # Weighted average of the (hemisphere-aligned) quaternions.
    total_weight = 0.0
    weighted_sum = np.zeros(4)
    for i, sample in enumerate(full_samples):
        age    = now - sample["t"]
        weight = np.exp(-SMOOTH_LAMBDA * age)
        weighted_sum += weight * quats[i]
        total_weight += weight
    quat_avg = weighted_sum / total_weight

    # Re-normalize: averaging breaks the unit-length property slightly.
    quat_avg = quat_avg / np.linalg.norm(quat_avg)
    return _quat_to_rvec(quat_avg)

# ============================================================================
# Quaternion ↔ Rodrigues-vector helpers
# These convert between two ways of representing the same 3D rotation.
# The internal math is hairy — treat them as black boxes.
# ============================================================================
def _rvec_to_quat(rvec):
    """Rodrigues vector → quaternion (qw, qx, qy, qz)."""
    R, _ = cv2.Rodrigues(rvec)
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])

def _quat_to_rvec(q):
    """Quaternion (qw, qx, qy, qz) → Rodrigues vector."""
    qw, qx, qy, qz = q
    angle = 2.0 * np.arccos(np.clip(qw, -1.0, 1.0))
    s = np.sqrt(1.0 - qw*qw)
    if s < 1e-6:
        return np.zeros((3, 1))
    axis = np.array([qx, qy, qz]) / s
    return (axis * angle).reshape(3, 1)

# ============================================================================
# Logging — one JSON line per update
# ============================================================================
def _log_tracks(now):
    """Write the current track list as one line of JSON to the log file."""
    tracks_json = []
    for track in _tracks:
        history_json = []
        for sample in track["history"]:
            history_json.append({
                "t":       round(sample["t"] - log_dir.START_TIME, 4),
                "tvec":    [round(float(c), 4) for c in sample["tvec"]],
                "rvec":    [round(float(c), 4) for c in sample["rvec"].flatten()],
                "partial": sample["partial"],
            })
        tracks_json.append({
            "id":        track["id"],
            "last_seen": round(track["last_seen"] - log_dir.START_TIME, 4),
            "history":   history_json,
        })
    snapshot = {
        "t":      round(now - log_dir.START_TIME, 4),
        "tracks": tracks_json,
    }
    _log_file.write(json.dumps(snapshot, indent=2) + "\n")


def _log_current_state(now):
    """Write the current SMOOTHED pose per gate to world_state.jsonl.
    Position as Cartesian (NED metres) and rotation as aerospace ZYX Euler
    angles (roll, pitch, yaw, degrees). Only gates with enough history to
    smooth are included."""
    gates_json = []
    for track in _tracks:
        if len(track["history"]) < 2:
            continue
        smoothed_position = _smooth_position(track["history"], now)
        smoothed_rotation = _smooth_rotation(track["history"], now)
        roll_deg, pitch_deg, yaw_deg = _rvec_to_euler_deg(smoothed_rotation)
        gates_json.append({
            "id":       track["id"],
            "position": {
                "x": round(float(smoothed_position[0]), 4),
                "y": round(float(smoothed_position[1]), 4),
                "z": round(float(smoothed_position[2]), 4),
            },
            "euler_deg": {
                "roll":  round(roll_deg,  2),
                "pitch": round(pitch_deg, 2),
                "yaw":   round(yaw_deg,   2),
            },
        })
    snapshot = {
        "t":     round(now - log_dir.START_TIME, 4),
        "gates": gates_json,
    }
    _state_log_file.write(json.dumps(snapshot, indent=2) + "\n")


def _rvec_to_euler_deg(rvec):
    """Convert a Rodrigues rotation vector to aerospace ZYX Euler angles
    (roll, pitch, yaw) in degrees."""
    R, _ = cv2.Rodrigues(rvec)
    # Standard ZYX decomposition.
    pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    if abs(R[2, 0]) < 0.999999:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw  = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock — pitch near ±90°. Roll becomes ambiguous; conventionally set to 0.
        roll = 0.0
        yaw  = np.arctan2(-R[0, 1], R[1, 1])
    return float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))
