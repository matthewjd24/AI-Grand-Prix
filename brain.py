"""
brain.py — top-level orchestrator.

Wires together: vision (perception) → world_building (state) → route_planner
(decision) → flight_control (action). Also keeps heartbeats alive and pumps
the debug Unity stream.
"""

import asyncio
import socket
import struct
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "CNN Gate Detection")
import log_dir   # noqa: F401  (import early so the run folder is created first)
_t_import = time.time()
print("[brain] loading YOLO gate-detector model...")
import predict   # noqa: F401  (loaded so vision can find it)
print(f"[brain] model loaded in {time.time()-_t_import:.2f}s")
import vision
import world_building
import flight_control
import route_planner

# --- Config ------------------------------------------------------------------
UNITY_HOST = "127.0.0.1"
UNITY_PORT = 9000

HEARTBEAT_HZ = 1
COMMAND_HZ   = 10

# UDP pose protocol
TOPIC_POSE     = 1   # per-gate pose:  object_id (u8) | px py pz (3xf32) | qx qy qz qw (4xf32)
TOPIC_COUNT    = 2   # gate count:     count (u8)
TOPIC_ATTITUDE = 3   # drone attitude: qx qy qz qw (4xf32, Unity-frame quaternion)
_POSE_FORMAT     = "<B3f4f"   # 29 bytes
_ATTITUDE_FORMAT = "<4f"      # 16 bytes


# --- Unity comms -------------------------------------------------------------
_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send_unity_heartbeat():
    _udp.sendto(b"heartbeat", (UNITY_HOST, UNITY_PORT))


def send_pose(object_id, px, py, pz, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
    payload = struct.pack(_POSE_FORMAT, object_id, px, py, pz, qx, qy, qz, qw)
    _udp.sendto(bytes([TOPIC_POSE]) + payload, (UNITY_HOST, UNITY_PORT))


def send_gate_count(count):
    _udp.sendto(bytes([TOPIC_COUNT, count]), (UNITY_HOST, UNITY_PORT))


def send_gate_poses_to_unity():
    tracks = world_building.get_smoothed_tracks()
    send_gate_count(len(tracks))
    for i, track in enumerate(tracks):
        (tx, ty, tz), (qx, qy, qz, qw) = _ned_to_unity(track["tvec"], track["rvec"])
        send_pose(i, tx, ty, tz, qx, qy, qz, qw)


def send_attitude_to_unity(roll, pitch, yaw):
    R_ned = _R_ned_from_ypr(roll, pitch, yaw)
    qx, qy, qz, qw = _R_ned_to_unity_quat(R_ned)
    payload = struct.pack(_ATTITUDE_FORMAT, qx, qy, qz, qw)
    _udp.sendto(bytes([TOPIC_ATTITUDE]) + payload, (UNITY_HOST, UNITY_PORT))


# --- World transform: NED world frame → Unity coordinates --------------------
# NED world (right-handed): X-North, Y-East, Z-Down.
# Unity (left-handed):      X-right, Y-up,  Z-forward.
# Mapping: Unity X = NED Y (east → right),  Unity Y = -NED Z (down → up),
#          Unity Z = NED X (north → forward).
_P_UNITY_FROM_NED = np.array([
    [0, 1,  0],
    [0, 0, -1],
    [1, 0,  0],
], dtype=np.float64)

# Our PnP gate frame has +Z as the normal (out of the face), but the Unity
# gate prefab uses +X as its forward axis. Rotate +90° about gate-Y so the
# prefab's front lines up with the detected gate's normal.
_R_PREFAB_FIX = np.array([
    [ 0, 0, 1],
    [ 0, 1, 0],
    [-1, 0, 0],
], dtype=np.float64)


def _ned_to_unity(tvec, rvec):
    t_unity  = _P_UNITY_FROM_NED @ np.asarray(tvec).flatten()
    R_ned, _ = cv2.Rodrigues(rvec)
    R_ned_prefab = R_ned @ _R_PREFAB_FIX
    q = _R_ned_to_unity_quat(R_ned_prefab)
    tx, ty, tz = t_unity
    return (tx, ty, tz), q


def _R_ned_from_ypr(roll, pitch, yaw):
    """Aerospace ZYX intrinsic — body-NED → world-NED rotation matrix."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr           ],
    ], dtype=np.float64)


def _R_ned_to_unity_quat(R_ned):
    """Basis-change an NED rotation matrix into Unity coords and return (qx,qy,qz,qw)."""
    R = _P_UNITY_FROM_NED @ R_ned @ _P_UNITY_FROM_NED.T
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s, 0.25/s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s, (R[2,1]-R[1,2])/s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s, (R[0,2]-R[2,0])/s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s, (R[1,0]-R[0,1])/s


# --- Main loop ---------------------------------------------------------------
async def main():
    def step(msg):
        print(f"[brain] t={log_dir.elapsed():6.2f}s  {msg}")

    step("connecting to flight controller (MAVSDK)...")
    await flight_control.connect()
    step("starting flight control telemetry...")
    await flight_control.start()
    step("warming up YOLO model (load + first CUDA inference)...")
    predict.warmup()
    step("starting vision UDP listener...")
    vision.start()
    step("starting world_building thread...")
    world_building.start()
    step("creating cv2 window...")
    cv2.namedWindow("vision", cv2.WINDOW_NORMAL)

    # Block until we've actually received and processed a frame from the sim.
    # Otherwise the command loop starts firing position targets while the
    # drone is "flying blind" with no vision data.
    step("waiting for first camera frame from sim...")
    wait_start = time.time()
    while True:
        _, frame_id = vision.get_detections()
        if frame_id > 0:
            break
        if time.time() - wait_start > 30.0:
            print("[brain] WARN: no frame after 30s — sim might not be running. Proceeding anyway.")
            break
        await asyncio.sleep(0.05)
    step(f"first frame received (frame_id={frame_id})")

    # Prime offboard mode here, not inside the main loop. The first
    # send_position() triggers MAVSDK's offboard.start() handshake which can
    # take several seconds (often times out the first attempt). Doing it now
    # means the main loop never has to pay that cost on the critical path —
    # otherwise the loop blocks here and video_writer init / vision display
    # are delayed by the same amount.
    step("priming offboard mode (first send_position can take a moment)...")
    await flight_control.send_position(0.0, 0.0, 0.0)
    step("offboard primed — entering command loop.")

    print(f"[brain] Mode: {route_planner.MODE}")

    t0                  = log_dir.START_TIME
    last_heartbeat      = 0.0
    last_command        = 0.0
    last_attitude_print = 0.0
    video_writer        = None   # lazily created once we see the first frame

    try:
        while True:
            now     = time.time()
            elapsed = now - t0

            # Heartbeat
            if now - last_heartbeat >= 1.0 / HEARTBEAT_HZ:
                send_unity_heartbeat()
                last_heartbeat = now

            # Print attitude and send to unity
            if now - last_attitude_print >= 0.1:
                att = flight_control.get_attitude()
                if att is not None:
                    r, p, y, *_ = att
                    send_attitude_to_unity(r, p, y)
                    #print(f"[brain] attitude: roll={np.degrees(r):+6.1f}°  pitch={np.degrees(p):+6.1f}°  yaw={np.degrees(y):+6.1f}°")
                # else:
                #     print("[brain] attitude: <none yet>")
                last_attitude_print = now

            # Send commands to the drone
            if now - last_command >= 1.0 / COMMAND_HZ:
                x, y, z, yaw = route_planner.current_target(elapsed)
                if yaw is None:
                    await flight_control.send_position(x, y, z)
                else:
                    await flight_control.send_position_yaw(x, y, z, yaw)
                last_command = now

            send_gate_poses_to_unity()
            raw_frame, _, annotated_frame = vision.get_latest()
            if annotated_frame is not None:
                frame = annotated_frame
            else:
                frame = raw_frame
            if frame is not None:
                frame = frame.copy()
                cv2.putText(frame, f"t={elapsed:6.2f}s", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                if video_writer is None:
                    h, w = frame.shape[:2]
                    video_path = log_dir.log_path("vision.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
                    print(f"[brain] recording video to {video_path}")
                video_writer.write(frame)
                cv2.imshow("vision", frame)
            cv2.waitKey(1)

            await asyncio.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[brain] Stopped.")
    finally:
        if video_writer is not None:
            video_writer.release()
            print("[brain] video saved.")


asyncio.run(main())
