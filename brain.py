"""
brain.py — autonomous gate-racing controller.

Receives the vision stream from the sim, detects gates with the CNN,
and sends MAVLink position targets to the SITL bridge.
"""

import glob
import random
import socket
import struct
import sys
import time

import cv2
import numpy as np
from pymavlink import mavutil

sys.path.insert(0, "CNN Gate Detection")
import predict
import vision

# --- Config ------------------------------------------------------------------
BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT  = 14550

UNITY_HOST = "127.0.0.1"
UNITY_PORT = 9000

HEARTBEAT_HZ = 1
COMMAND_HZ   = 10

SECONDS_PER_WAYPOINT = 3.5
LOOP_WAYPOINTS = True

# UDP pose protocol (topic 1)
# payload: object_id (uint8) | px py pz (3×float32 LE) | qx qy qz qw (4×float32 LE)
TOPIC_POSE   = 1
_POSE_FORMAT = "<B3f4f"   # 29 bytes

# --- Pre-written flight path (NED frame: z negative = up) --------------------
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


# --- MAVLink connection -------------------------------------------------------
print(f"[brain] Connecting to bridge at {BRIDGE_HOST}:{BRIDGE_PORT}...")
conn = mavutil.mavlink_connection(f"udpout:{BRIDGE_HOST}:{BRIDGE_PORT}")

_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_unity_heartbeat():
    _udp.sendto(b"heartbeat", (UNITY_HOST, UNITY_PORT))

def send_pose(object_id, px, py, pz, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
    payload = struct.pack(_POSE_FORMAT, object_id, px, py, pz, qx, qy, qz, qw)
    _udp.sendto(bytes([TOPIC_POSE]) + payload, (UNITY_HOST, UNITY_PORT))
    print("Sent pose")


def _opencv_to_unity(tvec, rvec):
    """Convert pose from OpenCV camera frame to Unity coordinate frame.
    OpenCV: right-handed, x right, y down, z forward
    Unity:  left-handed, x right, y up,   z forward
    Conversion: negate y on position; apply reflection to rotation.
    """
    tx, ty, tz = tvec.flatten()
    # flip y
    t_unity = (tx, -ty, tz)

    R, _ = cv2.Rodrigues(rvec)
    M = np.diag([1, -1, 1]).astype(np.float64)
    R_unity = M @ R @ M
    # convert rotation matrix to quaternion
    trace = R_unity[0,0] + R_unity[1,1] + R_unity[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = (R_unity[2,1]-R_unity[1,2])*s, (R_unity[0,2]-R_unity[2,0])*s, (R_unity[1,0]-R_unity[0,1])*s, 0.25/s
    elif R_unity[0,0] > R_unity[1,1] and R_unity[0,0] > R_unity[2,2]:
        s = 2.0 * np.sqrt(1.0 + R_unity[0,0] - R_unity[1,1] - R_unity[2,2])
        q = 0.25*s, (R_unity[0,1]+R_unity[1,0])/s, (R_unity[0,2]+R_unity[2,0])/s, (R_unity[2,1]-R_unity[1,2])/s
    elif R_unity[1,1] > R_unity[2,2]:
        s = 2.0 * np.sqrt(1.0 + R_unity[1,1] - R_unity[0,0] - R_unity[2,2])
        q = (R_unity[0,1]+R_unity[1,0])/s, 0.25*s, (R_unity[1,2]+R_unity[2,1])/s, (R_unity[0,2]-R_unity[2,0])/s
    else:
        s = 2.0 * np.sqrt(1.0 + R_unity[2,2] - R_unity[0,0] - R_unity[1,1])
        q = (R_unity[0,2]+R_unity[2,0])/s, (R_unity[1,2]+R_unity[2,1])/s, 0.25*s, (R_unity[1,0]-R_unity[0,1])/s
    return t_unity, q


def send_gate_poses(gates):
    for i, gate in enumerate(gates):
        if gate["pose"] is None:
            continue
        (tx, ty, tz), (qx, qy, qz, qw) = _opencv_to_unity(gate["pose"]["tvec"], gate["pose"]["rvec"])
        send_pose(i, tx, ty, tz, qx, qy, qz, qw)

def send_mav_heartbeat():
    conn.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_QUADROTOR,
        mavutil.mavlink.MAV_AUTOPILOT_GENERIC,
        0, 0, 0
    )

def send_position_target(x_ned, y_ned, z_ned):
    type_mask = 0b0000111111111000
    conn.mav.set_position_target_local_ned_send(
        0, 1, 1,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        x_ned, y_ned, z_ned,
        0, 0, 0,
        0, 0, 0,
        0, 0
    )

def current_waypoint(t_seconds):
    total_index = t_seconds / SECONDS_PER_WAYPOINT
    if LOOP_WAYPOINTS:
        i      = int(total_index) % len(WAYPOINTS)
        next_i = (i + 1) % len(WAYPOINTS)
    else:
        i      = min(int(total_index), len(WAYPOINTS) - 1)
        next_i = min(i + 1, len(WAYPOINTS) - 1)
    alpha = total_index - int(total_index)
    a, b  = WAYPOINTS[i], WAYPOINTS[next_i]
    return i, tuple(a[k] + (b[k] - a[k]) * alpha for k in range(3))

def _smoke_test():
    val_images = [
        p for p in glob.glob(r"CNN Gate Detection\dataset\images\val\*.png")
        if open(p.replace("images", "labels").replace(".png", ".txt")).read().strip()
    ]
    if not val_images:
        print("[brain] smoke test: no val images found, skipping")
        return
    img = cv2.imread(random.choice(val_images))
    gates, annotated = predict.detect(img)
    print(f"[brain] smoke test: {len(gates)} gate(s) detected")
    for g in gates:
        print(f"        keypoints: {g['keypoints'].tolist()}")
        print(f"        conf:      {g['keypoint_conf'].tolist()}")
        if g["pose"]:
            R, _ = cv2.Rodrigues(g["pose"]["rvec"])
            print(f"        d={g['pose']['distance']:.2f}m  tvec={g['pose']['tvec'].flatten()}  R={R}")
    send_gate_poses(gates)
    cv2.imshow("vision", annotated)
    cv2.waitKey(1)


def main():
    _smoke_test()
    vision.start()

    print(f"[brain] {len(WAYPOINTS)} waypoints. {'Looping.' if LOOP_WAYPOINTS else 'One pass.'}")
    print("[brain] Ctrl+C to stop.")

    start          = time.time()
    last_heartbeat = 0.0
    last_command   = 0.0
    last_index     = -1

    while True:
        _, gates, annotated = vision.get_latest()
        send_gate_poses(gates)
        if annotated is not None:
            cv2.imshow("vision", annotated)
        cv2.waitKey(1)
        time.sleep(0.01)

    # try:
    #     while True:
    #         now     = time.time()
    #         elapsed = now - start

    #         if now - last_heartbeat >= 1.0 / HEARTBEAT_HZ:
    #             send_mav_heartbeat()
    #             send_unity_heartbeat()
    #             last_heartbeat = now

    #         if now - last_command >= 1.0 / COMMAND_HZ:
    #             index, (x, y, z) = current_waypoint(elapsed)
    #             send_position_target(x, y, z)
    #             last_command = now

    #             if index != last_index:
    #                 print(f"[brain] t={elapsed:5.1f}s  waypoint {index}/{len(WAYPOINTS)-1} "
    #                       f"-> ({x:+.2f}, {y:+.2f}, {z:+.2f}) NED")
    #                 last_index = index

    #         time.sleep(0.01)

    # except KeyboardInterrupt:
    #     print("\n[brain] Stopped.")

main()
