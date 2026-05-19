"""
brain.py — sends a pre-written sequence of position targets to the SITL bridge.

Talks to bridge.py over UDP/MAVLink on port 14550.
Sends a heartbeat at 2 Hz and the current waypoint at 10 Hz.

Edit WAYPOINTS to change the flight path. Each entry is (x, y, z) in MAVLink NED
frame — z is DOWN, so altitude of 1m above ground = z of -1.0.
"""

import time
from pymavlink import mavutil

BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = 14550

HEARTBEAT_HZ = 2
COMMAND_HZ   = 10

SECONDS_PER_WAYPOINT = 3.5
LOOP_WAYPOINTS = True     # if False, stops at the last waypoint instead of cycling


# --- Pre-written flight path (NED frame: z negative = up) -------------------
WAYPOINTS = [
    ( 0.0,  0.0, -1.0),    # takeoff to 1m at origin
    ( 0.0,  0.0, -1.0),    # climb to 1m at origin
    ( 1.0,  0.0, -1.0),    # 1m north
    ( 1.0,  1.0, -1.0),    # 1m north, 1m east
    ( 0.0,  1.0, -1.0),    # 1m east
    ( 0.0,  0.0, -1.0),    # back to origin
    ( 0.0,  0.0, -2.0),    # climb to 2m
    ( 2.0,  0.0, -2.0),    # 2m north at 2m altitude
]


# --- Connect to the bridge ---------------------------------------------------
print(f"[brain] Connecting to bridge at {BRIDGE_HOST}:{BRIDGE_PORT}...")
conn = mavutil.mavlink_connection(f"udpout:{BRIDGE_HOST}:{BRIDGE_PORT}")


# --- MAVLink message helpers -------------------------------------------------
def send_heartbeat():
    conn.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_QUADROTOR,
        mavutil.mavlink.MAV_AUTOPILOT_GENERIC,
        0, 0, 0
    )


def send_position_target(x_ned, y_ned, z_ned):
    """Send a position target in MAVLink NED frame."""
    type_mask = 0b0000111111111000   # use position only

    conn.mav.set_position_target_local_ned_send(
        0,                                              # time_boot_ms
        1, 1,                                           # target system & component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        x_ned, y_ned, z_ned,
        0, 0, 0,                                        # velocity (ignored)
        0, 0, 0,                                        # acceleration (ignored)
        0, 0                                            # yaw, yaw_rate (ignored)
    )


# --- Current waypoint selection ---------------------------------------------
def current_waypoint(t_seconds):
    """Interpolate smoothly between waypoints."""
    segment_duration = SECONDS_PER_WAYPOINT
    total_index = t_seconds / segment_duration
    
    if LOOP_WAYPOINTS:
        i = int(total_index) % len(WAYPOINTS)
        next_i = (i + 1) % len(WAYPOINTS)
    else:
        i = min(int(total_index), len(WAYPOINTS) - 1)
        next_i = min(i + 1, len(WAYPOINTS) - 1)
    
    # Fraction through current segment (0.0 to 1.0)
    alpha = total_index - int(total_index)
    
    a = WAYPOINTS[i]
    b = WAYPOINTS[next_i]
    
    x = a[0] + (b[0] - a[0]) * alpha
    y = a[1] + (b[1] - a[1]) * alpha
    z = a[2] + (b[2] - a[2]) * alpha
    
    return i, (x, y, z)


# --- Main loop ---------------------------------------------------------------
print(f"[brain] Loaded {len(WAYPOINTS)} waypoints. "
      f"{'Looping.' if LOOP_WAYPOINTS else 'One pass only.'}")
print("[brain] Ctrl+C to stop.")

start = time.time()
last_heartbeat = 0.0
last_command   = 0.0
last_index     = -1

try:
    while True:
        now = time.time()
        elapsed = now - start

        if now - last_heartbeat >= 1.0 / HEARTBEAT_HZ:
            send_heartbeat()
            last_heartbeat = now

        if now - last_command >= 1.0 / COMMAND_HZ:
            index, (x, y, z) = current_waypoint(elapsed)
            send_position_target(x, y, z)
            last_command = now

            # Log only when the active waypoint changes
            if index != last_index:
                print(f"[brain] t={elapsed:5.1f}s  "
                      f"waypoint {index}/{len(WAYPOINTS)-1} -> "
                      f"({x:+.2f}, {y:+.2f}, {z:+.2f}) NED")
                last_index = index

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n[brain] Stopped.")