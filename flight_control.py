"""
flight_control.py — MAVSDK async client.

Same public API as the pymavlink version so brain.py changes are minimal:
  - connect() and start() must be awaited
  - send_position / send_position_yaw must be awaited
  - get_attitude() is synchronous (reads shared state updated by a background task)
  - send_heartbeat() is a no-op (MAVSDK handles heartbeats automatically)
"""

import asyncio
import math
import time
import threading

from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw

BRIDGE_HOST = "0.0.0.0"
BRIDGE_PORT = 14540

_drone            = System()
_latest_attitude  = None   # (roll_rad, pitch_rad, yaw_rad, 0, 0, 0, t_recv)
_state_lock       = threading.Lock()
_offboard_started = False


# --- Connection ---------------------------------------------------------------

async def connect(host=BRIDGE_HOST, port=BRIDGE_PORT):
    print(f"[flight_control] Connecting to {host}:{port}...")
    await _drone.connect(system_address=f"udpin://{host}:{port}")
    print("[flight_control] Waiting for heartbeat from sim...")
    async for state in _drone.core.connection_state():
        if state.is_connected:
            print("[flight_control] Connected.")
            break


async def start():
    asyncio.create_task(_attitude_loop())


# --- Heartbeat ----------------------------------------------------------------

def send_heartbeat():
    pass   # MAVSDK sends heartbeats automatically


# --- Telemetry ----------------------------------------------------------------

def get_attitude():
    """Return (roll, pitch, yaw, 0, 0, 0, t) in radians, or None."""
    with _state_lock:
        return _latest_attitude


async def _attitude_loop():
    global _latest_attitude
    async for att in _drone.telemetry.attitude_euler():
        with _state_lock:
            _latest_attitude = (
                math.radians(att.roll_deg),
                math.radians(att.pitch_deg),
                math.radians(att.yaw_deg),
                0.0, 0.0, 0.0,
                time.time(),
            )


# --- Commands -----------------------------------------------------------------

async def send_position(x_ned, y_ned, z_ned):
    await _send_setpoint(PositionNedYaw(x_ned, y_ned, z_ned, 0.0))


async def send_position_yaw(x_ned, y_ned, z_ned, yaw_rad):
    await _send_setpoint(PositionNedYaw(x_ned, y_ned, z_ned, math.degrees(yaw_rad)))


async def _send_setpoint(sp: PositionNedYaw):
    global _offboard_started
    if not _offboard_started:
        await _drone.offboard.set_position_ned(sp)
        try:
            await _drone.offboard.start()
            print("[flight_control] Offboard mode started.")
        except OffboardError as e:
            print(f"[flight_control] Offboard start: {e} (continuing)")
        _offboard_started = True
    await _drone.offboard.set_position_ned(sp)
