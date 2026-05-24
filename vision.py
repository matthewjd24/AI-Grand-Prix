"""
vision.py — receives the DCL chunked-JPEG stream, runs gate detection, and
exposes per-frame detection results.

Per-frame outputs are consumed by world_building.py (for track maintenance) and
brain.py (for display).
"""

import json
import socket
import struct
import threading
import time

import cv2
import numpy as np

import log_dir
import predict

_log_file   = log_dir.open_log()   # → vision.jsonl
_start_time = time.time()

VISION_HOST = "0.0.0.0"
VISION_PORT = 5600

_HEADER_FMT  = "<IHHIIQ"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAX_PENDING = 8

_latest_frame     = None
_latest_gates     = []
_latest_annotated = None
_frame_counter    = 0          # increments on every successful inference
_lock             = threading.Lock()
_raw_queue        = []         # raw decoded frames for display, drained by main thread

_log_file   = log_dir.open_log() 
_start_time = time.time()


def get_latest():
    """Return (frame, gates, annotated) for the most recently processed image."""
    with _lock:
        return _latest_frame, _latest_gates, _latest_annotated

def get_detections():
    """Return (gates, frame_id) — frame_id increments when a new inference completes.
    Consumers can compare frame_id against a previous value to skip stale data."""
    with _lock:
        return _latest_gates, _frame_counter

def pop_raw_frame():
    """Return the latest raw frame for display, or None. Drains the queue."""
    with _lock:
        if not _raw_queue:
            return None
        frame = _raw_queue[-1]
        _raw_queue.clear()
        return frame

def start():
    """Start the vision stream thread. Non-blocking."""
    t = threading.Thread(target=_thread, daemon=True, name="VisionStream")
    t.start()

def _parse_chunk(packet):
    if len(packet) < _HEADER_SIZE:
        return None
    frame_id, chunk_id, total_chunks, _, payload_size, _ = \
        struct.unpack(_HEADER_FMT, packet[:_HEADER_SIZE])
    return frame_id, chunk_id, total_chunks, packet[_HEADER_SIZE:_HEADER_SIZE + payload_size]

def _reassemble(pending, frame_id, chunk_id, total_chunks, chunk_data):
    entry = pending.setdefault(frame_id, {"total": total_chunks, "chunks": {}})
    entry["chunks"][chunk_id] = chunk_data
    if len(entry["chunks"]) < entry["total"]:
        return None
    jpeg = b"".join(entry["chunks"][i] for i in range(entry["total"]))
    del pending[frame_id]
    for fid in [f for f in pending if f < frame_id]:
        del pending[fid]
    while len(pending) > _MAX_PENDING:
        del pending[min(pending)]
    return jpeg

def _log_gate_corners(gates):
    """Write one JSON line per frame with each detected gate's pixel corners
    and pose (in camera frame). Corner order matches predict._GATE_PTS_3D
    (4 inner, optionally followed by 4 outer)."""
    gate_records = []
    for i, g in enumerate(gates):
        xy   = g["keypoints"]
        conf = g["keypoint_conf"]
        corners = []
        for k in range(len(xy)):
            corners.append({
                "px":   round(float(xy[k][0]), 2),
                "py":   round(float(xy[k][1]), 2),
                "conf": round(float(conf[k]), 3),
            })

        pose_record = None
        if g["pose"] is not None:
            pose_record = {
                "tvec":    [round(float(c), 4) for c in g["pose"]["tvec"].flatten()],
                "rvec":    [round(float(c), 4) for c in g["pose"]["rvec"].flatten()],
                "partial": bool(g["pose"].get("partial", False)),
            }

        gate_records.append({
            "id":      i,
            "corners": corners,
            "pose":    pose_record,
        })

    line = {
        "t":     round(time.time() - _start_time, 4),
        "frame": _frame_counter,
        "gates": gate_records,
    }
    _log_file.write(json.dumps(line) + "\n")

def _process_frame(jpeg):
    global _latest_frame, _latest_gates, _latest_annotated, _frame_counter
    t0 = time.perf_counter()
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return
    with _lock:
        _raw_queue.append(img)

    t1 = time.perf_counter()
    gates, annotated = predict.detect(img)
    t2 = time.perf_counter()

    for i, g in enumerate(gates):
        if g["pose"] is None:
            kind = "no pose"
        elif g["pose"].get("partial"):
            kind = "partial"
        else:
            kind = "pnp"
        # print(f"[vision] gate {i}: {kind}")

    t3 = time.perf_counter()
    if _frame_counter % 30 == 0:
        print(f"[vision] frame processing: {1000*(t3-t1):5.1f}ms. Inference: {1000*(t2-t1):5.1f}ms")

    _log_gate_corners(gates)

    with _lock:
        _latest_frame     = img
        _latest_gates     = gates
        _latest_annotated = annotated
        _frame_counter   += 1

def _thread():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    sock.bind((VISION_HOST, VISION_PORT))
    print(f"[vision] Listening on {VISION_HOST}:{VISION_PORT}")

    pending    = {}
    last_shown = -1

    last_sec = time.time()
    frames = 0
    while True:
        curr_sec = time.time()
        if curr_sec - last_sec > 1.0:
            last_sec = curr_sec
            print(f"[vision] {frames} fps")
            frames = 0

        packet, _ = sock.recvfrom(65535)
        parsed = _parse_chunk(packet)
        if parsed is None:
            continue

        frame_id, chunk_id, total_chunks, chunk_data = parsed
        if frame_id <= last_shown:
            continue

        jpeg = _reassemble(pending, frame_id, chunk_id, total_chunks, chunk_data)
        if jpeg is None:
            continue

        last_shown = frame_id
        _process_frame(jpeg)
        frames += 1

