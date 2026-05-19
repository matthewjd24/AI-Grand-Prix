"""
vision.py — receives the DCL chunked-JPEG stream, runs gate detection, and
exposes the latest frame and gate poses via get_latest().
"""

import socket
import struct
import threading
import time

import cv2
import numpy as np

import predict

VISION_HOST = "0.0.0.0"
VISION_PORT = 5600

_HEADER_FMT  = "<IHHIIQ"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAX_PENDING = 8

_latest_frame     = None
_latest_gates     = []
_latest_annotated = None
_lock             = threading.Lock()


def get_latest():
    """Return (frame, gates, annotated) for the most recently processed image."""
    with _lock:
        return _latest_frame, _latest_gates, _latest_annotated


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


def _process_frame(jpeg):
    global _latest_frame, _latest_gates, _latest_annotated
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return []
    t0 = time.perf_counter()
    gates, annotated = predict.detect(img)
    ms = (time.perf_counter() - t0) * 1000
    print(f"[vision] inference {ms:.1f}ms  {len(gates)} gate(s)")
    with _lock:
        _latest_frame     = img
        _latest_gates     = gates
        _latest_annotated = annotated
    return gates


def _log_nearest_gate(gates):
    best = min((g for g in gates if g["pose"]), key=lambda g: g["pose"]["distance"], default=None)
    if best:
        p = best["pose"]
        print(f"[vision] nearest gate: d={p['distance']:.2f}m  tvec={p['tvec'].flatten()}")


def _thread():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    sock.bind((VISION_HOST, VISION_PORT))
    print(f"[vision] Listening on {VISION_HOST}:{VISION_PORT}")

    pending    = {}
    last_shown = -1

    while True:
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
        gates = _process_frame(jpeg)
        if gates:
            _log_nearest_gate(gates)


def start():
    """Start the vision stream thread. Non-blocking."""
    t = threading.Thread(target=_thread, daemon=True, name="VisionStream")
    t.start()
