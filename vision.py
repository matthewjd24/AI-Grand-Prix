"""
vision.py — receives the DCL chunked-JPEG stream, runs gate detection, and
exposes per-frame detection results.

Two threads run independently:
  - Receiver thread: tight UDP-read loop, reassembles complete JPEGs, posts the
    LATEST one to a single-slot mailbox. Never blocked by inference, so it can
    keep up with the sim regardless of how long detection takes.
  - Inference thread: pops the latest JPEG, decodes it, runs predict.detect,
    and updates the public _latest_* state. If inference is slower than the
    incoming frame rate, the receiver simply overwrites the mailbox and the
    inference thread skips stale frames — no queue buildup, no packet drops.

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

VISION_HOST = "0.0.0.0"
VISION_PORT = 5600

_HEADER_FMT  = "<IHHIIQ"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAX_PENDING = 8           # max in-flight frame reassemblies before old ones are dropped

# ----------------------------------------------------------------------------
# Public state (mutated by the inference thread, read by world_building/brain).
# ----------------------------------------------------------------------------
_latest_frame     = None
_latest_gates     = []
_latest_annotated = None
_frame_counter    = 0      # increments on every successful inference
_state_lock       = threading.Lock()
_raw_queue        = []     # raw decoded frames for display, drained by main thread

# ----------------------------------------------------------------------------
# Mailbox between receiver and inference threads. Single-slot: the receiver
# overwrites whatever's there, so the inference thread always sees the newest
# JPEG and stale ones are dropped automatically.
# ----------------------------------------------------------------------------
_pending_jpeg     = None
_pending_jpeg_cv  = threading.Condition()

# ----------------------------------------------------------------------------
# Mailbox between inference and annotator threads. Inference posts the most
# recent (image, gates) here; the annotator draws boxes/keypoints into a fresh
# image and updates _latest_annotated. Single-slot, newest-wins.
# ----------------------------------------------------------------------------
_pending_annotation     = None    # (image, gates) tuple
_pending_annotation_cv  = threading.Condition()

# Colors and sizes for the manual annotator (BGR).
_ANNOT_BOX_COLOR     = (0, 200, 0)
_ANNOT_BOX_THICKNESS = 2
_ANNOT_KP_INNER_COLOR = (0, 255, 0)    # bright green for inner corners
_ANNOT_KP_OUTER_COLOR = (0, 200, 255)  # orange for outer corners
_ANNOT_KP_RADIUS      = 4
_ANNOT_LABEL_COLOR    = (255, 255, 255)

# Diagnostic counters (separately tracked so we can see receive vs inference fps).
_received_frames  = 0
_processed_frames = 0
_skipped_frames   = 0


def get_latest():
    """Return (frame, gates, annotated) for the most recently processed image."""
    with _state_lock:
        return _latest_frame, _latest_gates, _latest_annotated


def get_detections():
    """Return (gates, frame_id) — frame_id increments when a new inference completes.
    Consumers can compare frame_id against a previous value to skip stale data."""
    with _state_lock:
        return _latest_gates, _frame_counter


def pop_raw_frame():
    """Return the latest raw frame for display, or None. Drains the queue."""
    with _state_lock:
        if not _raw_queue:
            return None
        frame = _raw_queue[-1]
        _raw_queue.clear()
        return frame


def start():
    """Spawn the receiver, inference, and annotator threads. Non-blocking."""
    threading.Thread(target=_receiver_thread,  daemon=True, name="VisionReceiver").start()
    threading.Thread(target=_inference_thread, daemon=True, name="VisionInference").start()
    threading.Thread(target=_annotator_thread, daemon=True, name="VisionAnnotator").start()


# ----------------------------------------------------------------------------
# Receiver thread — tight UDP-read loop, posts complete JPEGs to the mailbox.
# ----------------------------------------------------------------------------
def _receiver_thread():
    global _pending_jpeg, _received_frames, _skipped_frames, _processed_frames

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    sock.bind((VISION_HOST, VISION_PORT))
    print(f"[vision] Listening on {VISION_HOST}:{VISION_PORT}")

    pending_reassembly = {}    # frame_id -> {"total": int, "chunks": {chunk_id: bytes}}
    last_complete_frame_id = -1
    last_log_time = time.time()

    while True:
        # Per-second diagnostic: receive fps vs inference fps.
        now = time.time()
        if now - last_log_time > 1.0:
            last_log_time = now
            with _state_lock:
                proc = _processed_frames
                _processed_frames = 0
            rec = _received_frames
            skipped = _skipped_frames
            _received_frames = 0
            _skipped_frames = 0
            print(f"[vision] received={rec} fps, processed={proc} fps, skipped={skipped}")

        packet, _addr = sock.recvfrom(65535)
        parsed = _parse_chunk(packet)
        if parsed is None:
            continue

        frame_id, chunk_id, total_chunks, chunk_data = parsed
        if frame_id <= last_complete_frame_id:
            continue   # this frame is older than one we already completed

        jpeg = _reassemble(pending_reassembly, frame_id, chunk_id, total_chunks, chunk_data)
        if jpeg is None:
            continue   # frame not fully assembled yet

        last_complete_frame_id = frame_id
        _received_frames += 1

        # Hand off to the inference thread. If the previous JPEG hasn't been
        # consumed yet, we overwrite it — newest-frame-wins semantics.
        with _pending_jpeg_cv:
            if _pending_jpeg is not None:
                _skipped_frames += 1
            _pending_jpeg = jpeg
            _pending_jpeg_cv.notify()


# ----------------------------------------------------------------------------
# Inference thread — waits for the receiver to post a JPEG, then runs detect.
# ----------------------------------------------------------------------------
def _inference_thread():
    global _pending_jpeg
    while True:
        with _pending_jpeg_cv:
            while _pending_jpeg is None:
                _pending_jpeg_cv.wait()
            jpeg = _pending_jpeg
            _pending_jpeg = None
        _process_frame(jpeg)


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
                "method":  g["pose"].get("method", "unknown"),
            }

        gate_records.append({
            "id":           i,
            "n_confident":  g.get("n_confident", 0),
            "pose_method":  g.get("pose_method", "none"),
            "corners":      corners,
            "pose":         pose_record,
        })

    line = {
        "t":     round(log_dir.elapsed(), 4),
        "frame": _frame_counter,
        "gates": gate_records,
    }
    _log_file.write(json.dumps(line, indent=2) + "\n")


def _process_frame(jpeg):
    global _latest_frame, _latest_gates, _frame_counter, _processed_frames, _pending_annotation

    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return
    with _state_lock:
        _raw_queue.append(img)

    # Run YOLO + PnP. The annotated frame is built off-thread by the annotator
    # so this critical path stays as short as possible.
    gates, _annotated_from_predict = predict.detect(img)

    if gates:
        summary = ", ".join(
            f"g{i}({g.get('n_confident', 0)}kp/{g.get('pose_method', 'none')})"
            for i, g in enumerate(gates)
        )
        print(f"[vision] t={log_dir.elapsed():6.2f}s  {summary}")

    _log_gate_corners(gates)

    with _state_lock:
        _latest_frame   = img
        _latest_gates   = gates
        _frame_counter += 1
        _processed_frames += 1

    # Hand the freshest (image, gates) to the annotator. Newest-wins: if the
    # annotator hasn't picked up the last one, we just overwrite it.
    with _pending_annotation_cv:
        _pending_annotation = (img, gates)
        _pending_annotation_cv.notify()


def _annotator_thread():
    """Draws bounding boxes and keypoint dots on the most recent frame using
    plain OpenCV calls (~2-3ms). Runs off the inference critical path so slow
    drawing never starves the inference thread."""
    global _pending_annotation, _latest_annotated
    while True:
        with _pending_annotation_cv:
            while _pending_annotation is None:
                _pending_annotation_cv.wait()
            img, gates = _pending_annotation
            _pending_annotation = None

        annotated = _draw_annotations(img, gates)

        with _state_lock:
            _latest_annotated = annotated


_KEYPOINT_LABELS = ["TLI", "TRI", "BLI", "BRI", "TLO", "TRO", "BLO", "BRO"]

def _draw_annotations(img, gates):
    """Return a copy of img with each gate's bbox and keypoints drawn."""
    out = img.copy()
    for gate in gates:
        box = gate.get("box")
        if box is not None:
            x0, y0, x1, y1 = [int(v) for v in box]
            cv2.rectangle(out, (x0, y0), (x1, y1), _ANNOT_BOX_COLOR, _ANNOT_BOX_THICKNESS)
            label = f"gate {gate.get('conf', 0):.2f}"
            cv2.putText(out, label, (x0, max(15, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _ANNOT_LABEL_COLOR, 1, cv2.LINE_AA)

        keypoints = gate.get("keypoints")
        confs     = gate.get("keypoint_conf")
        if keypoints is None or confs is None:
            continue
        for k, ((px, py), c) in enumerate(zip(keypoints, confs)):
            if c < predict.KEYPOINT_CONF_THRESHOLD:
                continue
            color = _ANNOT_KP_OUTER_COLOR if k >= 4 else _ANNOT_KP_INNER_COLOR
            ipx, ipy = int(px), int(py)
            cv2.circle(out, (ipx, ipy), _ANNOT_KP_RADIUS, color, -1)
            name = _KEYPOINT_LABELS[k] if k < len(_KEYPOINT_LABELS) else str(k)
            # Small offset so text doesn't overlap the dot.
            cv2.putText(out, f"{name} {c:.2f}", (ipx + 5, ipy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return out
