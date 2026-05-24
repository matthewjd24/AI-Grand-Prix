"""Gate detection module — import and call detect(image) to get gate poses.

Returns one dict per detected gate:
    box           np.ndarray (4,)        xyxy bounding box in pixels
    conf          float                  detection confidence
    keypoints     np.ndarray (N,2)       corner pixels (N=4 or 8, see USE_8_KEYPOINTS)
    keypoint_conf np.ndarray (N,)        per-corner confidence
    pose          dict or None           PnP result if >= 4 corners confident:
        tvec      np.ndarray (3,1)       gate position in camera frame (metres)
        rvec      np.ndarray (3,1)       rotation vector
        distance  float                  metres to gate centre

Keypoint order (must match Unity TrainingLabelWriter and data.yaml):
  4-kp: TL, TR, BL, BR
  8-kp: TL, TR, BL, BR, TL_out, TR_out, BL_out, BR_out
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

_WEIGHTS = os.path.join(os.path.dirname(__file__), r"runs\pose\runs\gate-pose-2\weights\best.pt")

# Camera intrinsics — matches spec §3.8 (fx=fy=320 at 640x360 → HFoV=90°,
# VFoV≈58.7°). sim.py renders with the same projection.
_K = np.array([[320,   0, 320],
               [  0, 320, 180],
               [  0,   0,   1]], dtype=np.float64)
_DIST = np.zeros(4)

# Gate keypoints in gate-local frame, metres, z=0 in the gate plane.
# Order MUST match the keypoint order produced by Unity TrainingImageQuality.
# - Inner square: 1.5 m (spec §3.7) — TL, TR, BL, BR
# - Outer frame:  2.7 m (spec §3.7) — TL_out, TR_out, BL_out, BR_out
# When the model is trained with `includeOuterCorners=true`, set USE_8_KEYPOINTS=True.
USE_8_KEYPOINTS = True

_hi = 1.5 / 2
_ho = 2.7 / 2
_INNER_PTS = np.array([
    [-_hi,  _hi, 0],
    [ _hi,  _hi, 0],
    [-_hi, -_hi, 0],
    [ _hi, -_hi, 0],
], dtype=np.float32)
_OUTER_PTS = np.array([
    [-_ho,  _ho, 0],
    [ _ho,  _ho, 0],
    [-_ho, -_ho, 0],
    [ _ho, -_ho, 0],
], dtype=np.float32)
_GATE_PTS_3D = np.concatenate([_INNER_PTS, _OUTER_PTS], axis=0) if USE_8_KEYPOINTS else _INNER_PTS

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = YOLO(_WEIGHTS)
        _model.to("cuda")
        print(f"[predict] YOLO loaded on {next(_model.model.parameters()).device}")
    return _model


def _estimate_partial_pose(xy, mask):
    """Rough pose from 2-3 visible corners. Uses the pair with the largest known
    3D distance to estimate depth, then projects the midpoint to get x/y.
    Orientation is unreliable from <4 points — we just set rvec to zero (facing camera)."""
    visible = np.where(mask)[0]

    # Find the pair of visible corners with the largest known 3D distance —
    # longer baselines give better depth resolution.
    best = None
    best_d3d = 0.0
    for i in range(len(visible)):
        for j in range(i + 1, len(visible)):
            a, b = visible[i], visible[j]
            d3d = float(np.linalg.norm(_GATE_PTS_3D[a] - _GATE_PTS_3D[b]))
            if d3d > best_d3d:
                best_d3d, best = d3d, (a, b)
    if best is None:
        return None

    a, b = best
    d_px = float(np.linalg.norm(xy[a] - xy[b]))
    if d_px < 1e-3:
        return None

    fx, fy = _K[0, 0], _K[1, 1]
    cx, cy = _K[0, 2], _K[1, 2]
    z = fx * best_d3d / d_px

    mid_px = xy[visible].mean(axis=0)
    x = (mid_px[0] - cx) * z / fx
    y = (mid_px[1] - cy) * z / fy

    tvec = np.array([[x], [y], [z]], dtype=np.float64)
    rvec = np.zeros((3, 1), dtype=np.float64)   # orientation underdetermined
    return {
        "rvec":     rvec,
        "tvec":     tvec,
        "distance": float(np.linalg.norm(tvec)),
        "partial":  True,
    }


def detect(image):
    """Run gate detection on a BGR image. Returns a list of gate dicts."""
    result = _get_model().predict(source=image, verbose=False, device="cuda")[0]
    kps = result.keypoints
    gates = []

    for i in range(len(result.boxes)):
        xy   = kps.xy[i].cpu().numpy()    # (N, 2) with N = len(_GATE_PTS_3D)
        conf = kps.conf[i].cpu().numpy()   # (N,)
        mask = conf > 0.2

        pose = None
        if mask.sum() >= 4:
            ok, rvec, tvec = cv2.solvePnP(_GATE_PTS_3D[mask], xy[mask], _K, _DIST)
            if ok:
                pose = {
                    "rvec":     rvec,
                    "tvec":     tvec,
                    "distance": float(np.linalg.norm(tvec)),
                    "partial":  False,
                }
        elif mask.sum() >= 2:
            pose = _estimate_partial_pose(xy, mask)

        gates.append({
            "box":           result.boxes.xyxy[i].cpu().numpy(),
            "conf":          float(result.boxes.conf[i]),
            "keypoints":     xy,
            "keypoint_conf": conf,
            "pose":          pose,
        })

    return gates, result.plot(line_width=2)
