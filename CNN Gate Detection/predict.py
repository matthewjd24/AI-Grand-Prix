"""Gate detection module — import and call detect(image) to get gate poses.

Returns one dict per detected gate:
    box           np.ndarray (4,)   xyxy bounding box in pixels
    conf          float             detection confidence
    keypoints     np.ndarray (4,2)  TL/TR/BL/BR corner pixels
    keypoint_conf np.ndarray (4,)   per-corner confidence
    pose          dict or None      PnP result if >= 4 corners confident:
        tvec      np.ndarray (3,1)  gate position in camera frame (metres)
        rvec      np.ndarray (3,1)  rotation vector
        distance  float             metres to gate centre
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

_WEIGHTS = os.path.join(os.path.dirname(__file__), r"runs\pose\runs\gate-pose\weights\best.pt")

# Camera intrinsics — Unity training camera, 73 deg VFoV, 640x360
_f = 180 / np.tan(np.radians(73 / 2))
_K = np.array([[_f,  0, 320],
               [ 0, _f, 180],
               [ 0,   0,  1]], dtype=np.float64)
_DIST = np.zeros(4)

# Gate outer corners in gate frame: TL, TR, BL, BR (metres, z=0 in gate plane)
_h = 2.7 / 2
_GATE_PTS_3D = np.array([
    [-_h,  _h, 0],
    [ _h,  _h, 0],
    [-_h, -_h, 0],
    [ _h, -_h, 0],
], dtype=np.float32)

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = YOLO(_WEIGHTS)
    return _model


def detect(image):
    """Run gate detection on a BGR image. Returns a list of gate dicts."""
    result = _get_model().predict(source=image, verbose=False)[0]
    kps = result.keypoints
    gates = []

    for i in range(len(result.boxes)):
        xy   = kps.xy[i].cpu().numpy()    # (4, 2)
        conf = kps.conf[i].cpu().numpy()   # (4,)
        mask = conf > 0.5

        pose = None
        if mask.sum() >= 4:
            ok, rvec, tvec = cv2.solvePnP(_GATE_PTS_3D[mask], xy[mask], _K, _DIST)
            if ok:
                pose = {
                    "rvec":     rvec,
                    "tvec":     tvec,
                    "distance": float(np.linalg.norm(tvec)),
                }

        gates.append({
            "box":           result.boxes.xyxy[i].cpu().numpy(),
            "conf":          float(result.boxes.conf[i]),
            "keypoints":     xy,
            "keypoint_conf": conf,
            "pose":          pose,
        })

    return gates, result.plot(line_width=2)
