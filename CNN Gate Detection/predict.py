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

import atexit
import os
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# log_dir lives at the project root; available when running under brain.py.
# If predict.py is imported standalone the import may fail — fall back to CWD.
try:
    import log_dir
    _DEBUG_VIDEO_PATH = log_dir.log_path("refinement.mp4")
except ImportError:
    _DEBUG_VIDEO_PATH = "refinement.mp4"

_WEIGHTS = os.path.join(os.path.dirname(__file__), r"runs\pose\runs\gate-pose-5\weights\best.engine")

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

# Keypoints below this confidence are treated as "the model isn't sure" — they
# don't feed into PnP, refinement, or the annotated frame. Shared with vision.py.
KEYPOINT_CONF_THRESHOLD = 0.7

# When True, run color-mask + cornerSubPix refinement on YOLO's keypoints
# before PnP. When False, PnP runs directly on the raw YOLO keypoints.
USE_REFINEMENT = False
# When True, detect() also calls result.plot() to produce an annotated frame
# for visualization. Costs ~5-10ms — set False during competition runs where
# nothing consumes the annotated frame.
RETURN_ANNOTATED_FRAME = True
# When True, runs the model in FP16 (half precision). Roughly 1.5-2x faster
# on RTX-20 and newer GPUs. Negligible accuracy impact for detection.
USE_HALF_PRECISION = True
# When True, prints a per-frame timing breakdown showing how long the model
# forward pass takes vs the rest of detect() (postprocessing, refinement, etc).
# Use to locate the real bottleneck.
TIME_INFERENCE = True
# When True, detect() builds a stacked composite of each gate's 4-panel debug
# view, shows it in a single live window, and writes it to a video file.
DEBUG_REFINEMENT = False
# Layout of the composite. Per-gate panel is resized to this size; the video
# stacks MAX_DEBUG_GATES of them vertically. Missing gates show a black slot.
_DEBUG_SLOT_W = 640
_DEBUG_SLOT_H = 480
MAX_DEBUG_GATES = 2

# Module-level state for the per-frame composite and the lazy video writer.
_debug_panels = []           # list of per-gate panel images for the current frame
_debug_video_writer = None
_debug_video_size = None     # (w, h) the writer was opened with


def _release_debug_video():
    """Close the video file cleanly so the mp4 trailer is written. Registered
    with atexit so it runs even on Ctrl+C or unhandled exceptions."""
    global _debug_video_writer
    if _debug_video_writer is not None:
        _debug_video_writer.release()
        _debug_video_writer = None
        print(f"[predict] refinement debug video closed: {_DEBUG_VIDEO_PATH}")

atexit.register(_release_debug_video)

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = YOLO(_WEIGHTS)
        # TensorRT engines (and other exported formats) already have device and
        # precision baked in — calling .to() or .half() on them errors out.
        # Only configure those for raw PyTorch (.pt) weights.
        is_pt = str(_WEIGHTS).lower().endswith(".pt")
        if is_pt:
            _model.to("cuda")
            if USE_HALF_PRECISION:
                # Force the underlying torch model to FP16. The half=True kwarg
                # on .predict() is unreliable across Ultralytics versions; this
                # is the belt-and-braces way to ensure FP16 math on the GPU.
                _model.model.half()
            dtype = next(_model.model.parameters()).dtype
            device = next(_model.model.parameters()).device
            print(f"[predict] YOLO (.pt) loaded on {device} dtype={dtype}")
        else:
            print(f"[predict] YOLO engine loaded: {os.path.basename(_WEIGHTS)}")
    return _model


def warmup():
    """Force the model to load and run a single dummy inference. Blocks until
    everything is ready. Call once at process startup so the first real frame
    doesn't pay the load + CUDA-compile cost."""
    dummy = np.zeros((360, 640, 3), dtype=np.uint8)
    detect(dummy)


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


# cv2.cornerSubPix tuning. Used as a 2nd-pass refinement after the color-mask
# corner finder, and as a 1st-pass refinement when the color-mask fails.
_SUBPIX_WIN  = (15, 15)
_SUBPIX_ZERO = (-1, -1)
_SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
_SUBPIX_MAX_DRIFT = 15.0

# Color-mask refinement: exploit the fact that each gate is a uniform-color
# square frame. We isolate that color with an HSV mask, fit quadrilaterals
# to the outer and inner contours, and use those quad corners as precise
# replacements for the CNN's keypoint predictions.
_HSV_H_TOL = 18       # hue tolerance around the sampled gate color
_HSV_S_TOL = 120      # saturation tolerance (loose — lighting can wash it out)
_HSV_V_TOL = 120      # value tolerance (loose — shading varies across the frame face)
_BBOX_PAD  = 8        # pixels of padding around the YOLO bbox before masking
_POLY_EPS_FRAC = 0.03 # cv2.approxPolyDP epsilon as a fraction of contour length
_CORNER_MAX_DRIFT = 20.0  # max pixel distance from YOLO seed to accept a color-mask corner
# True: fit a line through each of the 4 contour sides and intersect adjacent
# lines to get corners. Robust to image-edge clipping and contour kinks; gives
# subpixel corners by construction.
# False: use cv2.approxPolyDP (the original approach).
USE_LINE_FIT = True
# When line-fit is on, a side needs at least this many contour pixels for the
# fitted line to be trustworthy. Otherwise the whole quad fit is rejected.
_LINE_FIT_MIN_PTS_PER_SIDE = 6
# When the gate frame is clipped by the image edge, the contour follows the
# image border and approxPolyDP returns a fake "corner" sitting on the border.
# Reject any quad corner within this many pixels of any image edge so we
# don't snap the YOLO seed to a fake border-corner.
_BORDER_REJECT_PX = 3.0


def _build_refinement_panel(image_bgr, box, yolo_xy, hsv_mask, outer_quad, inner_quad, refined_xy, sample_points=None):
    """Build (don't show) a 2x2 panel image for one gate's refinement stages.
    The caller appends the returned image to _debug_panels for later compositing."""
    h_img, w_img = image_bgr.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    pad = 20
    rx0 = max(0, x0 - pad); ry0 = max(0, y0 - pad)
    rx1 = min(w_img, x1 + pad); ry1 = min(h_img, y1 + pad)

    def crop_and_label(canvas, _title=None):
        out = canvas[ry0:ry1, rx0:rx1].copy()
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        return out

    # Panel 1: YOLO seeds (raw input to refinement) + the midpoint pixels we
    # sampled for HSV color (small magenta dots).
    seeds_canvas = image_bgr.copy()
    cv2.rectangle(seeds_canvas, (x0, y0), (x1, y1), (255, 0, 0), 1)
    for k in range(len(yolo_xy)):
        cv2.circle(seeds_canvas, (int(yolo_xy[k, 0]), int(yolo_xy[k, 1])), 3, (0, 0, 255), -1)
    if sample_points:
        for (px, py) in sample_points:
            cv2.circle(seeds_canvas, (int(px), int(py)), 2, (255, 0, 255), -1)
    panel_seeds = crop_and_label(seeds_canvas, "1: YOLO seeds + sample pts")

    # Panel 2: HSV mask (full ROI shown as grayscale-on-bgr).
    if hsv_mask is not None:
        full_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        # hsv_mask is ROI-sized; paste it into a full-image canvas.
        bx0 = max(0, x0 - _BBOX_PAD); by0 = max(0, y0 - _BBOX_PAD)
        bx1 = min(w_img, x1 + _BBOX_PAD); by1 = min(h_img, y1 + _BBOX_PAD)
        mh = min(hsv_mask.shape[0], by1 - by0)
        mw = min(hsv_mask.shape[1], bx1 - bx0)
        full_mask[by0:by0 + mh, bx0:bx0 + mw] = hsv_mask[:mh, :mw]
        panel_mask = crop_and_label(full_mask, "2: HSV mask")
    else:
        panel_mask = crop_and_label(np.zeros_like(image_bgr), "2: HSV mask (none)")

    # Panel 3: fitted quadrilaterals.
    quad_canvas = image_bgr.copy()
    if outer_quad is not None:
        cv2.polylines(quad_canvas, [outer_quad.astype(np.int32)], True, (0, 200, 255), 1)
        for c in outer_quad:
            cv2.circle(quad_canvas, (int(c[0]), int(c[1])), 3, (0, 200, 255), -1)
    if inner_quad is not None:
        cv2.polylines(quad_canvas, [inner_quad.astype(np.int32)], True, (255, 200, 0), 1)
        for c in inner_quad:
            cv2.circle(quad_canvas, (int(c[0]), int(c[1])), 3, (255, 200, 0), -1)
    panel_quads = crop_and_label(quad_canvas, "3: fitted quads (cyan=inner, orange=outer)")

    # Panel 4: final refined keypoints overlaid on top of YOLO seeds with arrows.
    final_canvas = image_bgr.copy()
    for k in range(len(yolo_xy)):
        a = (int(yolo_xy[k, 0]), int(yolo_xy[k, 1]))
        b = (int(refined_xy[k, 0]), int(refined_xy[k, 1]))
        cv2.circle(final_canvas, a, 2, (0, 0, 255), -1)            # red = yolo
        cv2.circle(final_canvas, b, 3, (0, 255, 0), 1)             # green = final
        if a != b:
            cv2.arrowedLine(final_canvas, a, b, (255, 255, 255), 1, tipLength=0.3)
    panel_final = crop_and_label(final_canvas, "4: final (red=YOLO, green=refined)")

    # Stack into 2x2.
    target_h = max(panel_seeds.shape[0], panel_mask.shape[0],
                   panel_quads.shape[0], panel_final.shape[0])
    target_w = max(panel_seeds.shape[1], panel_mask.shape[1],
                   panel_quads.shape[1], panel_final.shape[1])
    def pad_to(p):
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:p.shape[0], :p.shape[1]] = p
        return canvas
    top    = np.hstack([pad_to(panel_seeds), pad_to(panel_mask)])
    bottom = np.hstack([pad_to(panel_quads), pad_to(panel_final)])
    return np.vstack([top, bottom])


def _compose_and_show_debug_video(frame_w, frame_h):
    """Resize each per-gate panel in _debug_panels to a fixed slot, place
    MAX_DEBUG_GATES of them side-by-side horizontally (missing slots = black),
    show in a live window, and write to the lazy-init video writer."""
    global _debug_video_writer, _debug_video_size

    composite_h = _DEBUG_SLOT_H
    composite_w = _DEBUG_SLOT_W * MAX_DEBUG_GATES
    composite = np.zeros((composite_h, composite_w, 3), dtype=np.uint8)

    for i in range(min(MAX_DEBUG_GATES, len(_debug_panels))):
        panel = _debug_panels[i]
        if panel is None or panel.size == 0:
            continue
        resized = cv2.resize(panel, (_DEBUG_SLOT_W, _DEBUG_SLOT_H), interpolation=cv2.INTER_AREA)
        x0 = i * _DEBUG_SLOT_W
        composite[:, x0:x0 + _DEBUG_SLOT_W] = resized

    cv2.imshow("refinement debug", composite)
    cv2.waitKey(1)   # non-blocking; keeps the window responsive during flight

    # Lazy-open the video writer the first time we have a composite.
    if _debug_video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        _debug_video_size = (composite_w, composite_h)
        _debug_video_writer = cv2.VideoWriter(_DEBUG_VIDEO_PATH, fourcc, 30.0, _debug_video_size)
        print(f"[predict] recording refinement debug video to {_DEBUG_VIDEO_PATH}")
    _debug_video_writer.write(composite)


def _line_intersection(line_a, line_b):
    """Intersect two lines, each in cv2.fitLine format (vx, vy, x0, y0).
    Returns (px, py) or None if the lines are (nearly) parallel."""
    vx1, vy1, x1, y1 = line_a
    vx2, vy2, x2, y2 = line_b
    det = vx1 * vy2 - vy1 * vx2
    if abs(det) < 1e-6:
        return None
    t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / det
    return (float(x1 + t * vx1), float(y1 + t * vy1))


def _reorder_quad_image_axes(corners):
    """Reorder 4 corners into image-frame TL, TR, BL, BR order
    (top two by y, then within each row left-to-right by x)."""
    by_y = corners[np.argsort(corners[:, 1])]
    top_two = by_y[:2]
    bot_two = by_y[2:]
    top_two = top_two[np.argsort(top_two[:, 0])]
    bot_two = bot_two[np.argsort(bot_two[:, 0])]
    return np.array([top_two[0], top_two[1], bot_two[0], bot_two[1]], dtype=np.float32)


def _fit_quad_via_lines(contour):
    """Fit a line to each of the 4 sides of a quadrilateral contour and
    intersect adjacent lines to get 4 corners. Robust to corners that are
    clipped by the image edge: the missing pixels just reduce the number of
    samples on that side; the fitted line still extrapolates correctly.
    Returns 4 corners in TL/TR/BL/BR order, or None on failure."""
    if len(contour) < 4 * _LINE_FIT_MIN_PTS_PER_SIDE:
        return None

    pts = contour.reshape(-1, 2).astype(np.float32)

    # Use the minAreaRect to learn the gate's rotation, then bucket each
    # contour point into one of 4 sides based on which axis of the rotated
    # rect it dominates. This works regardless of image rotation.
    rect = cv2.minAreaRect(contour)
    (cx, cy), _, angle_deg = rect
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(-angle_rad), np.sin(-angle_rad)

    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    lx = cos_a * dx - sin_a * dy   # rect-local x
    ly = sin_a * dx + cos_a * dy   # rect-local y

    abs_lx, abs_ly = np.abs(lx), np.abs(ly)
    horiz = abs_lx > abs_ly        # this point is closer to a left/right side
    side_masks = [
        ~horiz & (ly < 0),         # top
        horiz  & (lx > 0),         # right
        ~horiz & (ly > 0),         # bottom
        horiz  & (lx < 0),         # left
    ]

    # Fit a line to each side. Bail if any side has too few points.
    lines = []
    for mask in side_masks:
        if mask.sum() < _LINE_FIT_MIN_PTS_PER_SIDE:
            return None
        line = cv2.fitLine(pts[mask], cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        lines.append(line)

    # Adjacent intersections: top∩right, right∩bottom, bottom∩left, left∩top.
    corners = []
    for i in range(4):
        pt = _line_intersection(lines[i], lines[(i + 1) % 4])
        if pt is None:
            return None
        corners.append(pt)

    return _reorder_quad_image_axes(np.array(corners, dtype=np.float32))


def _refine_via_color_mask(image_bgr, box, yolo_xy, conf):
    """Exploit the gate's uniform color and known shape: HSV-mask the ROI,
    fit quadrilaterals to the outer and inner contours, and pair their
    corners to YOLO's seeds. Returns (refined_xy, refined_mask) where
    refined_mask[k] is True if keypoint k was successfully refined."""
    h_img, w_img = image_bgr.shape[:2]
    refined = yolo_xy.copy()
    refined_mask = np.zeros(len(yolo_xy), dtype=bool)

    # Pad and clip the bbox so we can find the outer contour without it
    # being cut off by the bbox edge.
    x0 = max(0, int(box[0]) - _BBOX_PAD)
    y0 = max(0, int(box[1]) - _BBOX_PAD)
    x1 = min(w_img, int(box[2]) + _BBOX_PAD)
    y1 = min(h_img, int(box[3]) + _BBOX_PAD)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return refined, refined_mask

    roi = image_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Sample gate color at midpoints between each inner/outer keypoint pair.
    # Both ends of each pair sit on the gate's edges (inner ring and outer
    # ring), so their midpoint is guaranteed to land on the green frame
    # itself — much more reliable than sampling toward the bbox center,
    # which lands inside the hole.
    samples = []
    sample_points_full = []   # in full-image coords, for debug viz
    if len(yolo_xy) >= 8:
        for k in range(4):
            inner = yolo_xy[k]
            outer = yolo_xy[k + 4]
            # Skip the pair if either keypoint is unreliable.
            if conf[k] < 0.3 or conf[k + 4] < 0.3:
                continue
            mx_full = (inner[0] + outer[0]) / 2.0
            my_full = (inner[1] + outer[1]) / 2.0
            # Both ends and the midpoint must lie inside the image.
            if not (0 <= mx_full < image_bgr.shape[1] and 0 <= my_full < image_bgr.shape[0]):
                continue
            sx = int(mx_full) - x0
            sy = int(my_full) - y0
            if 0 <= sx < hsv.shape[1] and 0 <= sy < hsv.shape[0]:
                samples.append(hsv[sy, sx])
                sample_points_full.append((mx_full, my_full))

    # Fallback for 4-keypoint mode (no outer pair available): sample slightly
    # outward from each inner corner toward the bbox edge.
    if not samples:
        cx_box = (box[0] + box[2]) / 2.0
        cy_box = (box[1] + box[3]) / 2.0
        inner_count = min(4, len(yolo_xy))
        for k in range(inner_count):
            ix, iy = yolo_xy[k]
            # Step outward (away from center) by ~25% of the half-box-width.
            dx = ix - cx_box
            dy = iy - cy_box
            mx_full = ix + 0.25 * dx
            my_full = iy + 0.25 * dy
            if not (0 <= mx_full < image_bgr.shape[1] and 0 <= my_full < image_bgr.shape[0]):
                continue
            sx = int(mx_full) - x0
            sy = int(my_full) - y0
            if 0 <= sx < hsv.shape[1] and 0 <= sy < hsv.shape[0]:
                samples.append(hsv[sy, sx])
                sample_points_full.append((mx_full, my_full))

    if not samples:
        return refined, refined_mask
    h_med, s_med, v_med = [int(v) for v in np.median(samples, axis=0)]

    # OpenCV hue is 0-180 with red sitting across the 0/180 wrap. If our
    # hue window crosses either edge, do two inRange calls and OR them.
    s_lo = max(0, s_med - _HSV_S_TOL)
    v_lo = max(0, v_med - _HSV_V_TOL)
    h_lo_raw = h_med - _HSV_H_TOL
    h_hi_raw = h_med + _HSV_H_TOL
    if h_lo_raw < 0:
        m1 = cv2.inRange(hsv, np.array([0,            s_lo, v_lo], dtype=np.uint8),
                              np.array([h_hi_raw,     255,  255 ], dtype=np.uint8))
        m2 = cv2.inRange(hsv, np.array([180 + h_lo_raw, s_lo, v_lo], dtype=np.uint8),
                              np.array([180,           255,  255 ], dtype=np.uint8))
        mask = cv2.bitwise_or(m1, m2)
    elif h_hi_raw > 180:
        m1 = cv2.inRange(hsv, np.array([h_lo_raw,     s_lo, v_lo], dtype=np.uint8),
                              np.array([180,          255,  255 ], dtype=np.uint8))
        m2 = cv2.inRange(hsv, np.array([0,            s_lo, v_lo], dtype=np.uint8),
                              np.array([h_hi_raw - 180, 255, 255], dtype=np.uint8))
        mask = cv2.bitwise_or(m1, m2)
    else:
        mask = cv2.inRange(hsv,
                           np.array([h_lo_raw, s_lo, v_lo], dtype=np.uint8),
                           np.array([h_hi_raw, 255,  255 ], dtype=np.uint8))
    # Clean up small specks and gaps.
    # Bigger closing kernel fills small holes left by lighting/shading variance
    # inside the frame ring; opening removes thin speckle bleed.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours or hierarchy is None:
        return refined, refined_mask

    # Outer contour = largest top-level contour. Its inner hole, if any,
    # appears as a child contour in the CCOMP hierarchy.
    h = hierarchy[0]
    top_level = [i for i in range(len(contours)) if h[i][3] == -1]
    if not top_level:
        return refined, refined_mask
    outer_idx = max(top_level, key=lambda i: cv2.contourArea(contours[i]))

    # The inner hole is whichever child has the largest area.
    children = []
    j = h[outer_idx][2]
    while j != -1:
        children.append(j)
        j = h[j][0]
    inner_idx = max(children, key=lambda i: cv2.contourArea(contours[i])) if children else None

    def _fit_quad(idx):
        c = contours[idx]
        if USE_LINE_FIT:
            quad_local = _fit_quad_via_lines(c)
            if quad_local is None:
                return None
            return quad_local + np.array([x0, y0], dtype=np.float32)
        # Fallback: cv2.approxPolyDP.
        eps = _POLY_EPS_FRAC * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) != 4:
            return None
        return approx.astype(np.float32) + np.array([x0, y0], dtype=np.float32)

    outer_quad = _fit_quad(outer_idx)
    inner_quad = _fit_quad(inner_idx) if inner_idx is not None else None

    # Mark which quad corners are clipped by the image edge. Those positions
    # are fake (contour followed the border) and must not be paired to YOLO seeds.
    outer_valid = _mark_valid_corners(outer_quad, w_img, h_img)
    inner_valid = _mark_valid_corners(inner_quad, w_img, h_img)

    # Pair each YOLO seed with the nearest unclaimed CV corner from the
    # appropriate quad (inner seeds → inner_quad, outer seeds → outer_quad).
    has_outer_kpts = len(yolo_xy) >= 8
    inner_seed_count = 4 if has_outer_kpts else len(yolo_xy)
    _assign_quad_corners(yolo_xy, refined, refined_mask,
                         seed_start=0, seed_count=inner_seed_count,
                         quad=inner_quad, valid=inner_valid)
    if has_outer_kpts:
        _assign_quad_corners(yolo_xy, refined, refined_mask,
                             seed_start=4, seed_count=4,
                             quad=outer_quad, valid=outer_valid)

    if DEBUG_REFINEMENT:
        panel = _build_refinement_panel(image_bgr, box, yolo_xy, mask,
                                        outer_quad, inner_quad, refined,
                                        sample_points_full)
        _debug_panels.append(panel)
    return refined, refined_mask


def _mark_valid_corners(quad, image_w, image_h):
    """Return a 4-element boolean array marking each quad corner as valid
    (False if it sits within _BORDER_REJECT_PX of any image edge — likely a
    fake corner produced by the contour following the image border)."""
    if quad is None or len(quad) != 4:
        return None
    valid = np.ones(4, dtype=bool)
    for i, (x, y) in enumerate(quad):
        if (x < _BORDER_REJECT_PX or x > image_w - 1 - _BORDER_REJECT_PX
                or y < _BORDER_REJECT_PX or y > image_h - 1 - _BORDER_REJECT_PX):
            valid[i] = False
    return valid


def _assign_quad_corners(yolo_xy, refined, refined_mask, seed_start, seed_count, quad, valid=None):
    """Match each of the seeds [seed_start..seed_start+seed_count) to its
    nearest unclaimed corner in `quad`. Skip pairings where the corner is
    farther than _CORNER_MAX_DRIFT from the seed, or where `valid[i]` is False."""
    if quad is None or len(quad) != 4 or seed_count == 0:
        return
    seeds = yolo_xy[seed_start:seed_start + seed_count]
    n = min(seed_count, 4)
    # Build cost matrix (n seeds x 4 quad corners) and greedily pick the
    # globally-smallest pair, then the next, etc. n is tiny so greedy is fine.
    dist = np.linalg.norm(seeds[:, None, :] - quad[None, :, :], axis=2)
    # Mask out invalid corners (e.g. clipped by image edge) so they're never picked.
    if valid is not None:
        for j in range(4):
            if not valid[j]:
                dist[:, j] = np.inf
    taken_seeds = set()
    taken_corners = set()
    for _ in range(n):
        flat = np.argmin(dist)
        i, j = divmod(flat, 4)
        if i in taken_seeds or j in taken_corners:
            dist[i, j] = np.inf
            continue
        if dist[i, j] > _CORNER_MAX_DRIFT:
            break   # everything left is too far — abandon the rest
        if dist[i, j] < np.inf:
            refined[seed_start + i] = quad[j]
            refined_mask[seed_start + i] = True
        taken_seeds.add(i)
        taken_corners.add(j)
        dist[i, :] = np.inf
        dist[:, j] = np.inf


def _refine_corners(image_bgr, xy, conf, image_w, image_h):
    """Run cv2.cornerSubPix on confident keypoints that sit inside the image.
    Returns a refined copy of `xy`. Untouched corners (low conf, off-screen,
    or refinement drifted too far) keep their original CNN coordinates."""
    refined = xy.copy()
    if image_bgr is None:
        return refined

    margin = _SUBPIX_WIN[0] + 1
    eligible_idx = []
    seeds = []
    for k in range(len(xy)):
        if conf[k] < KEYPOINT_CONF_THRESHOLD:                   # too uncertain to trust
            continue
        x, y = float(xy[k, 0]), float(xy[k, 1])
        if x < margin or x >= image_w - margin:                 # too close to image edge
            continue
        if y < margin or y >= image_h - margin:
            continue
        eligible_idx.append(k)
        seeds.append([x, y])

    if not seeds:
        return refined

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    seeds_arr = np.array(seeds, dtype=np.float32).reshape(-1, 1, 2)
    cv2.cornerSubPix(gray, seeds_arr, _SUBPIX_WIN, _SUBPIX_ZERO, _SUBPIX_CRIT)
    seeds_arr = seeds_arr.reshape(-1, 2)

    for n, k in enumerate(eligible_idx):
        drift = float(np.linalg.norm(seeds_arr[n] - xy[k]))
        if drift <= _SUBPIX_MAX_DRIFT:
            refined[k] = seeds_arr[n]
        # else: reject the refinement, keep the CNN seed.
    return refined


def detect(image):
    """Run gate detection on a BGR image.
    Returns (detected_gates, annotated_frame) where:
      detected_gates  — list of per-gate dicts (see module docstring)
      annotated_frame — BGR image with boxes + keypoints drawn, or None when
                        RETURN_ANNOTATED_FRAME is False (saves ~5-10ms/frame)."""
    if DEBUG_REFINEMENT:
        _debug_panels.clear()

    if TIME_INFERENCE:
        t_total_start = time.perf_counter()
        torch.cuda.synchronize()
        t_model_start = time.perf_counter()

    yolo_result = _get_model().predict(
        source=image,
        verbose=False,
        device="cuda",
        half=USE_HALF_PRECISION,
    )[0]

    if TIME_INFERENCE:
        torch.cuda.synchronize()
        t_model_end = time.perf_counter()
    yolo_keypoints = yolo_result.keypoints
    image_h, image_w = image.shape[:2]
    if TIME_INFERENCE:
        t_after_unpack = time.perf_counter()
    detected_gates = []

    for gate_index in range(len(yolo_result.boxes)):
        keypoint_pixels  = yolo_keypoints.xy[gate_index].cpu().numpy()    # (N, 2)
        keypoint_confs   = yolo_keypoints.conf[gate_index].cpu().numpy()  # (N,)
        bounding_box     = yolo_result.boxes.xyxy[gate_index].cpu().numpy()
        box_confidence   = float(yolo_result.boxes.conf[gate_index])

        # Refinement strategy (skipped entirely when USE_REFINEMENT is False):
        # 1. Color-mask + contour quadrilateral fit replaces YOLO seeds with
        #    geometrically-correct corners derived from the actual edges in
        #    the image. Best precision when it works.
        # 2. cv2.cornerSubPix as a fallback for keypoints the color mask
        #    couldn't refine (partial mask, gate clipped, color clash).
        if USE_REFINEMENT:
            keypoint_pixels, color_refined_mask = _refine_via_color_mask(
                image, bounding_box, keypoint_pixels, keypoint_confs)
            confs_for_subpix = np.where(color_refined_mask, 0.0, keypoint_confs)
            keypoint_pixels = _refine_corners(
                image, keypoint_pixels, confs_for_subpix, image_w, image_h)

        confident_keypoint_mask = keypoint_confs > KEYPOINT_CONF_THRESHOLD
        n_confident = int(confident_keypoint_mask.sum())

        pose = None
        pose_method = "none"   # "ippe", "partial", or "none"
        if n_confident >= 4:
            # IPPE is the right solver for coplanar points (all 8 gate corners
            # are at z=0 in the gate frame). Faster and more accurate than the
            # default iterative solver for this geometry.
            obj_pts = _GATE_PTS_3D[confident_keypoint_mask]
            img_pts = keypoint_pixels[confident_keypoint_mask]
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, _K, _DIST,
                                          flags=cv2.SOLVEPNP_IPPE)
            if ok:
                # Polish with 1-2 Levenberg-Marquardt iterations on top — tightens
                # reprojection error further, costs ~1ms.
                rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, _K, _DIST,
                                                  rvec, tvec)
            if ok:
                pose = {
                    "rvec":     rvec,
                    "tvec":     tvec,
                    "distance": float(np.linalg.norm(tvec)),
                    "partial":  False,
                    "method":   "ippe",
                }
                pose_method = "ippe"

        detected_gates.append({
            "box":           bounding_box,
            "conf":          box_confidence,
            "keypoints":     keypoint_pixels,
            "keypoint_conf": keypoint_confs,
            "n_confident":   n_confident,
            "pose":          pose,
            "pose_method":   pose_method,
        })

    if TIME_INFERENCE:
        t_after_gate_loop = time.perf_counter()

    if DEBUG_REFINEMENT:
        _compose_and_show_debug_video(image_w, image_h)

    if TIME_INFERENCE:
        t_after_debug = time.perf_counter()

    annotated_frame = yolo_result.plot(line_width=2) if RETURN_ANNOTATED_FRAME else None

    if TIME_INFERENCE:
        t_total_end = time.perf_counter()
        sync_ms       = 1000 * (t_model_start - t_total_start)
        model_ms      = 1000 * (t_model_end - t_model_start)
        unpack_ms     = 1000 * (t_after_unpack - t_model_end)
        gate_loop_ms  = 1000 * (t_after_gate_loop - t_after_unpack)
        debug_ms      = 1000 * (t_after_debug - t_after_gate_loop)
        plot_ms       = 1000 * (t_total_end - t_after_debug)
        total_ms      = 1000 * (t_total_end - t_total_start)
        # print(f"[predict] total={total_ms:5.1f}  sync={sync_ms:4.1f}  model={model_ms:5.1f}  "
        #       f"unpack={unpack_ms:4.1f}  gate_loop={gate_loop_ms:5.1f}  debug={debug_ms:4.1f}  "
        #       f"plot={plot_ms:4.1f}  gates={len(detected_gates)}")

    return detected_gates, annotated_frame
