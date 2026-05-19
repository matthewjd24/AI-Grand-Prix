"""Convert Unity gate-annotation JSON sidecars into YOLO-pose .txt labels.

Before running:
  - PNGs are in   dataset/images/<split>/
  - JSON sidecars are in the SAME folder as their PNGs (copy them there)
This writes        dataset/labels/<split>/<name>.txt   for EVERY image.
Frames with no gates get an empty .txt (negative training examples).

Run:  c:\\python313\\python.exe convert.py
"""

import json
import os
import glob

BASE = r"C:\Users\matt\Desktop\CNN\dataset"

# split name -> (images dir, labels dir).
SPLITS = {
    "train": (os.path.join(BASE, "images", "train"), os.path.join(BASE, "labels", "train")),
    "val":   (os.path.join(BASE, "images", "val"),   os.path.join(BASE, "labels", "val")),
}

CLASS_ID = 0  # single class: gate


def lines_for_json(json_path):
    """Return a list of YOLO-pose label lines for one frame's JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)

    W = data.get("width") or 640
    H = data.get("height") or 360
    out = []

    for gate in data.get("gates", []):
        corners = gate.get("corners", [])
        kpts = []          # (x_norm, y_norm, visibility) per corner, in JSON order
        xs, ys = [], []    # on-screen corner pixels, used for the bbox

        for c in corners:
            if not c.get("present", False):
                kpts.append((0.0, 0.0, 0))      # 0 = not labeled
                continue
            x, y = float(c["x"]), float(c["y"])
            # YOLO visibility: 2 = visible, 1 = labeled but occluded/offscreen
            v = 2 if c.get("visible", False) else 1
            kpts.append((x / W, y / H, v))
            if c.get("onScreen", False):
                xs.append(x)
                ys.append(y)

        # bbox from on-screen corners; fall back to all present corners
        if not xs:
            for c in corners:
                if c.get("present", False):
                    xs.append(float(c["x"]))
                    ys.append(float(c["y"]))
        if not xs:
            continue  # gate has no usable corners

        x0, x1 = max(0.0, min(xs)), min(float(W), max(xs))
        y0, y1 = max(0.0, min(ys)), min(float(H), max(ys))
        bw, bh = (x1 - x0) / W, (y1 - y0) / H
        if bw <= 0 or bh <= 0:
            continue
        cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H

        parts = [CLASS_ID, cx, cy, bw, bh]
        for kx, ky, kv in kpts:
            parts += [kx, ky, kv]
        out.append(" ".join(
            str(p) if isinstance(p, int) else f"{p:.6f}" for p in parts
        ))

    return out


def main():
    for split, (img_dir, lbl_dir) in SPLITS.items():
        if not os.path.isdir(img_dir):
            print(f"[skip] {split}: {img_dir} does not exist")
            continue
        os.makedirs(lbl_dir, exist_ok=True)

        images = glob.glob(os.path.join(img_dir, "*.png"))
        written, empty, missing = 0, 0, 0

        for img in images:
            name = os.path.splitext(os.path.basename(img))[0]
            json_path = os.path.join(img_dir, name + ".json")
            txt_path = os.path.join(lbl_dir, name + ".txt")

            if os.path.exists(json_path):
                lines = lines_for_json(json_path)
            else:
                lines = []
                missing += 1

            with open(txt_path, "w") as f:
                f.write("\n".join(lines))
            written += 1
            if not lines:
                empty += 1

        print(f"[{split}] {written} labels written "
              f"({empty} empty/negative, {missing} images had no JSON)")


if __name__ == "__main__":
    main()
