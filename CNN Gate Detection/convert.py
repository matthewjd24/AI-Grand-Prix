"""Convert Unity gate-annotation JSON sidecars into YOLO-pose .txt labels.

Layout (matches data.yaml):
  - PNGs and JSON sidecars live together in the *Images folders below.
  - This script writes a .txt next to every PNG. YOLO finds same-folder labels.

Run:  c:\\python313\\python.exe convert.py
"""

import json
import os
import glob

BASE = r"C:\Users\matt\Documents\GitHub\AI-Grand-Prix\Unity"

# split name -> image folder. Labels are written into the same folder.
SPLITS = {
    "train": os.path.join(BASE, "TrainingImages"),
    "val":   os.path.join(BASE, "ValidationImages"),
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
    for split, img_dir in SPLITS.items():
        if not os.path.isdir(img_dir):
            print(f"[skip] {split}: {img_dir} does not exist")
            continue

        images = glob.glob(os.path.join(img_dir, "*.png"))

        # Skip the whole split if every PNG already has a .txt next to it.
        # Cheap idempotency: lets test.py call convert.main() unconditionally.
        if images and all(os.path.exists(os.path.splitext(p)[0] + ".txt") for p in images):
            print(f"[skip] {split}: all {len(images)} labels up to date")
            continue

        written, empty, missing = 0, 0, 0

        for img in images:
            name = os.path.splitext(os.path.basename(img))[0]
            json_path = os.path.join(img_dir, name + ".json")
            txt_path  = os.path.join(img_dir, name + ".txt")

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

        print(f"[{split}] {written} labels written into {img_dir} "
              f"({empty} empty/negative, {missing} images had no JSON)")


if __name__ == "__main__":
    main()
