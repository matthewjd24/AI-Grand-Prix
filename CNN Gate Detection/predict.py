"""Step through validation images one at a time: predict, show, wait for a key.

Controls (focus the image window):
  any key  -> next image
  q / Esc  -> quit
"""

import glob
import os
import cv2
from ultralytics import YOLO

WEIGHTS = r"runs\pose\runs\gate-pose\weights\best.pt"
VAL_DIR = r"dataset\images\val"

if __name__ == "__main__":
    model = YOLO(WEIGHTS)
    images = sorted(glob.glob(os.path.join(VAL_DIR, "*.png")))
    print(f"{len(images)} validation images. Press a key for next, q/Esc to quit.")

    for i, img in enumerate(images):
        result = model.predict(source=img, verbose=False)[0]
        annotated = result.plot(line_width=2)  # numpy image with boxes + keypoints

        title = f"[{i+1}/{len(images)}] {os.path.basename(img)} - {len(result.boxes)} gate(s)"
        cv2.imshow("prediction", annotated)
        cv2.setWindowTitle("prediction", title)
        print(title)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):  # q or Esc
            break

    cv2.destroyAllWindows()
