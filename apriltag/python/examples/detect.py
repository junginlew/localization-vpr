"""
Minimal Python example: detect AprilTags in an image file.

python detect.py path/to/image.png
python detect.py path/to/image.png --tag-size 0.16 --fx 600 --fy 600 --cx 320 --cy 240
"""

from __future__ import annotations

import argparse

import aidall_apriltag as at
import cv2  # only used for image loading / preview
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--family", default="tag36h11")
    parser.add_argument("--tag-size", type=float, default=0.0, help="meters")
    parser.add_argument("--fx", type=float, default=0.0)
    parser.add_argument("--fy", type=float, default=0.0)
    parser.add_argument("--cx", type=float, default=0.0)
    parser.add_argument("--cy", type=float, default=0.0)
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"failed to read image: {args.image}")

    camera = None
    if args.fx and args.fy:
        camera = at.CameraParams(args.fx, args.fy, args.cx, args.cy)

    detector = at.Detector(
        family=args.family,
        tag_size=args.tag_size,
        camera=camera,
    )
    detections = detector.detect(image)
    print(f"found {len(detections)} tag(s)")

    for d in detections:
        line = f"  id={d.id} hamming={d.hamming} margin={d.decision_margin:.1f}"
        if d.pose is not None:
            t = d.pose.t
            line += f" t=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) m"
        print(line)

        corners = np.array(d.corners, dtype=np.int32)
        cv2.polylines(image, [corners], True, (0, 255, 0), 2)
        cv2.putText(
            image,
            str(d.id),
            (int(d.center[0]), int(d.center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    cv2.imwrite("detections.png", image)
    print("wrote detections.png")


if __name__ == "__main__":
    main()
