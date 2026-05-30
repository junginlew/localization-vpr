"""
Example: generate AprilTag images in every supported format.

python generate.py            # writes tag0.png, tag0.svg, sheet.pdf
"""

from __future__ import annotations

import aidall_apriltag as at


def main() -> None:
    family = "tag36h11"

    # 1) PNG (raster) — good for screens and detection tests.
    at.save_png("tag0.png", family, 0, pixels_per_cell=20, quiet_zone=1)
    print("wrote tag0.png")

    # 2) SVG (vector) — exact physical size for a single printed tag.
    at.save_svg("tag0.svg", family, 0, tag_size_mm=80.0, quiet_zone_cells=1)
    print("wrote tag0.svg")

    # 3) PDF sheet — many tags at exact size, ready to print. Needs reportlab.
    try:
        at.save_pdf_sheet("sheet.pdf", family, ids=range(0, 24), tag_size_mm=50.0)
        print("wrote sheet.pdf")
    except ImportError as exc:
        print(f"skipping PDF sheet: {exc}")


if __name__ == "__main__":
    main()
