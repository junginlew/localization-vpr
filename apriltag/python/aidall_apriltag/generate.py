"""Tag image generation and print layout.

Three output forms, sharing one source of truth (the official renderer):

* **NumPy / PNG** — raster, for screens and detection round-trip tests.
* **SVG** — vector, scales to any size; ideal for a single printed tag.
* **PDF sheet** — many tags laid out on a page at exact physical size, with
  quiet zones, ID labels, and optional cut marks; ideal for printing a batch.

The native grid comes from :func:`aidall_apriltag._core.render`, which calls the
upstream ``apriltag_to_image``. Every generated tag is therefore guaranteed
decodable by :class:`aidall_apriltag.Detector`.

Sizing
------
``apriltag_to_image`` returns the full native grid (``total_width`` cells per
side). The *black border* — the edge that the pose ``tag_size`` refers to —
spans ``border_width`` cells, centered inside that grid. When you ask for a
printed ``tag_size_mm`` we size the black border to exactly that, so a generated
tag's printed size matches what you later pass as ``tag_size`` for pose.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

from . import _core
from ._core import Family

FamilyLike = Union[Family, str]

__all__ = [
    "generate",
    "add_quiet_zone",
    "scale",
    "save_png",
    "to_svg",
    "save_svg",
    "save_pdf_sheet",
    "border_size_to_total_mm",
]


def _resolve_family(family: FamilyLike) -> Family:
    if isinstance(family, str):
        fam = _core.family_from_string(family)
        if fam is None:
            raise ValueError(f"unknown AprilTag family: {family!r}")
        return fam
    return family


def generate(family: FamilyLike, tag_id: int) -> np.ndarray:
    """Render a tag to its native cell grid as a 2-D uint8 array (0/255)."""
    return _core.render(_resolve_family(family), tag_id)


def add_quiet_zone(grid: np.ndarray, cells: int = 1) -> np.ndarray:
    """
    Pad a grid with `cells` of white (255) on every side.

    A quiet zone is REQUIRED for reliable detection — never print a tag flush to
    its edge.
    """
    if cells <= 0:
        return grid
    return np.pad(grid, pad_width=cells, mode="constant", constant_values=255)


def scale(grid: np.ndarray, pixels_per_cell: int) -> np.ndarray:
    """
    Nearest-neighbor upscale so each cell becomes a sharp square block.

    Never use interpolation here — blurred edges hurt both print and detection.
    """
    if pixels_per_cell < 1:
        raise ValueError("pixels_per_cell must be >= 1")
    return np.kron(grid, np.ones((pixels_per_cell, pixels_per_cell), dtype=np.uint8))


def border_size_to_total_mm(family: FamilyLike, tag_size_mm: float) -> float:
    """
    Physical edge of the full native grid for a desired black-border size.

    ``tag_size_mm`` is the black-border edge (== pose ``tag_size``). The return
    value is how wide the whole rendered grid prints, before any quiet zone.
    """
    fam = _resolve_family(family)
    total = _core.family_total_width(fam)
    border = _core.family_border_width(fam)
    return tag_size_mm * total / border


# --------------------------------------------------------------------------- #
# PNG
# --------------------------------------------------------------------------- #
def save_png(
    path: str,
    family: FamilyLike,
    tag_id: int,
    *,
    pixels_per_cell: int = 10,
    quiet_zone: int = 1,
) -> np.ndarray:
    """
    Render a tag and write it as a PNG. Returns the pixel array.

    Requires Pillow (``pip install pillow``; included in the ``generate`` extra).
    """
    grid = add_quiet_zone(generate(family, tag_id), quiet_zone)
    pixels = scale(grid, pixels_per_cell)
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "save_png() needs Pillow. Install with: pip install pillow"
        ) from exc
    Image.fromarray(pixels, mode="L").save(path)
    return pixels


# --------------------------------------------------------------------------- #
# SVG (vector, exact physical size)
# --------------------------------------------------------------------------- #
def to_svg(
    family: FamilyLike,
    tag_id: int,
    *,
    tag_size_mm: float = 50.0,
    quiet_zone_cells: int = 1,
    label: bool = True,
) -> str:
    """
    Return an SVG string for a single tag at exact physical size.

    ``tag_size_mm`` sizes the black border (the pose ``tag_size``). A white quiet
    zone of ``quiet_zone_cells`` is added around the grid; an optional ID label
    is printed below the tag (outside the quiet zone, so it never interferes).
    """
    fam = _resolve_family(family)
    grid = generate(fam, tag_id)
    h, w = grid.shape
    border = _core.family_border_width(fam)
    cell_mm = tag_size_mm / border

    qz = quiet_zone_cells
    total_w = (w + 2 * qz) * cell_mm
    total_h = (h + 2 * qz) * cell_mm
    label_mm = cell_mm * 2.0 if label else 0.0
    doc_h = total_h + label_mm

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{total_w:.4f}mm" height="{doc_h:.4f}mm" '
        f'viewBox="0 0 {total_w:.4f} {doc_h:.4f}">',
        f'<rect x="0" y="0" width="{total_w:.4f}" height="{doc_h:.4f}" fill="white"/>',
    ]
    # One black rect per black cell (white is the background).
    for y in range(h):
        for x in range(w):
            if grid[y, x] == 0:
                px = (x + qz) * cell_mm
                py = (y + qz) * cell_mm
                parts.append(
                    f'<rect x="{px:.4f}" y="{py:.4f}" '
                    f'width="{cell_mm:.4f}" height="{cell_mm:.4f}" fill="black"/>'
                )
    if label:
        parts.append(
            f'<text x="{total_w / 2:.4f}" y="{total_h + label_mm * 0.7:.4f}" '
            f'font-family="sans-serif" font-size="{cell_mm * 1.2:.4f}" '
            f'text-anchor="middle" fill="black">'
            f"{_core.family_to_string(fam)} id={tag_id}</text>"
        )
    parts.append("</svg>")
    return "\n".join(parts)


def save_svg(path: str, family: FamilyLike, tag_id: int, **kwargs) -> None:
    """Write a single-tag SVG (see :func:`to_svg` for keyword arguments)."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(to_svg(family, tag_id, **kwargs))


# --------------------------------------------------------------------------- #
# PDF sheet (many tags, exact physical size, for printing)
# --------------------------------------------------------------------------- #
def save_pdf_sheet(
    path: str,
    family: FamilyLike,
    ids: Sequence[int],
    *,
    tag_size_mm: float = 50.0,
    quiet_zone_cells: int = 2,
    margin_mm: float = 10.0,
    spacing_mm: float = 8.0,
    page: str = "A4",
    label: bool = True,
    cut_marks: bool = True,
) -> None:
    """
    Lay out tags on a paged PDF at exact physical size, ready to print.

    Requires reportlab (``pip install reportlab``; included in the ``print``
    extra). Tags flow left-to-right, top-to-bottom, paginating as needed.
    """
    try:
        from reportlab.lib.pagesizes import A4, letter
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "save_pdf_sheet() needs reportlab. Install with: pip install reportlab"
        ) from exc

    fam = _resolve_family(family)
    border = _core.family_border_width(fam)

    pagesize = {"a4": A4, "letter": letter}.get(page.lower())
    if pagesize is None:
        raise ValueError("page must be 'A4' or 'letter'")
    page_w, page_h = pagesize

    c = canvas.Canvas(path, pagesize=pagesize)

    # Precompute one representative geometry (all ids in a family share it).
    sample = generate(fam, ids[0]) if ids else np.zeros((1, 1), np.uint8)
    grid_cells = sample.shape[0]
    cell_mm = tag_size_mm / border
    block_mm = (grid_cells + 2 * quiet_zone_cells) * cell_mm
    label_mm = cell_mm * 2.5 if label else 0.0
    cell_h_mm = block_mm + label_mm

    usable_w = page_w / mm - 2 * margin_mm
    usable_h = page_h / mm - 2 * margin_mm
    cols = max(1, int((usable_w + spacing_mm) // (block_mm + spacing_mm)))
    rows = max(1, int((usable_h + spacing_mm) // (cell_h_mm + spacing_mm)))
    per_page = cols * rows

    def draw_tag(tag_id: int, x_mm: float, y_top_mm: float) -> None:
        grid = generate(fam, tag_id)
        h, w = grid.shape
        # y_top_mm is the top of the block; reportlab origin is bottom-left.
        c.setFillColorRGB(0, 0, 0)
        for gy in range(h):
            for gx in range(w):
                if grid[gy, gx] == 0:
                    px = (x_mm + (gx + quiet_zone_cells) * cell_mm) * mm
                    # rows count downward from the top of the grid
                    py = (
                        page_h / mm - (y_top_mm + (gy + 1 + quiet_zone_cells) * cell_mm)
                    ) * mm
                    c.rect(px, py, cell_mm * mm, cell_mm * mm, stroke=0, fill=1)
        if cut_marks:
            _draw_cut_marks(c, x_mm, y_top_mm, block_mm, page_h, mm)
        if label:
            c.setFont("Helvetica", max(6.0, cell_mm * 3.0))
            ty = (page_h / mm - (y_top_mm + block_mm + label_mm * 0.6)) * mm
            c.drawCentredString(
                (x_mm + block_mm / 2) * mm,
                ty,
                f"{_core.family_to_string(fam)}  id={tag_id}  {tag_size_mm:g}mm",
            )

    for i, tag_id in enumerate(ids):
        slot = i % per_page
        if i > 0 and slot == 0:
            c.showPage()
        col = slot % cols
        row = slot // cols
        x_mm = margin_mm + col * (block_mm + spacing_mm)
        y_top_mm = margin_mm + row * (cell_h_mm + spacing_mm)
        draw_tag(tag_id, x_mm, y_top_mm)

    c.showPage()
    c.save()


def _draw_cut_marks(c, x_mm, y_top_mm, block_mm, page_h, mm) -> None:
    """Short corner ticks just outside the tag block."""
    c.setStrokeColorRGB(0.6, 0.6, 0.6)
    c.setLineWidth(0.3)
    t = 3.0  # tick length, mm
    corners = [
        (x_mm, y_top_mm),
        (x_mm + block_mm, y_top_mm),
        (x_mm, y_top_mm + block_mm),
        (x_mm + block_mm, y_top_mm + block_mm),
    ]
    for cx, cy in corners:
        x = cx * mm
        y = (page_h / mm - cy) * mm
        c.line(x - t * mm, y, x + t * mm, y)
        c.line(x, y - t * mm, x, y + t * mm)
