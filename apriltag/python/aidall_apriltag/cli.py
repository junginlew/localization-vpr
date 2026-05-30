"""Command-line interface for generating and printing AprilTags.

    aidall-apriltag generate --family tag36h11 --id 0 --out tag0.png
    aidall-apriltag generate --family tag36h11 --id 0 --out tag0.svg --tag-size-mm 80
    aidall-apriltag sheet    --family tag36h11 --ids 0-23 --out sheet.pdf --tag-size-mm 50

`generate` writes one tag; the output format is chosen by the file extension
(.png / .svg). `sheet` writes a multi-tag PDF for printing.
"""

from __future__ import annotations

import argparse
from typing import List

from .generate import save_pdf_sheet, save_png, save_svg


def _parse_ids(spec: str) -> List[int]:
    """Parse an id spec like '0-23' or '0,2,5' or '0-3,10,12-14'."""
    ids: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return ids


def _cmd_generate(args: argparse.Namespace) -> None:
    out = args.out
    ext = out.rsplit(".", 1)[-1].lower()
    if ext == "png":
        save_png(
            out,
            args.family,
            args.id,
            pixels_per_cell=args.pixels_per_cell,
            quiet_zone=args.quiet_zone,
        )
    elif ext == "svg":
        save_svg(
            out,
            args.family,
            args.id,
            tag_size_mm=args.tag_size_mm,
            quiet_zone_cells=args.quiet_zone,
            label=not args.no_label,
        )
    else:
        raise SystemExit(f"unsupported output extension: .{ext} (use .png or .svg)")
    print(f"wrote {out}")


def _cmd_sheet(args: argparse.Namespace) -> None:
    ids = _parse_ids(args.ids)
    if not ids:
        raise SystemExit("no ids parsed from --ids")
    save_pdf_sheet(
        args.out,
        args.family,
        ids,
        tag_size_mm=args.tag_size_mm,
        quiet_zone_cells=args.quiet_zone,
        margin_mm=args.margin_mm,
        spacing_mm=args.spacing_mm,
        page=args.page,
        label=not args.no_label,
        cut_marks=not args.no_cut_marks,
    )
    print(f"wrote {args.out} ({len(ids)} tag(s))")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="aidall-apriltag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="write a single tag (.png or .svg)")
    g.add_argument("--family", default="tag36h11")
    g.add_argument("--id", type=int, required=True)
    g.add_argument("--out", required=True, help="output path; .png or .svg")
    g.add_argument("--pixels-per-cell", type=int, default=10, help="PNG only")
    g.add_argument(
        "--tag-size-mm", type=float, default=50.0, help="SVG only; black-border edge"
    )
    g.add_argument("--quiet-zone", type=int, default=1, help="white border, in cells")
    g.add_argument("--no-label", action="store_true", help="SVG only")
    g.set_defaults(func=_cmd_generate)

    s = sub.add_parser("sheet", help="write a multi-tag PDF for printing")
    s.add_argument("--family", default="tag36h11")
    s.add_argument("--ids", required=True, help="e.g. '0-23' or '0,2,5' or '0-3,10'")
    s.add_argument("--out", required=True, help="output PDF path")
    s.add_argument("--tag-size-mm", type=float, default=50.0, help="black-border edge")
    s.add_argument("--quiet-zone", type=int, default=2, help="white border, in cells")
    s.add_argument("--margin-mm", type=float, default=10.0)
    s.add_argument("--spacing-mm", type=float, default=8.0)
    s.add_argument("--page", default="A4", choices=["A4", "letter"])
    s.add_argument("--no-label", action="store_true")
    s.add_argument("--no-cut-marks", action="store_true")
    s.set_defaults(func=_cmd_sheet)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
