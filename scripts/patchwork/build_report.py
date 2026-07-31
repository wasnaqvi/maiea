#!/usr/bin/env python
"""Assemble a Patchwork report PDF from figures and Markdown notes.

Takes an ORDERED list of inputs and concatenates them into one PDF.
``.pdf`` inputs are merged as-is; ``.md`` inputs are typeset to PDF first
(headings, bold/italic/code, links, bullets, and Markdown tables).

    python scripts/patchwork/build_report.py -o ~/Desktop/report.pdf \
        fig1.pdf fig2.pdf notes.md wave1_review.pdf

Order on the command line is the order in the output. Requires pypdf and
reportlab. Text is set in DejaVu Sans (shipped with matplotlib) so Greek,
arrows and the ± / × / ⊕ glyphs used in these tables render properly —
the reportlab default fonts do not cover them.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
from pathlib import Path

# Emoji with no DejaVu coverage -> typographic equivalents.
_GLYPH_FALLBACK = {"✅": "✓", "❌": "✗", "⚠": "▲", "🔴": "●", "🟢": "○"}

_INK = "#1A2F6B"
_MUTED = "#4E5866"
_RULE = "#B9C2CC"


def _register_fonts():
    """DejaVu Sans from matplotlib's bundled fonts; fall back to Helvetica."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    try:
        import matplotlib
        d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        pdfmetrics.registerFont(TTFont("DejaVu", str(d / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold",
                                       str(d / "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Oblique",
                                       str(d / "DejaVuSans-Oblique.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVuMono",
                                       str(d / "DejaVuSansMono.ttf")))
        from reportlab.pdfbase.pdfmetrics import registerFontFamily
        registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                           italic="DejaVu-Oblique", boldItalic="DejaVu-Bold")
        return "DejaVu", "DejaVu-Bold", "DejaVuMono"
    except Exception:
        return "Helvetica", "Helvetica-Bold", "Courier"


def _inline(text: str, mono: str) -> str:
    """Markdown inline spans -> reportlab's mini-HTML."""
    for bad, good in _GLYPH_FALLBACK.items():
        text = text.replace(bad, good)
    # Protect <br> before escaping, restore after.
    text = re.sub(r"<br\s*/?>", "\x00BR\x00", text)
    text = html.escape(text, quote=False)
    text = text.replace("\x00BR\x00", "<br/>")
    text = re.sub(r"`([^`]+)`", rf'<font face="{mono}" size="8">\1</font>', text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                  r'<link href="\2" color="#0072B2">\1</link>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![*\w])\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", text)
    return text


def markdown_to_pdf(md_path: Path, out_path: Path) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (HRFlowable, KeepTogether, PageBreak,
                                    Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)

    base, bold, mono = _register_fonts()
    margin = 0.55 * inch
    avail = letter[0] - 2 * margin

    st = {
        "h1": ParagraphStyle("h1", fontName=bold, fontSize=17, leading=21,
                             textColor=colors.HexColor(_INK), spaceAfter=8),
        "h2": ParagraphStyle("h2", fontName=bold, fontSize=13, leading=16,
                             textColor=colors.HexColor(_INK),
                             spaceBefore=14, spaceAfter=6),
        "h3": ParagraphStyle("h3", fontName=bold, fontSize=11, leading=14,
                             textColor=colors.HexColor(_INK),
                             spaceBefore=10, spaceAfter=4),
        "p": ParagraphStyle("p", fontName=base, fontSize=9, leading=13,
                            alignment=TA_LEFT, spaceAfter=6),
        "li": ParagraphStyle("li", fontName=base, fontSize=9, leading=13,
                             leftIndent=14, bulletIndent=4, spaceAfter=3),
        "cell": ParagraphStyle("cell", fontName=base, fontSize=6.4,
                               leading=8.0),
        "cellh": ParagraphStyle("cellh", fontName=bold, fontSize=6.4,
                                leading=8.0, textColor=colors.white),
    }

    flow = []
    lines = md_path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # ---- table ----
        if stripped.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            rows = []
            for r in block:
                cells = [c.strip() for c in r.strip("|").split("|")]
                if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    continue  # separator row
                rows.append(cells)
            if not rows:
                continue
            ncol = max(len(r) for r in rows)
            rows = [r + [""] * (ncol - len(r)) for r in rows]

            # Column widths from content length, clamped so no column
            # collapses and none dominates.
            weights = []
            for c in range(ncol):
                longest = max(len(re.sub(r"<[^>]+>|\*\*|`", "", r[c]))
                              for r in rows)
                weights.append(min(max(longest, 8), 34))
            total = sum(weights)
            widths = [avail * w / total for w in weights]

            data = [[Paragraph(_inline(c, mono),
                               st["cellh"] if ri == 0 else st["cell"])
                     for c in row] for ri, row in enumerate(rows)]
            tbl = Table(data, colWidths=widths, repeatRows=1)
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_INK)),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(_RULE)),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#F4F6F9")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            flow += [Spacer(1, 4), tbl, Spacer(1, 8)]
            continue

        # ---- headings / rules / lists / paragraphs ----
        if stripped.startswith("### "):
            flow.append(Paragraph(_inline(stripped[4:], mono), st["h3"]))
        elif stripped.startswith("## "):
            flow.append(Paragraph(_inline(stripped[3:], mono), st["h2"]))
        elif stripped.startswith("# "):
            flow.append(Paragraph(_inline(stripped[2:], mono), st["h1"]))
            flow.append(HRFlowable(width="100%", thickness=0.8,
                                   color=colors.HexColor(_RULE),
                                   spaceAfter=8))
        elif re.fullmatch(r"-{3,}", stripped):
            flow.append(HRFlowable(width="100%", thickness=0.6,
                                   color=colors.HexColor(_RULE),
                                   spaceBefore=6, spaceAfter=6))
        elif re.match(r"^[-*]\s+", stripped) or re.match(r"^\d+\.\s+", stripped):
            body = re.sub(r"^([-*]|\d+\.)\s+", "", stripped)
            # absorb continuation lines
            while (i + 1 < len(lines) and lines[i + 1].startswith("  ")
                   and lines[i + 1].strip()
                   and not re.match(r"^\s*([-*]|\d+\.)\s", lines[i + 1])):
                i += 1
                body += " " + lines[i].strip()
            bullet = "•" if not re.match(r"^\d+\.", stripped) else \
                stripped.split(".", 1)[0] + "."
            flow.append(Paragraph(_inline(body, mono), st["li"],
                                  bulletText=bullet))
        else:
            body = stripped
            # A continuation line only ends the paragraph if it starts a
            # NEW block. '*' or '-' must be followed by a space to be a
            # bullet -- otherwise a wrapped line beginning with *italic*
            # markup gets orphaned into its own paragraph.
            def _starts_block(s: str) -> bool:
                s = s.strip()
                return (s.startswith(("|", "#"))
                        or bool(re.match(r"^([-*]\s+|\d+\.\s+)", s))
                        or bool(re.fullmatch(r"-{3,}", s)))

            while (i + 1 < len(lines) and lines[i + 1].strip()
                   and not _starts_block(lines[i + 1])):
                i += 1
                body += " " + lines[i].strip()
            flow.append(Paragraph(_inline(body, mono), st["p"]))
        i += 1

    doc = SimpleDocTemplate(str(out_path), pagesize=letter,
                            leftMargin=margin, rightMargin=margin,
                            topMargin=0.55 * inch, bottomMargin=0.55 * inch,
                            title=md_path.stem)
    doc.build(flow)
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Concatenate figures and Markdown notes into one PDF.")
    ap.add_argument("inputs", nargs="+",
                    help="PDF and/or Markdown files, in output order.")
    ap.add_argument("-o", "--output", required=True, help="Output PDF.")
    args = ap.parse_args(argv)

    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        print("pypdf is required:  pip install pypdf")
        return 1

    out_path = Path(os.path.expanduser(args.output)).resolve()
    tmp_dir = out_path.parent / ".report_build"
    tmp_dir.mkdir(exist_ok=True)

    writer = PdfWriter()
    page_no = 0
    for raw in args.inputs:
        p = Path(os.path.expanduser(raw)).resolve()
        if not p.is_file():
            print(f"  MISSING  {p}")
            continue
        if p.suffix.lower() in (".md", ".markdown"):
            rendered = markdown_to_pdf(p, tmp_dir / f"{p.stem}.pdf")
            src = rendered
            kind = "md->pdf"
        else:
            src = p
            kind = "pdf"
        reader = PdfReader(str(src))
        writer.add_outline_item(p.stem.replace("_", " "), page_no)
        for page in reader.pages:
            writer.add_page(page)
            page_no += 1
        print(f"  {kind:8s} {len(reader.pages):3d}p  {p.name}")

    with out_path.open("wb") as fh:
        writer.write(fh)
    print(f"\nWrote {out_path}  ({page_no} pages, "
          f"{out_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
