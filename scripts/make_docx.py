#!/usr/bin/env python3
"""Convert a project markdown document to a styled .docx.

Pandoc does the structural conversion (headings, tables, code blocks, inline
formatting); this script applies the document styling, so the output matches the
typography of the thesis documents rather than pandoc's defaults.

    python scripts/make_docx.py docs/AGENTS_AND_COMMUNICATION.md
    python scripts/make_docx.py docs/ARCHITECTURE.md -o dist/architecture.docx

Requires: pandoc, python-docx.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Emu, Mm, Pt, RGBColor

REPO_ROOT = Path(__file__).resolve().parents[1]

_VML_NS = "urn:schemas-microsoft-com:vml"

BODY_FONT = "Times New Roman"
MONO_FONT = "Consolas"
BODY_SIZE = Pt(11)
ACCENT = RGBColor(0x1F, 0x3B, 0x5C)      # heading colour
HEADER_FILL = "E8EEF7"                   # table header shading
MONO_SIZE = Pt(8.5)


def _set_font(run_or_style_font, name: str) -> None:
    """Set a font for every script range, not just the Latin one.

    python-docx's ``font.name`` only writes ``w:ascii``. Word falls back to its
    own default for anything else, which is why a document can look right in one
    renderer and wrong in another.
    """
    run_or_style_font.name = name
    element = run_or_style_font.element
    rpr = element.get_or_add_rPr() if hasattr(element, "get_or_add_rPr") else element.rPr
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), name)


#: Child order of <w:tblPr>, per ECMA-376. The schema enforces this sequence, so
#: appending an element to the end of tblPr produces a file Word rejects.
_TBLPR_ORDER = (
    "tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize",
    "tblStyleColBandSize", "tblW", "jc", "tblCellSpacing", "tblInd", "tblBorders",
    "shd", "tblLayout", "tblCellMar", "tblLook", "tblCaption", "tblDescription",
)


def _set_tblpr_child(tbl_pr, tag: str, attrs: dict[str, str]):
    """Replace (or create) a <w:tblPr> child, keeping the schema's child order."""
    qualified = qn(f"w:{tag}")
    existing = tbl_pr.find(qualified)
    if existing is not None:
        tbl_pr.remove(existing)

    node = tbl_pr.makeelement(qualified, {})
    for key, value in attrs.items():
        node.set(qn(key), value)

    index = _TBLPR_ORDER.index(tag)
    for child in tbl_pr:
        child_tag = child.tag.split("}")[-1]
        if child_tag in _TBLPR_ORDER and _TBLPR_ORDER.index(child_tag) > index:
            child.addprevious(node)
            return node
    tbl_pr.append(node)
    return node


def _shade(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.makeelement(qn("w:shd"), {})
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def _strip_horizontal_rules(doc) -> int:
    """Remove pandoc's ``---`` rules.

    Pandoc renders a thematic break as a VML ``<v:rect o:hr>``, which Word and
    macOS render as a heavy black bar across the page. In a document where every
    section already carries a numbered heading, the rules add noise rather than
    structure, so they are dropped.
    """
    removed = 0
    for paragraph in list(doc.paragraphs):
        if paragraph._p.findall(f".//{{{_VML_NS}}}rect"):
            paragraph._p.getparent().remove(paragraph._p)
            removed += 1
    return removed


#: A4 with 2 cm margins, matching the thesis documents in the companion repo.
#: Pandoc leaves the page geometry unset, which makes the usable width depend on
#: whichever renderer opens the file — so it is pinned explicitly here.
PAGE_WIDTH = Mm(210)
PAGE_HEIGHT = Mm(297)
MARGIN = Mm(20)


def _set_page_geometry(doc) -> int:
    """Pin the page to A4 with 2 cm margins; return the usable width in twips."""
    for section in doc.sections:
        section.page_width = PAGE_WIDTH
        section.page_height = PAGE_HEIGHT
        section.left_margin = MARGIN
        section.right_margin = MARGIN
        section.top_margin = MARGIN
        section.bottom_margin = MARGIN
        # pandoc writes no <w:pgMar>; python-docx then creates a partial one.
        # header/footer/gutter are schema-required attributes of that element.
        section.header_distance = Mm(12)
        section.footer_distance = Mm(12)
        section.gutter = 0
    usable = int(PAGE_WIDTH - 2 * MARGIN)
    return usable // 635          # EMU -> twips


def _fit_table(table, total_width: int) -> None:
    """Make the table span the text column, with columns sized to their content.

    Pandoc gives every column an identical width, which leaves a narrow index
    column wrapping its contents while a wide prose column has room to spare, and
    the table itself stopping short of the right margin.
    """
    columns = len(table.columns)
    if not columns:
        return

    # weight each column by the longest cell it has to hold
    weights = []
    for c in range(columns):
        longest = 1
        for row in table.rows:
            try:
                longest = max(longest, len(row.cells[c].text.strip()))
            except IndexError:
                continue
        weights.append(min(longest, 60) ** 0.75)   # damped: avoid one column dominating

    total_weight = sum(weights) or 1.0
    min_width = int(total_width * 0.055)
    widths = [max(min_width, int(total_width * w / total_weight)) for w in weights]
    # correct rounding drift so the columns sum to exactly the text width
    widths[-1] += total_width - sum(widths)

    tbl_pr = table._tbl.tblPr
    _set_tblpr_child(tbl_pr, "tblW", {"w:type": "dxa", "w:w": str(total_width)})
    _set_tblpr_child(tbl_pr, "tblLayout", {"w:type": "fixed"})

    grid = table._tbl.find(qn("w:tblGrid"))
    if grid is not None:
        for col, width in zip(grid.findall(qn("w:gridCol")), widths):
            col.set(qn("w:w"), str(width))
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = Emu(width * 635)      # 1 twip = 635 EMU


def _set_cell_margins(table, *, left: int = 110, right: int = 110,
                      top: int = 50, bottom: int = 50) -> None:
    """Give cells breathing room.

    Pandoc emits no cell margins, so adjacent columns touch — a narrow index
    column reads as part of the text beside it ("0 OrchestratorAgent").
    """
    cell_mar = _set_tblpr_child(table._tbl.tblPr, "tblCellMar", {})
    for tag, value in (("w:top", top), ("w:left", left),
                       ("w:bottom", bottom), ("w:right", right)):
        node = cell_mar.makeelement(qn(tag), {})
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")
        cell_mar.append(node)


def _repeat_header(row) -> None:
    """Mark a table row as a header so it repeats across page breaks."""
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = tr_pr.makeelement(qn("w:tblHeader"), {})
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def convert(md_path: Path, out_path: Path, *, title: str | None = None,
            subtitle: str | None = None) -> Path:
    if not shutil.which("pandoc"):
        raise SystemExit("pandoc is required: brew install pandoc")

    with tempfile.TemporaryDirectory() as tmp:
        source = md_path
        if title:
            # the document gets a styled title block, so the markdown's own H1
            # would otherwise appear twice
            text = md_path.read_text(encoding="utf-8")
            lines = text.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("# "):
                    del lines[i]
                    while i < len(lines) and not lines[i].strip():
                        del lines[i]
                    break
            source = Path(tmp) / md_path.name
            source.write_text("\n".join(lines), encoding="utf-8")

        raw = Path(tmp) / "raw.docx"
        subprocess.run(
            ["pandoc", str(source), "-o", str(raw), "--from=gfm"],
            check=True, capture_output=True,
        )
        doc = Document(str(raw))

        # --- base typography ------------------------------------------- #
        normal = doc.styles["Normal"]
        _set_font(normal.font, BODY_FONT)
        normal.font.size = BODY_SIZE
        normal.paragraph_format.space_after = Pt(6)
        normal.paragraph_format.line_spacing = 1.15

        for name in ("Body Text", "First Paragraph", "Compact"):
            if name in (s.name for s in doc.styles):
                style = doc.styles[name]
                _set_font(style.font, BODY_FONT)
                style.font.size = BODY_SIZE

        # --- headings ---------------------------------------------------- #
        for level, size, space_before in ((1, 16, 18), (2, 13, 14), (3, 11.5, 12)):
            name = f"Heading {level}"
            if name not in (s.name for s in doc.styles):
                continue
            style = doc.styles[name]
            _set_font(style.font, BODY_FONT)
            style.font.size = Pt(size)
            style.font.bold = True
            style.font.color.rgb = ACCENT
            style.paragraph_format.space_before = Pt(space_before)
            style.paragraph_format.space_after = Pt(6)
            style.paragraph_format.keep_with_next = True

        # --- code blocks -------------------------------------------------- #
        for name in ("Source Code", "Verbatim Char", "Code"):
            if name in (s.name for s in doc.styles):
                style = doc.styles[name]
                _set_font(style.font, MONO_FONT)
                style.font.size = MONO_SIZE

        _strip_horizontal_rules(doc)

        # --- tables ------------------------------------------------------- #
        text_width = _set_page_geometry(doc)
        for table in doc.tables:
            try:
                table.style = doc.styles["Table Grid"]
            except KeyError:
                pass
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            # autofit must stay off: it resets the table layout to "auto", which
            # discards the fixed column widths set below and lets narrow tables
            # shrink to their content instead of spanning the text column
            table.autofit = False
            _fit_table(table, text_width)
            _set_cell_margins(table)
            for r, row in enumerate(table.rows):
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        paragraph.paragraph_format.space_after = Pt(2)
                        paragraph.paragraph_format.space_before = Pt(2)
                        for run in paragraph.runs:
                            _set_font(run.font, BODY_FONT)
                            run.font.size = Pt(9.5)
                            if r == 0:
                                run.font.bold = True
                if r == 0:
                    _repeat_header(row)
                    for cell in row.cells:
                        _shade(cell, HEADER_FILL)

        # --- title block --------------------------------------------------- #
        if title:
            # built in reverse: each insert goes above the previous one
            if subtitle:
                sub = doc.paragraphs[0].insert_paragraph_before(subtitle)
                sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in sub.runs:
                    _set_font(run.font, BODY_FONT)
                    run.font.size = Pt(10.5)
                    run.font.italic = True
                    run.font.color.rgb = RGBColor(0x55, 0x5F, 0x6B)
                sub.paragraph_format.space_after = Pt(20)
            head = doc.paragraphs[0].insert_paragraph_before(title)
            head.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in head.runs:
                _set_font(run.font, BODY_FONT)
                run.font.size = Pt(19)
                run.font.bold = True
                run.font.color.rgb = ACCENT
            head.paragraph_format.space_after = Pt(4)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(out_path))
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("markdown", help="source .md file")
    parser.add_argument("-o", "--output", help="destination .docx (default: dist/<name>.docx)")
    parser.add_argument("--title", help="title to place at the top of the document")
    parser.add_argument("--subtitle", help="subtitle line under the title")
    args = parser.parse_args()

    md_path = Path(args.markdown)
    if not md_path.exists():
        print(f"not found: {md_path}", file=sys.stderr)
        return 1
    out = Path(args.output) if args.output else REPO_ROOT / "dist" / f"{md_path.stem}.docx"

    result = convert(md_path, out, title=args.title, subtitle=args.subtitle)
    print(f"  -> {result}  ({result.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
