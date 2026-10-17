import html
import re
from io import BytesIO
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    LongTable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)
import reportlab


def _register_fonts() -> tuple[str, str]:
    fonts_dir = Path(reportlab.__file__).parent / "fonts"
    regular = fonts_dir / "Vera.ttf"
    bold = fonts_dir / "VeraBd.ttf"
    if not regular.is_file() or not bold.is_file():
        raise RuntimeError("ReportLab's bundled Unicode fonts are unavailable.")
    if "BotanicalVera" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("BotanicalVera", str(regular)))
    if "BotanicalVera-Bold" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("BotanicalVera-Bold", str(bold)))
    return "BotanicalVera", "BotanicalVera-Bold"


def _inline_markup(value: str) -> str:
    escaped = html.escape(value, quote=False)
    escaped = re.sub(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
        lambda match: (
            f'<link href="{html.escape(match.group(2), quote=True)}" '
            'color="#287548">'
            f"{match.group(1)}</link>"
        ),
        escaped,
    )
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)
    return escaped.replace("\n", "<br/>")


def _make_styles() -> dict[str, ParagraphStyle]:
    regular_font, bold_font = _register_fonts()
    sample = getSampleStyleSheet()
    body = ParagraphStyle(
        "BotanicalBody",
        parent=sample["BodyText"],
        fontName=regular_font,
        fontSize=9.5,
        leading=14,
        spaceAfter=6,
        wordWrap="CJK",
    )
    return {
        "title": ParagraphStyle(
            "BotanicalTitle",
            parent=body,
            fontName=bold_font,
            fontSize=21,
            leading=27,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#264b35"),
            spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "BotanicalH1",
            parent=body,
            fontName=bold_font,
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#264b35"),
            spaceBefore=14,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "BotanicalH2",
            parent=body,
            fontName=bold_font,
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#356447"),
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "BotanicalH3",
            parent=body,
            fontName=bold_font,
            fontSize=10.5,
            leading=14,
            textColor=colors.HexColor("#356447"),
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": body,
        "bullet": ParagraphStyle(
            "BotanicalBullet",
            parent=body,
            leftIndent=14,
            firstLineIndent=-8,
            spaceAfter=4,
        ),
        "table": ParagraphStyle(
            "BotanicalTable",
            parent=body,
            fontSize=8,
            leading=11,
            spaceAfter=0,
        ),
        "table_header": ParagraphStyle(
            "BotanicalTableHeader",
            parent=body,
            fontName=bold_font,
            fontSize=8,
            leading=11,
            textColor=colors.white,
            spaceAfter=0,
        ),
    }


def _table_cells(line: str, style: ParagraphStyle) -> list[Paragraph]:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return [Paragraph(_inline_markup(cell), style) for cell in cells]


def _render_table(
    lines: list[str],
    styles: dict[str, ParagraphStyle],
    available_width: float,
) -> LongTable:
    body_lines = lines
    if len(lines) > 1 and all(
        re.fullmatch(r"\s*:?-{3,}:?\s*", cell)
        for cell in lines[1].strip().strip("|").split("|")
    ):
        body_lines = [lines[0], *lines[2:]]

    data: list[list[Paragraph]] = []
    for index, line in enumerate(body_lines):
        style = styles["table_header"] if index == 0 else styles["table"]
        data.append(_table_cells(line, style))

    if not data:
        return LongTable([[""]], colWidths=[available_width])
    column_count = max(len(row) for row in data)
    for row in data:
        row.extend([Paragraph("", styles["table"])] * (column_count - len(row)))

    table = LongTable(
        data,
        colWidths=[available_width / column_count] * column_count,
        repeatRows=1,
        splitByRow=1,
        hAlign="LEFT",
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#356447")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c6d4c8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white,
            colors.HexColor("#f3f7f1"),
        ]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def generate_report_pdf(
    report_markdown: str,
    title: str = "AI-generated botanical literature review",
    notice: str = (
        "AI-generated literature review — not a peer-reviewed publication"
    ),
) -> bytes:
    """Render a readable, multi-page PDF from report Markdown."""
    if not report_markdown.strip():
        raise ValueError("The research report is empty.")

    styles = _make_styles()
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=19 * mm,
        leftMargin=19 * mm,
        topMargin=22 * mm,
        bottomMargin=20 * mm,
        title=title,
        author="Botanical Research Assistant",
        subject="AI-generated literature review",
    )
    available_width = A4[0] - document.leftMargin - document.rightMargin
    story = [
        Paragraph(_inline_markup(title), styles["title"]),
        Paragraph(_inline_markup(notice), styles["body"]),
        Spacer(1, 6),
    ]

    lines = report_markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue

        if line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            story.append(_render_table(table_lines, styles, available_width))
            story.append(Spacer(1, 8))
            continue

        if line.startswith("#"):
            heading_level = min(len(line) - len(line.lstrip("#")), 3)
            heading_text = line.lstrip("#").strip()
            if heading_text.casefold() in {
                title.casefold(),
                "botanical research report",
            }:
                index += 1
                continue
            style = styles[f"h{heading_level}"]
            story.append(Paragraph(_inline_markup(heading_text), style))
        elif re.match(r"^[-*]\s+", line):
            story.append(Paragraph(
                "&#8226; " + _inline_markup(re.sub(r"^[-*]\s+", "", line)),
                styles["bullet"],
            ))
        elif re.match(r"^\d+[.)]\s+", line):
            story.append(Paragraph(
                _inline_markup(line),
                styles["bullet"],
            ))
        elif line == "---":
            story.append(Spacer(1, 10))
        else:
            story.append(Paragraph(_inline_markup(line), styles["body"]))
        index += 1

    def draw_page(canvas: Any, doc: SimpleDocTemplate) -> None:
        canvas.saveState()
        canvas.setFont("BotanicalVera", 8)
        canvas.setFillColor(colors.HexColor("#66776a"))
        canvas.drawString(doc.leftMargin, 12 * mm, "Botanical Research Assistant")
        canvas.drawRightString(
            A4[0] - doc.rightMargin,
            12 * mm,
            f"Page {doc.page}",
        )
        canvas.restoreState()

    document.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    pdf_data = buffer.getvalue()
    if not pdf_data.startswith(b"%PDF-"):
        raise RuntimeError("PDF generation returned an invalid document.")
    return pdf_data
