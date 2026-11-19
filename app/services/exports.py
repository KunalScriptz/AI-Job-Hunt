from __future__ import annotations

import logging
import re
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from app.services.resume_format import (
    Role,
    body_text,
    flatten_links,
    parse_resume,
    to_plain,
    unbold,
)
from app.settings import get_settings

log = logging.getLogger("jobhunt.exports")

MARGIN = 48
BODY, BOLD = "Helvetica", "Helvetica-Bold"
INK, MUTED, RULE = (0.07, 0.08, 0.15), (0.35, 0.38, 0.45), (0.8, 0.82, 0.87)


def _runs(text: str) -> list[tuple[str, bool]]:
    """Split Markdown emphasis into (text, bold) runs: "**Data:** Kafka" -> [("Data:", True), (" Kafka", False)]."""
    return [(part, index % 2 == 1) for index, part in enumerate((text or "").split("**")) if part] or [("", False)]


def _segments(row: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Words of one wrapped line grouped into the fewest possible (text, font) runs."""
    out: list[tuple[str, str]] = []
    for word, word_font in row:
        if out and out[-1][1] == word_font:
            out[-1] = (f"{out[-1][0]} {word}", word_font)
        else:
            out.append((word, word_font))
    return out


def markdown_to_simple_pdf(markdown: str) -> bytes:
    """Render the resume Markdown dialect (# / ## / ### Title | Company | Location | Dates / - bullets)."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    usable = width - 2 * MARGIN
    y = height - MARGIN
    after_name = False

    def room(needed: float) -> None:
        nonlocal y
        if y - needed < MARGIN:
            c.showPage()
            y = height - MARGIN

    def paragraph(text: str, font: str = BODY, size: float = 9.5, indent: float = 0, color=INK,
                  leading: float | None = None, bullet: bool = False) -> None:
        """One wrapped paragraph. **bold** spans are drawn bold, so skill groups and labels keep their emphasis."""
        nonlocal y
        leading = leading or size * 1.35
        bold = BOLD if font == BODY else font
        space = stringWidth(" ", font, size)
        limit = usable - indent
        rows: list[list[tuple[str, str]]] = [[]]
        used = 0.0
        for chunk, is_bold in _runs(text):
            for word in chunk.split():
                word_font = bold if is_bold else font
                word_width = stringWidth(word, word_font, size)
                if rows[-1] and used + space + word_width > limit:
                    rows.append([])
                    used = 0.0
                used += (space if rows[-1] else 0) + word_width
                rows[-1].append((word, word_font))
        for i, row in enumerate(rows):
            room(leading)
            c.setFillColorRGB(*color)
            if bullet and i == 0:
                c.setFont(font, size)
                c.drawString(MARGIN + indent - 9, y - size, "\u2022")
            x = MARGIN + indent
            # Consecutive words in the same font are drawn as one string, so the PDF still extracts as real
            # sentences (ATS parsers and pypdf read it the way a person does).
            for segment, segment_font in _segments(row):
                c.setFont(segment_font, size)
                c.drawString(x, y - size, segment)
                x += stringWidth(segment, segment_font, size) + space
            y -= leading

    for raw in body_text(markdown).splitlines():
        line = flatten_links(to_plain(raw.rstrip()))
        stripped = line.strip()
        if not stripped:
            y -= 4
            continue
        if stripped.startswith("### "):
            parts = [p.strip() for p in unbold(stripped[4:]).split("|")]
            dates = parts[-1] if len(parts) > 2 else ""
            title = parts[0]
            rest = " | ".join(parts[1:-1] if dates else parts[1:])
            y -= 3
            size = 10.5
            left_width = stringWidth(title, BOLD, size) + (stringWidth(f"  {rest}", BODY, 9.5) if rest else 0)
            dates_width = stringWidth(dates, BODY, 9.5) if dates else 0
            if left_width + dates_width + 12 <= usable:
                room(size * 1.4)
                c.setFillColorRGB(*INK)
                c.setFont(BOLD, size)
                c.drawString(MARGIN, y - size, title)
                if rest:
                    c.setFont(BODY, 9.5)
                    c.setFillColorRGB(*MUTED)
                    c.drawString(MARGIN + stringWidth(title, BOLD, size), y - size, f"  {rest}")
                if dates:
                    c.setFont(BODY, 9.5)
                    c.setFillColorRGB(*MUTED)
                    c.drawRightString(width - MARGIN, y - size, dates)
                y -= size * 1.4
            else:
                paragraph(stripped[4:], BOLD, size)
            continue
        if stripped.startswith("## "):
            y -= 8
            room(26)
            paragraph(unbold(stripped[3:]).upper(), BOLD, 10.5, color=INK, leading=14)
            c.setStrokeColorRGB(*RULE)
            c.setLineWidth(0.6)
            c.line(MARGIN, y + 2, width - MARGIN, y + 2)
            y -= 4
            after_name = False
            continue
        if stripped.startswith("# "):
            paragraph(unbold(stripped[2:]), BOLD, 18, leading=24)
            after_name = True
            continue
        if stripped.startswith("- "):
            paragraph(stripped[2:], indent=12, bullet=True)
            continue
        paragraph(stripped, color=MUTED if after_name else INK)
    c.save()
    buffer.seek(0)
    return buffer.read()


# ───────────── LaTeX rendering (optional, settings.latex_render_url) ─────────────

_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&",
    "%": r"\%", "#": r"\#", "_": r"\_", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}
_LATEX_SPECIAL_RE = re.compile("|".join(re.escape(c) for c in _LATEX_SPECIAL))


def _latex_escape(text: str) -> str:
    """Every one of LaTeX's special characters, mapped to its safe form. Runs in one pass over the original text,
    so a replacement (itself containing `\\`/`{`/`}`) is never re-scanned and double-escaped."""
    return _LATEX_SPECIAL_RE.sub(lambda m: _LATEX_SPECIAL[m.group()], text or "")


def _tex_text(text: str) -> str:
    """Markdown **bold** spans as `\\textbf{...}`. Escaping runs on each run's own raw text before it is wrapped,
    never after — otherwise the wrapper's own backslash and braces would be escaped too."""
    return "".join(f"\\textbf{{{_latex_escape(part)}}}" if bold else _latex_escape(part)
                   for part, bold in _runs(text))


_LATEX_PREAMBLE = "\n".join([
    r"\documentclass[10pt]{article}",
    r"\usepackage[margin=0.65in]{geometry}",
    r"\usepackage[T1]{fontenc}",
    r"\usepackage{xcolor}",
    r"\definecolor{accent}{HTML}{1F6F78}",
    r"\usepackage{titlesec}",
    r"\titleformat{\section}{\color{accent}\large\bfseries}{}{0pt}{}[{\color{accent}\titlerule}]",
    r"\titlespacing{\section}{0pt}{10pt}{4pt}",
    r"\usepackage{enumitem}",
    r"\setlist[itemize]{leftmargin=14pt,itemsep=1pt,topsep=2pt,parsep=0pt}",
    r"\usepackage{hyperref}",
    r"\hypersetup{colorlinks=true,urlcolor=accent,linkcolor=accent}",
    r"\pagestyle{empty}",
    r"\setlength{\parindent}{0pt}",
    r"\begin{document}",
])
_LATEX_CLOSE = r"\end{document}" + "\n"


def _tex_header_line(line: str) -> str:
    """One header line under the name: 3 `|`-separated parts lay out left/center/right (the base-resume style this
    is modeled on); anything else just centers, joined by a bullet."""
    parts = [p.strip() for p in line.split("|") if p.strip()]
    if len(parts) == 3:
        left, center, right = (_tex_text(p) for p in parts)
        return "\n".join([
            r"\noindent\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcr}",
            f"{left} & {center} & {right} \\\\",
            r"\end{tabular*}",
        ])
    if parts:
        return r"\begin{center}" + r" \textbullet\ ".join(_tex_text(p) for p in parts) + r"\end{center}"
    return ""


def _tex_role(role: Role) -> str:
    """One Experience role: bold title + right-aligned dates, italic company/location, then its bullets."""
    title = _tex_text(role.title)
    dates = _tex_text(role.dates)
    first = r"\noindent\textbf{" + title + "}"
    if dates:
        first += r"\hfill\textit{" + dates + "}"
    lines = [first + r"\\"]
    company_loc = ", ".join(p for p in (role.company, role.location) if p)
    if company_loc:
        lines.append(r"\textit{" + _tex_text(company_loc) + r"}\\")
    if role.bullets:
        lines.append(r"\begin{itemize}")
        lines.extend(r"\item " + _tex_text(b) for b in role.bullets)
        lines.append(r"\end{itemize}")
    return "\n".join(lines)


def _tex_lines(lines: list[str]) -> str:
    """A non-experience section's raw markdown lines as LaTeX: consecutive "- " bullets become one itemize block,
    a stray "### " heading (rare outside Experience) gets a bold/italic treatment like a role, anything else is a
    plain paragraph line."""
    out: list[str] = []
    bullets: list[str] = []

    def flush() -> None:
        if bullets:
            out.append(r"\begin{itemize}")
            out.extend(r"\item " + item for item in bullets)
            out.append(r"\end{itemize}")
            bullets.clear()

    for raw in lines:
        line = flatten_links(raw)
        if line.startswith("- "):
            bullets.append(_tex_text(line[2:]))
            continue
        flush()
        if line.startswith("### "):
            parts = [p.strip() for p in unbold(line[4:]).split("|") if unbold(p.strip())]
            if parts:
                head = r"\noindent\textbf{" + _tex_text(parts[0]) + "}"
                rest = ", ".join(_tex_text(p) for p in parts[1:])
                out.append(head + (r"\textit{ -- " + rest + "}" if rest else "") + r"\\")
        elif line.strip():
            out.append(_tex_text(line) + r"\\")
    flush()
    return "\n".join(out)


def _tex_source(markdown: str) -> str:
    """The resume Markdown dialect as a complete .tex document."""
    parsed = parse_resume(body_text(markdown))
    body = [_LATEX_PREAMBLE, r"\begin{center}{\LARGE\textbf{" + _latex_escape(parsed.name or "Candidate") + "}}"
            r"\end{center}"]
    body.extend(_tex_header_line(line) for line in parsed.header_lines)
    for section in parsed.sections:
        body.append(r"\section*{" + _latex_escape(section.title) + "}")
        if section.kind == "experience":
            body.append("\n".join(_tex_role(role) for role in section.roles))
        else:
            body.append(_tex_lines(section.lines))
    body.append(_LATEX_CLOSE)
    return "\n".join(part for part in body if part)


def markdown_to_latex_pdf(markdown: str, *, timeout: float = 20.0) -> bytes:
    """The resume PDF from the self-hosted LaTeX compile service at settings.latex_render_url. Raises on any
    failure (missing URL, network error, timeout, non-200 response) — render_resume_pdf is what callers should use;
    it falls back to the built-in renderer so a resume download never hard-fails on this external service."""
    from app.services.external_api import external_client

    url = get_settings().latex_render_url
    if not url:
        raise RuntimeError("LATEX_RENDER_URL is not configured")
    with external_client("LATEX", service="resume_export", operation="compile_pdf", timeout=timeout) as client:
        response = client.post(url, json={"source": _tex_source(markdown)})
    response.raise_for_status()
    return response.content


def render_resume_pdf(markdown: str) -> bytes:
    """The resume PDF a download gets: LaTeX-rendered when LATEX_RENDER_URL is configured and the service answers,
    the built-in reportlab renderer otherwise — a download must never hard-fail on the external service."""
    if not get_settings().latex_render_url:
        return markdown_to_simple_pdf(markdown)
    try:
        return markdown_to_latex_pdf(markdown)
    except Exception as exc:  # noqa: BLE001  a resume download must never hard-fail on this
        log.warning("LaTeX resume render failed (%s); falling back to the built-in renderer", exc, exc_info=True)
        return markdown_to_simple_pdf(markdown)


DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def markdown_to_docx(markdown: str) -> bytes:
    """The same resume Markdown as a Word document, for portals that reject PDF."""
    from docx import Document

    document = Document()

    def write(paragraph, text: str) -> None:
        """Markdown emphasis becomes real bold runs instead of literal ** markers."""
        for index, part in enumerate(text.split("**")):
            if part:
                paragraph.add_run(part).bold = index % 2 == 1

    for line in (markdown or "").split("## Source Resume")[0].splitlines():
        line = flatten_links(line)
        if line.startswith("# "):
            document.add_heading(unbold(line[2:]), level=0)
        elif line.startswith("### "):
            parts = [p.strip() for p in unbold(line[4:]).split("|")]
            paragraph = document.add_paragraph()
            paragraph.add_run(parts[0]).bold = True
            if len(parts) > 1:
                paragraph.add_run("  |  " + "  |  ".join(parts[1:]))
        elif line.startswith("## "):
            document.add_heading(unbold(line[3:]), level=1)
        elif line.startswith("- "):
            write(document.add_paragraph(style="List Bullet"), line[2:])
        elif line.strip():
            write(document.add_paragraph(), line)
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


# What a downloaded packet file is called. The apply kit shows this name and the download headers use the same
# one, so what lands in the Downloads folder is what the screen said it would be.
MAX_SLUG = 80


def packet_slug(name: str, company: str) -> str:
    """`Kunal_K_ComtecNova_Pte_Ltd` — the candidate and the employer, split on anything that isn't a letter or
    digit and rejoined with underscores. Each chunk keeps its own casing as written (never lowercased or
    re-title-cased), so "ComtecNova" survives as one word instead of being re-split into "Comtec" + "Nova"."""
    chunks = re.findall(r"[A-Za-z0-9]+", f"{name or ''} {company or ''}")
    slug = "_".join(chunks)[:MAX_SLUG].rstrip("_")
    return slug or "resume"


COVER_LETTER_KINDS = {"cover_letter": "txt", "cover_letter_pdf": "pdf", "cover_letter_docx": "docx"}


def packet_filename(slug: str, kind: str, version: int | None = None, *, draft: bool = False) -> str:
    """The file name for one part of a packet: resume PDF and DOCX (`pdf`, `docx`), or the cover letter as text
    (`cover_letter`, the Apply screen's download), PDF (`cover_letter_pdf`) or DOCX (`cover_letter_docx`). A `draft`
    (a version nobody approved yet) says so in its name."""
    suffix = "_draft" if draft else ""
    if kind in COVER_LETTER_KINDS:
        return f"Cover_Letter_{slug}{suffix}.{COVER_LETTER_KINDS[kind]}"
    return f"{slug}_v{version or 1}{suffix}.{kind}"
