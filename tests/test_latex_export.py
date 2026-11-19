"""The LaTeX resume renderer: escaping, the escape-before-wrap order, and render_resume_pdf's fallback behavior.

Network calls go through httpx.MockTransport (app.services.external_api.external_client), never a live request.
"""
import httpx
import pytest

from app.services import exports
from app.settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_every_special_character_is_escaped():
    assert exports._latex_escape("\\") == r"\textbackslash{}"
    assert exports._latex_escape("{") == r"\{"
    assert exports._latex_escape("}") == r"\}"
    assert exports._latex_escape("$") == r"\$"
    assert exports._latex_escape("&") == r"\&"
    assert exports._latex_escape("%") == r"\%"
    assert exports._latex_escape("#") == r"\#"
    assert exports._latex_escape("_") == r"\_"
    assert exports._latex_escape("~") == r"\textasciitilde{}"
    assert exports._latex_escape("^") == r"\textasciicircum{}"
    assert exports._latex_escape("$500K & 50%") == r"\$500K \& 50\%"


def test_escaping_happens_before_the_bold_wrapper_not_after():
    # "**bold** & more" must become "\textbf{bold} \& more", never a mangled escape of the \textbf wrapper itself.
    assert exports._tex_text("**bold** & more") == r"\textbf{bold} \& more"
    assert exports._tex_text("no bold here") == "no bold here"
    assert exports._tex_text("**C++ & C#**") == r"\textbf{C++ \& C\#}"


def test_header_line_with_three_parts_uses_a_tabular_row():
    out = exports._tex_header_line("left | center | right")
    assert "tabular*" in out
    assert "left & center & right" in out


def test_header_line_otherwise_centers_with_a_bullet_separator():
    out = exports._tex_header_line("a | b")
    assert out == r"\begin{center}a \textbullet\ b\end{center}"
    # The separator itself must never be re-escaped (a regression of exactly this bug was caught during review).
    assert "textasciitilde" not in out
    assert "textbackslash" not in out


MARKDOWN = """# Jane Doe
AI Engineer
India | jane@example.com

## Summary
Built systems saving **$500K** & 50% faster.

## Experience
### Engineer | Acme Corp | Remote | Jan 2020 - Present
- Shipped **100K+** units.

## Projects
Widget: A cool thing | Python
- Did a thing.
"""


def test_tex_source_has_no_double_escaped_control_sequences():
    tex = exports._tex_source(MARKDOWN)
    assert r"\$500K" in tex
    assert r"\textbf{\$500K}" in tex
    assert "\\begin{document}" in tex and "\\end{document}" in tex
    assert "Widget: A cool thing" in tex  # Projects section survives, not silently dropped


def test_render_resume_pdf_uses_reportlab_when_url_is_unset(monkeypatch):
    monkeypatch.setattr(get_settings(), "latex_render_url", None)
    monkeypatch.setattr(exports, "markdown_to_simple_pdf", lambda md: b"REPORTLAB_SENTINEL")
    assert exports.render_resume_pdf(MARKDOWN) == b"REPORTLAB_SENTINEL"


def _patched_client(monkeypatch, responder):
    """Replace app.services.external_api.external_client with one that always uses this MockTransport — captured
    before patching, so the replacement calls the real factory, never itself."""
    from app.services import external_api

    real_external_client = external_api.external_client

    def fake(provider, **kwargs):
        return real_external_client(provider, transport=httpx.MockTransport(responder), **kwargs)

    monkeypatch.setattr(external_api, "external_client", fake)


def test_markdown_to_latex_pdf_returns_pdf_bytes_on_success(monkeypatch):
    monkeypatch.setattr(get_settings(), "latex_render_url", "https://latex.test/compile")

    def responder(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://latex.test/compile"
        return httpx.Response(200, content=b"%PDF-1.7 fake pdf bytes", headers={"content-type": "application/pdf"})

    _patched_client(monkeypatch, responder)
    assert exports.markdown_to_latex_pdf(MARKDOWN) == b"%PDF-1.7 fake pdf bytes"


def test_render_resume_pdf_falls_back_on_a_400(monkeypatch):
    monkeypatch.setattr(get_settings(), "latex_render_url", "https://latex.test/compile")
    monkeypatch.setattr(exports, "markdown_to_simple_pdf", lambda md: b"REPORTLAB_SENTINEL")

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="source too short")

    _patched_client(monkeypatch, responder)
    assert exports.render_resume_pdf(MARKDOWN) == b"REPORTLAB_SENTINEL"


def test_render_resume_pdf_falls_back_on_a_connection_error(monkeypatch):
    monkeypatch.setattr(get_settings(), "latex_render_url", "https://latex.test/compile")
    monkeypatch.setattr(exports, "markdown_to_simple_pdf", lambda md: b"REPORTLAB_SENTINEL")

    def responder(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _patched_client(monkeypatch, responder)
    assert exports.render_resume_pdf(MARKDOWN) == b"REPORTLAB_SENTINEL"
