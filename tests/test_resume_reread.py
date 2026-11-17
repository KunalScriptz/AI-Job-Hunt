"""Re-reading a version's own file with resume extraction V2 (POST /api/resumes/{id}/reread).

A version uploaded before V2 keeps the older reader's text: pypdf ignores /ActualText, so a ligature glyph a font maps
wrongly stays wrong ("Kaga" for "Kafka", "hCps" for "https") when it occurs too rarely for text_repair to learn it.
Re-reading makes a new version from the same file; the old one and the active selection stay as they are."""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app import auth
from app.auth import CurrentUser
from app.db import SessionLocal
from app.main import app
from app.models import Resume, ResumeExtraction
from app.services import worker_settings
from app.settings import get_settings
from tests.conftest import make_user

DOCS = Path(__file__).parent / "fixtures" / "resume_extraction" / "documents"
client = TestClient(app)


@pytest.fixture
def user(monkeypatch):
    user_id = make_user(SessionLocal)
    monkeypatch.setattr(auth, "authenticate", lambda request: CurrentUser(user_id, f"{user_id}@jobhunt.test", False))
    monkeypatch.setattr(get_settings(), "llm_provider", "mock")
    monkeypatch.setattr(get_settings(), "resume_extraction_v2", False)
    return user_id


def _db(user_id):
    db = SessionLocal()
    db.info["user_id"] = user_id
    return db


def _v2(user_id, on: bool = True):
    with _db(user_id) as db:
        worker_settings.update(db, user_id, {"resume_parsing": {"extraction_v2": on}})


def _upload(name: str = "ana-single-actualtext.pdf") -> int:
    out = client.post("/api/profile/resume",
                      files={"file": (name, io.BytesIO((DOCS / name).read_bytes()), "application/pdf")})
    assert out.status_code == 200, out.text
    return out.json()["resume_id"]


def _record(db, resume_id):
    return db.scalar(select(ResumeExtraction).where(ResumeExtraction.resume_id == resume_id))


def test_rereading_an_old_version_makes_a_new_version_with_the_files_real_text(user):
    old = _upload()  # read without V2: the older reader keeps the broken glyph
    with _db(user) as db:
        before = db.get(Resume, old)
        assert "Kaga" in before.raw_text and _record(db, old) is None and before.is_active
        old_text, old_data = before.raw_text, before.structured_data
    _v2(user)
    out = client.post(f"/api/resumes/{old}/reread")
    assert out.status_code == 200, out.text
    body = out.json()
    new = body["id"]
    assert new != old and body["reread_from"] == old and body["is_active"] is False
    assert body["extraction"] is not None and body["extraction"]["quality"]["route"]
    with _db(user) as db:
        created = db.get(Resume, new)
        assert "Apache Kafka" in created.raw_text and "Kaga" not in created.raw_text
        assert "https://linkedin.example.test/in/anawhitfield" in created.raw_text
        assert _record(db, new) is not None and _record(db, old) is None
        kept = db.get(Resume, old)
        assert kept.raw_text == old_text and kept.structured_data == old_data and kept.is_active, \
            "the old version, its data and the active choice stay as they were"
        assert created.resume_document_id == kept.resume_document_id and created.name == kept.name


def test_rereading_again_reuses_the_new_version(user):
    old = _upload()
    _v2(user)
    first = client.post(f"/api/resumes/{old}/reread").json()["id"]
    again = client.post(f"/api/resumes/{old}/reread")
    assert again.status_code == 200 and again.json()["id"] == first
    with _db(user) as db:
        assert db.scalar(select(func.count()).select_from(Resume).where(Resume.user_id == user)) == 2


def test_a_version_already_read_with_v2_is_not_reread(user):
    _v2(user)
    resume_id = _upload()
    out = client.post(f"/api/resumes/{resume_id}/reread")
    assert out.status_code == 409 and "already" in out.json()["detail"]


def test_reread_needs_v2_on(user):
    old = _upload()
    out = client.post(f"/api/resumes/{old}/reread")
    assert out.status_code == 409 and "new reader" in out.json()["detail"]


def test_a_missing_or_changed_file_is_refused(user):
    old = _upload()
    _v2(user)
    with _db(user) as db:
        path = Path(db.get(Resume, old).source_storage_path)
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n% changed")
        changed = client.post(f"/api/resumes/{old}/reread")
        assert changed.status_code == 409 and "does not match" in changed.json()["detail"]
        path.unlink()
        missing = client.post(f"/api/resumes/{old}/reread")
        assert missing.status_code == 409 and "upload it again" in missing.json()["detail"]
    finally:
        path.write_bytes(original)


def test_another_accounts_version_is_not_found(user, monkeypatch):
    old = _upload()
    other = make_user(SessionLocal)
    _v2(other)
    monkeypatch.setattr(auth, "authenticate", lambda request: CurrentUser(other, f"{other}@jobhunt.test", False))
    assert client.post(f"/api/resumes/{old}/reread").status_code == 404


# ───────────── review findings (PR 147) ─────────────

def test_rereading_the_active_resume_keeps_its_download(user):
    old = _upload()
    _v2(user)
    assert client.get("/api/profile/resume").status_code == 200
    assert client.post(f"/api/resumes/{old}/reread").status_code == 200
    assert client.get("/api/profile/resume").status_code == 200, "the active version still uses the shared file"


def test_a_file_removed_from_the_library_can_still_be_reread(user):
    out = client.post("/api/profile/resume", files={"file": (
        "ana.pdf", io.BytesIO((DOCS / "ana-single-actualtext.pdf").read_bytes()), "application/pdf")}).json()
    _v2(user)
    assert client.delete(f"/api/profile/resumes/{out['id']}").status_code == 200
    assert client.get(f"/api/resumes/{out['resume_id']}/original").status_code == 200
    reread = client.post(f"/api/resumes/{out['resume_id']}/reread")
    assert reread.status_code == 200, reread.text
    new = reread.json()["id"]
    assert new != out["resume_id"] and reread.json()["has_original"]
    with _db(user) as db:
        created, old = db.get(Resume, new), db.get(Resume, out["resume_id"])
        assert "Apache Kafka" in created.raw_text and _record(db, new) is not None
        assert (created.source_storage_path, created.source_file_hash) == (old.source_storage_path,
                                                                           old.source_file_hash)
    assert client.get(f"/api/resumes/{new}/original").status_code == 200


def test_same_text_offers_the_new_structure_as_a_comparison(user):
    """A paragraph-only Word file reads to the same text, so its version is kept; the new reader's sections still find
    bullets the old parse missed, and they are offered for review, never applied on their own."""
    docx = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    out = client.post("/api/profile/resume", files={"file": (
        "ana.docx", io.BytesIO((DOCS / "ana.docx").read_bytes()), docx)}).json()
    with _db(user) as db:
        before = db.get(Resume, out["resume_id"]).structured_data
    _v2(user)
    reread = client.post(f"/api/resumes/{out['resume_id']}/reread")
    assert reread.status_code == 200, reread.text
    body = reread.json()
    assert body["id"] == out["resume_id"], "the same text keeps its version"
    assert body["extraction"] is not None
    proposed = " ".join(str(c["proposed"]) for c in body["pending_reparse"]["changes"])
    assert "Architected a Kafka streaming platform" in proposed
    with _db(user) as db:
        assert db.get(Resume, out["resume_id"]).structured_data == before, "offered, not applied"
    assert client.post(f"/api/resumes/{out['resume_id']}/reread").status_code == 409
