"""Jobs an assistant adds, corrects or puts away (the add_job, add_jobs_bulk, update_job and archive_job MCP tools).

    add(db, fields)          → validate → normalise the URL → an existing job of the same posting (canonical URL, a
                               manual job's ref, or company + title + city: app.services.hunter.find_duplicate)? return
                               it untouched : read the description when none was given (jd_recovery.fetch_posting,
                               HTTP only) → sources.upsert_job(source="manual") → discovery.admit, exactly as a job the
                               person adds in Discovery (domain filter KEPT; recovery, parsing and ranking follow)
    update(db, job, fields)  → corrected fields; a new description is parsed and ranked again (replace_description)
    archive(db, job, reason) → the Discovery "archive" triage action: status archived, out of ranking, never deleted

Fit scores come from the ranker's model, in the background (app.services.ranker.sweep), never here. Page text and
descriptions are untrusted employer data: stored as text, never followed. Nothing here submits or sends anything.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from bs4 import BeautifulSoup
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.models import Job
from app.services.apify_adapters import TRACKING_PARAM, canonical_url, dedupe_key
from app.services.job_description import normalize_description
from app.settings import get_settings
from app.tenancy import require_user_id

MAX_URL = 1200
MAX_TITLE = 300
MAX_COMPANY = 200
MAX_LOCATION = 300
MAX_DESCRIPTION = 50_000
MAX_SOURCE = 50
MAX_SALARY = 200
MAX_NOTES = 2000
MAX_REASON = 500
WORKPLACE_TYPES = ("onsite", "hybrid", "remote")
WORKPLACE_ALIASES = {"on-site": "onsite", "on site": "onsite", "office": "onsite", "in office": "onsite",
                     "in-office": "onsite", "wfh": "remote", "work from home": "remote"}
# A job with a tailored packet or an application is past triage: archiving it is refused (as app.workspace does).
PACKET_STATUSES = ("tailored", "needs_retailor", "approved", "applied", "pending_approval")
ARCHIVED = "archived"
TEXT_FIELDS = ("title", "company", "location", "workplace_type", "description", "apply_url", "notes", "posted_at",
               "salary")
UPDATE_FIELDS = TEXT_FIELDS
# Where intake details with no column of their own are kept on the job (Job.raw), namespaced away from board data.
RAW_KEY = "intake"
SOURCE_LABELS = {"builtin.com": "builtin", "workable.com": "workable", "greenhouse.io": "greenhouse",
                 "lever.co": "lever", "ashbyhq.com": "ashby", "linkedin.com": "linkedin", "indeed.com": "indeed",
                 "wellfound.com": "wellfound", "myworkdayjobs.com": "workday", "smartrecruiters.com": "smartrecruiters"}
HTML_TAG = re.compile(r"<\s*/?\s*[a-z][a-z0-9]*(\s[^>]*)?>", re.IGNORECASE)


class IntakeError(ValueError):
    """A request refused as a whole (unknown job, nothing to change, a change not allowed for this job)."""


@dataclass
class Invalid(ValueError):
    """Input that failed validation: one message per field."""

    errors: list[str]

    def __str__(self) -> str:
        return "; ".join(self.errors)


@dataclass
class Added:
    job: Job
    duplicate: bool
    matched_by: str | None = None  # url | dedupe_key, for a duplicate
    warnings: list[str] = field(default_factory=list)


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# ───────────── normalising ─────────────

def clean_url(url: str) -> str:
    """The URL as given, without its fragment or tracking parameters (utm_*, trackingId, refId, ...); the scheme, host
    and every other parameter are kept in their order, so the link still opens the same posting."""
    parsed = urlparse(url.strip())
    query = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if not TRACKING_PARAM.match(k)]
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, parsed.params, urlencode(query), ""))


def to_text(description: str) -> str:
    """Plain text from a description given as text or HTML. Never executed or followed: only its words are kept."""
    text = description.strip()
    if HTML_TAG.search(text):
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style", "noscript", "template", "iframe", "object"]):
            tag.decompose()  # code and embeds are not posting text
        text = normalize_description(str(soup))
    return text.strip()


def source_label(url: str) -> str:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    for suffix, label in SOURCE_LABELS.items():
        if host == suffix or host.endswith("." + suffix):
            return label
    return "company_site"


def manual_ref(canon: str) -> str:
    return hashlib.sha256(canon.encode()).hexdigest()[:40]


# ───────────── validation ─────────────

def _text(fields: dict[str, Any], name: str, limit: int, errors: list[str], *, required: bool = False) -> str | None:
    value = fields.get(name)
    if value is None:
        if required:
            errors.append(f"{name} is required")
        return None
    if not isinstance(value, str):
        errors.append(f"{name} must be text")
        return None
    value = value.strip()
    if required and not value:
        errors.append(f"{name} is required")
        return None
    if len(value) > limit:
        errors.append(f"{name} is longer than {limit} characters")
        return None
    return value


def validate(fields: dict[str, Any], *, partial: bool = False) -> dict[str, Any]:
    """The checked, normalised fields (only those present when partial). Raises Invalid with every problem."""
    if not isinstance(fields, dict):
        raise Invalid(["each job must be an object"])
    errors: list[str] = []
    out: dict[str, Any] = {}
    required = not partial
    url = _text(fields, "apply_url", MAX_URL, errors, required=required)
    if url is not None:
        parsed = urlparse(url)
        if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
            errors.append("apply_url must be an http:// or https:// web address")
        else:
            out["apply_url"] = clean_url(url)
    for name, limit in (("title", MAX_TITLE), ("company", MAX_COMPANY)):
        value = _text(fields, name, limit, errors, required=required)
        if value is not None:
            if not value:
                errors.append(f"{name} cannot be empty")
            else:
                out[name] = value
    for name, limit in (("location", MAX_LOCATION), ("salary", MAX_SALARY), ("notes", MAX_NOTES)):
        value = _text(fields, name, limit, errors)
        if value is not None:
            out[name] = value
    source = _text(fields, "source", MAX_SOURCE, errors)
    if source:
        if not re.fullmatch(r"[A-Za-z0-9_.\- ]+", source):
            errors.append("source must be a short label such as builtin, workable or company_site")
        else:
            out["source"] = re.sub(r"[\s.\-]+", "_", source.strip().lower())
    workplace = _text(fields, "workplace_type", 20, errors)
    if workplace:
        kind = WORKPLACE_ALIASES.get(workplace.lower(), workplace.lower())
        if kind not in WORKPLACE_TYPES:
            errors.append("workplace_type must be onsite, hybrid or remote")
        else:
            out["workplace_type"] = kind
    description = fields.get("description")
    if description is not None:
        if not isinstance(description, str):
            errors.append("description must be text")
        elif len(description) > MAX_DESCRIPTION:
            errors.append(f"description is longer than {MAX_DESCRIPTION} characters")
        elif description.strip():
            out["description"] = to_text(description)[:MAX_DESCRIPTION]
    posted = _text(fields, "posted_at", 40, errors)
    if posted:
        try:
            when = datetime.fromisoformat(posted)
            out["posted_at"] = (when.astimezone(UTC).replace(tzinfo=None) if when.tzinfo else when, posted)
        except ValueError:
            errors.append("posted_at must be an ISO date such as 2026-11-01")
    if "fetch_description" in fields and fields["fetch_description"] is not None:
        if not isinstance(fields["fetch_description"], bool):
            errors.append("fetch_description must be true or false")
        else:
            out["fetch_description"] = fields["fetch_description"]
    unknown = sorted(set(fields) - {*TEXT_FIELDS, "source", "fetch_description"})
    if unknown:
        errors.append(f"unknown field(s): {', '.join(map(str, unknown))}")
    if errors:
        raise Invalid(errors)
    return out


# ───────────── duplicates ─────────────

def find_existing(db: Session, owner: int, *, url: str, title: str = "", company: str = "", location: str = "",
                  exclude: int | None = None, by_key: bool = True) -> tuple[Job | None, str | None]:
    """The saved job this posting is: the same URL (canonical, as discovery matches it, or as stored by a manual
    add), then — with by_key — the same company + title + city (hunter's dedupe key). Returns the job and how it
    matched ("url" or "dedupe_key")."""
    canon = canonical_url(url)
    mine = select(Job).where(Job.user_id == owner)
    if exclude is not None:
        mine = mine.where(Job.id != exclude)
    urls = list(dict.fromkeys(u for u in (url, canon) if u))
    by_url = [Job.url.in_(urls), Job.direct_apply_url.in_(urls), Job.source_url.in_(urls),
              (Job.source == "manual") & Job.source_ref.in_([manual_ref(u) for u in urls])]
    if canon:
        by_url += [Job.canonical_url == canon, Job.canonical_apply_url == canon]
    found = db.scalars(mine.where(or_(*by_url)).order_by(Job.id).limit(1)).first()
    if found is not None:
        return found, "url"
    if by_key:
        # The same key hunter.find_duplicate matches discovered jobs by (company suffixes dropped, first city part).
        found = db.scalars(mine.where(Job.dedupe_key == dedupe_key(title, company, location))
                           .order_by(Job.id).limit(1)).first()
        if found is not None:
            return found, "dedupe_key"
    return None, None


# ───────────── adding ─────────────

def _intake(job: Job) -> dict[str, Any]:
    return dict((job.raw or {}).get(RAW_KEY) or {})


def _set_intake(job: Job, **values: Any) -> None:
    raw = dict(job.raw or {})
    raw[RAW_KEY] = {**_intake(job), **{k: v for k, v in values.items() if v is not None}}
    job.raw = raw  # a new dict, so the JSON column is written


def add(db: Session, fields: dict[str, Any], *, via: str = "mcp") -> Added:
    """Add one job (see the module docstring). Raises Invalid for bad input. Commits."""
    from app.services import jd_recovery, job_removal
    from app.services.sources import upsert_job

    data = validate(fields)
    owner = require_user_id(db)
    url, title, company = data["apply_url"], data["title"], data["company"]
    location = data.get("location", "")
    existing, matched = find_existing(db, owner, url=url, title=title, company=company, location=location)
    if existing is not None:
        warnings = ["Already saved: nothing was changed. Use update_job to correct or enrich it."]
        if data.get("description") and not (existing.description or "").strip():
            warnings.append("The saved job has no description; pass yours to update_job to add it.")
        return Added(existing, True, matched, warnings)

    warnings: list[str] = []
    canon = canonical_url(url) or url
    key = dedupe_key(title, company, location)
    if job_removal.is_removed(db, owner, canonical_url=canon, dedupe_key=key):
        warnings.append("This posting was removed earlier because it no longer existed; it was added again anyway.")
    description, fetched = data.get("description", ""), None
    if not description and data.get("fetch_description", True):
        db.rollback()  # nothing is written yet: no transaction stays open while the page is fetched
        fetched = jd_recovery.fetch_posting(url, user_id=owner, title=title, company=company)
        if fetched.ok:
            description = to_text(fetched.text)[:MAX_DESCRIPTION]
        else:
            warnings.append(f"Description missing: {fetched.message}. The job was saved; "
                            + ("Job Hunt OS will try to recover it in the background, or "
                               if get_settings().jd_fetch_enabled else "")
                            + "pass the posting text to update_job.")
    elif not description:
        warnings.append("Description missing: pass the posting text to update_job so the job can be ranked.")
    posted_at, posted_raw = data.get("posted_at", (None, None))
    source = data.get("source") or source_label(url)
    job, created = upsert_job(
        db, source="manual", source_ref=manual_ref(canon), title=title, company=company, location=location, url=url,
        description=description, posted_at=posted_at, posted_date_raw=posted_raw,
        raw={RAW_KEY: {k: v for k, v in {
            "via": via, "source": source, "notes": data.get("notes"), "salary": data.get("salary"),
            "workplace_type": data.get("workplace_type"), "added_at": utcnow().isoformat(),
            "description_from": ("provided" if data.get("description") else
                                 f"fetched:{fetched.method}" if fetched and fetched.ok else None)}.items()
            if v is not None}},
        extra={"canonical_url": canon, "dedupe_key": key, "source_board": source,
               "remote_type": data.get("workplace_type")})
    if not created:  # another call saved the same posting a moment ago
        return Added(job, True, "url", ["Already saved: nothing was changed."])
    if fetched is not None and fetched.ok:
        job.description_fetch_method, job.description_fetched_at = jd_recovery.HTTP, utcnow()
        job.description_source_url, job.description_final_url = url, (fetched.final_url or url)[:MAX_URL]
        db.commit()
    return Added(job, False, None, warnings)


# ───────────── updating ─────────────

def is_applied(job: Job) -> bool:
    return job.status == "applied" or bool(job.application and job.application.applied_at)


def replace_description(db: Session, job: Job, text: str, *, actor: str) -> str | None:
    """Set the posting text as the person's own (recovery never replaces it), parse it and rank it again, exactly as
    Discovery's "save description" does. Returns None when ranking was queued, otherwise why not. Does not commit."""
    from app.services import jd_recovery, job_description, ranker

    job.description = text
    job_description.normalize_job(db, job)
    jd_recovery.on_description_changed(db, job, by_user=True)
    job_description.queue_parse(db, job, reason="description_changed")
    ranker.reopen_for_description(db, job, actor=actor)
    return ranker.queue_job(db, job, policy=ranker.policy_for(db, job.user_id), actor=actor,
                            reason="description_changed", force=True)


def update(db: Session, job_id: int, fields: dict[str, Any], *, actor: str = "mcp") -> tuple[Job, list[str], dict]:
    """Correct or enrich a job. An applied job only takes notes. Returns (job, changed fields, rescoring) where
    rescoring says whether a new description was queued for ranking. Commits."""
    from app.services import job_description

    data = validate({k: v for k, v in fields.items() if v is not None}, partial=True)
    data.pop("fetch_description", None)
    data.pop("source", None)
    if not data:
        raise IntakeError(f"Pass at least one field to change: {', '.join(UPDATE_FIELDS)}")
    job = job_description.lock_job(db, job_id)
    if job is None:
        raise IntakeError(f"Job {job_id} not found")
    if is_applied(job) and set(data) - {"notes"}:
        raise IntakeError(f"Job {job.id} was already applied to: only notes can change")
    changed: list[str] = []
    if "apply_url" in data and data["apply_url"] != job.url:
        other, _ = find_existing(db, job.user_id, url=data["apply_url"], exclude=job.id, by_key=False)
        if other is not None:
            raise IntakeError(f"That URL is already saved as job {other.id} ({other.title} at {other.company})")
        job.url, job.canonical_url = data["apply_url"], canonical_url(data["apply_url"])
        changed.append("apply_url")
    for name in ("title", "company", "location"):
        if name in data and data[name] != getattr(job, name):
            setattr(job, name, data[name])
            changed.append(name)
    if {"title", "company", "location"} & set(changed):
        job.dedupe_key = dedupe_key(job.title, job.company, job.location)
    if "workplace_type" in data and data["workplace_type"] != job.remote_type:
        job.remote_type = data["workplace_type"]
        changed.append("workplace_type")
    if "posted_at" in data:
        job.posted_at, job.posted_date_raw = data["posted_at"]
        changed.append("posted_at")
    intake = {k: data[k] for k in ("notes", "salary", "workplace_type") if k in data}
    if intake:
        _set_intake(job, **intake, updated_at=utcnow().isoformat())
        changed += [k for k in ("notes", "salary") if k in data]
    rescoring: dict[str, Any] = {}
    if "description" in data and data["description"] != (job.description or ""):
        why = replace_description(db, job, data["description"], actor=f"{actor}:update_job")
        changed.append("description")
        rescoring = {"rescoring": "queued" if why is None else "not_queued", **({"rescoring_reason": why} if why else {})}
    db.commit()
    db.refresh(job)
    return job, changed, rescoring


# ───────────── archiving ─────────────

def archive(db: Session, job_id: int, reason: str, *, actor: str = "mcp") -> tuple[Job, bool]:
    """Put a job away as Discovery's "archive" does: status archived, out of the queue and of ranking, never deleted
    and teaching the domain filter nothing. Refused once it has a packet or an application. Returns (job, changed).
    Commits."""
    if not isinstance(reason, str) or not reason.strip():
        raise IntakeError("reason is required")
    if len(reason) > MAX_REASON:
        raise IntakeError(f"reason is longer than {MAX_REASON} characters")
    job = db.get(Job, job_id)
    if job is None:
        raise IntakeError(f"Job {job_id} not found")
    if job.status == ARCHIVED and job.triaged_at is not None:
        return job, False
    if job.status in PACKET_STATUSES:
        raise IntakeError(f"Job {job.id} is {job.status}: a job with a tailored packet or an application is not "
                          "archived")
    job.status, job.triaged_at = ARCHIVED, utcnow()
    _set_intake(job, archived_reason=reason.strip(), archived_at=utcnow().isoformat(), archived_by=actor)
    db.commit()
    return job, True
