"""Reviewer notes as a person reads them: requirement and evidence ids replaced by what they stand for.

A reviewer model sometimes writes "req_024: ..." or "evidence IDs (127, 129)" — ids from its prompt that mean
nothing on a screen. This is a display transform only: stored notes are untouched (the next tailoring pass reads
them as written) and an id it cannot resolve is left as it is, never guessed.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Evidence, Job, JobRequirementEvidence

# "req_024: Timestamps - no evidence" / "req_024 - ..." at the start: the model already wrote the text after the id.
_LEADING_REQ = re.compile(r"^\s*(req_\d{1,4})\s*[:\-–—]\s*(?=\S)", re.IGNORECASE)
_REQ = re.compile(r"\b(req_\d{1,4})\b", re.IGNORECASE)
# "evidence 5", "evidence IDs (127, 129)", "bullet IDs 119, 120": a list of ids, never a range such as "1-12".
_IDS = re.compile(r"\b(evidence|bullet)(s)?(\s+ids?)?\s*(\(?)(\d+(?:\s*,\s*\d+)*)(?![\d\-–])(\)?)", re.IGNORECASE)
QUOTE_CHARS = 70
LISTED = 3


def _short(text: str, limit: int = QUOTE_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def requirement_texts(db: Session, job: Job) -> dict[str, str]:
    """Requirement id -> its words: the parsed requirements, then the ranking's own assessments."""
    names = {str(a["requirement_id"]).lower(): str(a["requirement"])
             for a in (job.fit_analysis or {}).get("requirement_assessments") or []
             if isinstance(a, dict) and a.get("requirement_id") and a.get("requirement")}
    for row in db.scalars(select(JobRequirementEvidence).where(JobRequirementEvidence.user_id == job.user_id,
                                                            JobRequirementEvidence.job_id == job.id)).all():
        names.setdefault(row.item_id.lower(), row.normalized_value or row.source_text)
    return names


def evidence_texts(db: Session, user_id: int, ids: Iterable[int]) -> dict[int, str]:
    wanted = {int(i) for i in ids}
    if not wanted:
        return {}
    rows = db.scalars(select(Evidence).where(Evidence.user_id == user_id, Evidence.id.in_(wanted))).all()
    return {e.id: e.claim for e in rows if e.claim}


def cited_evidence_ids(lines: Iterable[str]) -> set[int]:
    return {int(n) for line in lines for m in _IDS.finditer(str(line)) for n in re.findall(r"\d+", m.group(5))}


def readable(line: str, requirements: Mapping[str, str], evidence: Mapping[int, str]) -> str:
    text = str(line)
    lead = _LEADING_REQ.match(text)
    if lead and lead.group(1).lower() in requirements:
        text = text[lead.end():]  # the words follow the id already
    text = _REQ.sub(lambda m: f"“{_short(requirements[m.group(1).lower()])}”"
                    if m.group(1).lower() in requirements else m.group(0), text)

    def ids(m: re.Match) -> str:
        numbers = [int(n) for n in re.findall(r"\d+", m.group(5))]
        known = [n for n in numbers if n in evidence]
        if not known:
            return m.group(0)
        quotes = [f"“{_short(evidence[n])}”" for n in known[:LISTED]]
        rest = len(numbers) - len(quotes)
        noun = m.group(1).lower() + ("s" if len(numbers) > 1 else "")
        return f"{noun} {', '.join(quotes)}{f' and {rest} more' if rest else ''}"

    return _IDS.sub(ids, text)


def readable_lines(db: Session, job: Job, lines: list[str]) -> list[str]:
    """`lines` with every id it can resolve replaced; unchanged when none appear."""
    if not any(_REQ.search(str(x)) or _IDS.search(str(x)) for x in lines):
        return lines
    requirements = requirement_texts(db, job)
    evidence = evidence_texts(db, job.user_id, cited_evidence_ids(lines))
    return [readable(x, requirements, evidence) for x in lines]
