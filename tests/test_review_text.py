from app.services.review_text import cited_evidence_ids, readable, readable_lines

REQS = {"req_024": "Timestamps and numeric precision", "req_034": "Financial or market data experience"}
EVIDENCE = {5: "Built the ingestion pipeline for 40M daily events", 127: "Led a team of 6 engineers",
            129: "Cut p99 latency by 35%", 131: "Owned the on-call rota", 141: "Migrated to Kubernetes"}


def test_a_leading_requirement_id_is_dropped_when_its_words_follow():
    assert readable("req_024: Timestamps and numeric precision - no evidence", REQS, EVIDENCE) \
        == "Timestamps and numeric precision - no evidence"


def test_an_inline_requirement_id_becomes_its_text_and_an_unknown_one_stays():
    assert readable("Gap on req_034 and req_099", REQS, EVIDENCE) \
        == "Gap on “Financial or market data experience” and req_099"


def test_evidence_ids_become_quotes_with_the_rest_counted():
    out = readable("Claims cite evidence IDs (127, 129, 131, 141) that are not shown", REQS, EVIDENCE)
    assert out == ("Claims cite evidences “Led a team of 6 engineers”, “Cut p99 latency by 35%”, "
                   "“Owned the on-call rota” and 1 more that are not shown")
    assert readable("evidence 5 mentions ingestion", REQS, EVIDENCE) \
        == "evidence “Built the ingestion pipeline for 40M daily events” mentions ingestion"


def test_ranges_and_unknown_ids_are_left_alone():
    line = "only evidence IDs 1-12, 15 exist; bullet 999 is missing"
    assert readable(line, REQS, EVIDENCE) == line
    assert cited_evidence_ids(["evidence IDs (127, 129)", "bullet 5"]) == {127, 129, 5}


def test_lines_without_ids_need_no_lookup():
    lines = ["Strong Python background"]
    assert readable_lines(None, None, lines) is lines
