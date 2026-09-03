"""Document ingestion.

No guardrail rail runs on a document at ingest time — it is chunked and
indexed exactly as uploaded, once normalized. Every test here runs without
an API key: extraction, chunking, oversized-document quarantine, and the
corpus itself are all deterministic, and a keyless deployment must still
get them.
"""



from __future__ import annotations



import pytest



from backend.guardrails import AuditLog, Corpus, Document, Engine, IngestError, load

from backend.guardrails.knowledge import (

    OCRUnavailable,

    chunk_text,

    extract,

    new_document_id,
    stem,

    tokens,

)

from backend.guardrails.knowledge.seed import CORPUS
from backend.guardrails.rails.vault import CORPUS_OWNER
from tests.conftest import REPO





@pytest.fixture

def corpus():

    return Corpus(seed=True)          # memory only, never touches data/





@pytest.fixture

def ingest_engine(tmp_path, corpus):

    return Engine(

        load(REPO / "config" / "policy.yaml"),

        llm=None,

        audit=AuditLog(tmp_path / "audit.log"),

        corpus=corpus,

    )








# ── chunking ───────────────────────────────────────────────────────

def test_short_document_is_one_chunk():

    assert len(chunk_text("A single short paragraph.", 700, 80)) == 1





def test_chunks_respect_the_size_budget():

    text = "\n\n".join(f"Paragraph {i}. " + "filler words here. " * 12 for i in range(20))

    chunks = chunk_text(text, 500, 60)

    assert len(chunks) > 1

    assert all(len(c) <= 500 + 60 for c in chunks), [len(c) for c in chunks]





def test_a_paragraph_longer_than_the_budget_is_split_not_dropped():

    para = "word " * 600                      # ~3000 chars, one paragraph

    chunks = chunk_text(para, 400, 40)

    assert len(chunks) > 1

    assert sum(len(c) for c in chunks) >= 2000





def test_overlap_carries_context_between_chunks():

    text = "\n\n".join(f"Paragraph {i} about licences and renewals." for i in range(40))

    with_overlap = chunk_text(text, 300, 80)

    without = chunk_text(text, 300, 0)

    assert len(with_overlap) >= len(without)





def test_empty_text_produces_no_chunks():

    assert chunk_text("   \n\n  ", 700, 80) == []





# ── extraction ─────────────────────────────────────────────────────

def test_text_extraction_handles_utf8_and_cp1252():

    assert "café" in extract("a.txt", "café".encode("utf-8")).text

    assert "café" in extract("a.txt", "café".encode("cp1252")).text





def test_unknown_extension_is_rejected_with_the_accepted_list():

    with pytest.raises(IngestError) as exc:

        extract("payload.exe", b"MZ", allowed=["txt", "md"])

    assert "txt" in str(exc.value) and "md" in str(exc.value)





def test_document_ids_are_unique_for_the_same_title():

    assert new_document_id("Fee schedule") != new_document_id("Fee schedule")





# ── the pipeline ───────────────────────────────────────────────────

def test_clean_document_is_indexed_and_searchable(ingest_engine):

    result = ingest_engine.ingest(

        "Late renewal circular",

        "A trade licence renewed after expiry attracts a late surcharge of 25 percent "

        "of the standard renewal fee.",

    )

    assert result.document.indexed

    assert not result.quarantined

    hits = ingest_engine.corpus.search("late surcharge renewal expiry")

    assert any(h.doc_id == result.document.id for h in hits)







def test_the_corpus_owner_is_not_a_name_an_account_can_hold(monkeypatch, tmp_path):
    """`CORPUS_OWNER` is unforgeable only while nobody can sign in as it.

    `add_user` allows letters, digits, `-`, `_` and `.`; `@` is not on that list,
    which is the whole reason the corpus bucket is out of reach. This is here so
    that widening the username charset fails loudly, rather than quietly handing
    every ingested document to whoever registers the name.
    """
    from backend.server import auth
    from backend.server.auth import Directory

    monkeypatch.setattr(auth, "USERS_PATH", tmp_path / "users.json")
    monkeypatch.setattr(auth, "SESSIONS_PATH", tmp_path / "sessions.json")
    with pytest.raises(ValueError, match="username may contain"):
        Directory().add_user(CORPUS_OWNER, "a-password", "admin")





def test_oversized_document_is_refused(ingest_engine):

    ingest_engine.policy.values["ingest.max_document_chars"] = 500

    result = ingest_engine.ingest("Big", "word " * 400)

    assert result.quarantined

    assert "max_document_chars" in result.reason





def test_ingestion_is_audited(ingest_engine, tmp_path):

    ingest_engine.ingest("Notice", "The renewal fee is 1,200 rupees.")

    ok, message = AuditLog(tmp_path / "audit.log").verify()

    assert ok, message





# ── the corpus ─────────────────────────────────────────────────────

def test_seed_corpus_is_present(corpus):

    stats = corpus.stats()

    assert stats["documents"] == len(CORPUS)

    assert stats["indexed"] == len(CORPUS)





def test_bm25_prefers_the_document_that_is_actually_about_the_query(corpus):
    corpus.add(Document(id="fee-doc", title="Trade licence fees", source="test",
                        kind="txt", chars=50,
                        chunks=["The renewal fee for a small shop is 1,200 rupees, "
                               "billed at renewal."],
                        status="indexed", verdict="pass"))
    corpus.add(Document(id="renewal-doc", title="Trade licence renewal", source="test",
                        kind="txt", chars=50,
                        chunks=["Renewing a trade licence needs Form 4B and proof "
                               "of premises."],
                        status="indexed", verdict="pass"))

    hits = corpus.search("trade licence renewal fee for a small shop")

    assert hits

    assert "fee" in hits[0].title.lower()





def test_embedding_rerank_fixes_a_paraphrase_bm25_gets_wrong(corpus, monkeypatch):
    """BM25 is lexical: a query sharing more literal words with an irrelevant
    chunk outranks a genuinely relevant paraphrase with fewer shared terms.
    Stubs `embedding_rerank.rerank` rather than loading a real model — this
    tests the wiring (does `search_with_rerank` call the reranker and use its
    order?), not embedding quality, the same way this suite stubs Presidio/
    toxicity elsewhere rather than testing a real model's accuracy."""
    from backend.guardrails import load
    from backend.guardrails.knowledge.ingest import search_with_rerank
    from backend.guardrails.rails import embedding_rerank

    corpus.add(Document(id="wrong-doc", title="Wrong", source="test", kind="txt", chars=50,
                        chunks=["Renewal renewal renewal: every renewal form mentions "
                               "renewal fees and renewal deadlines for renewal."],
                        status="indexed", verdict="pass"))
    corpus.add(Document(id="right-doc", title="Right", source="test", kind="txt", chars=50,
                        chunks=["Obtaining a fresh trade permit requires submitting proof "
                               "of premises to the municipal office."],
                        status="indexed", verdict="pass"))

    query = "how do I renew my trade licence"
    plain = corpus.search(query, k=2)
    assert plain[0].title == "Wrong", \
        "test setup assumption broken: BM25 should prefer the keyword-stuffed chunk here"

    policy = load(REPO / "config" / "policy.yaml")
    policy.values["retrieval.engine"] = "bm25+embedding"
    policy.values["retrieval.embedding_candidates"] = 4

    monkeypatch.setattr(
        embedding_rerank, "rerank",
        lambda q, hits, top_k: sorted(hits, key=lambda h: h.title != "Right")[:top_k])

    reranked = search_with_rerank(corpus, query, 2, 0.15, policy)
    assert reranked[0].title == "Right"


def test_search_with_rerank_falls_back_to_bm25_order_while_model_loads(corpus, monkeypatch):
    """`embedding_rerank.rerank` returning None (model not loaded yet) must not
    lose results — the same 'never worse than plain BM25' contract every
    other local model in this codebase already has."""
    from backend.guardrails import load
    from backend.guardrails.knowledge.ingest import search_with_rerank
    from backend.guardrails.rails import embedding_rerank

    corpus.add(Document(id="only-doc", title="Only", source="test", kind="txt", chars=50,
                        chunks=["Trade licence renewal needs Form 4B."],
                        status="indexed", verdict="pass"))
    policy = load(REPO / "config" / "policy.yaml")
    policy.values["retrieval.engine"] = "bm25+embedding"
    monkeypatch.setattr(embedding_rerank, "rerank", lambda q, hits, top_k: None)

    hits = search_with_rerank(corpus, "trade licence renewal", 4, 0.15, policy)
    assert hits and hits[0].title == "Only"


def test_retrieval_engine_off_by_default_never_touches_the_reranker(corpus, monkeypatch):
    """Default config ('bm25') must not even import/call the reranker —
    proves the feature costs nothing when nobody has opted in."""
    from backend.guardrails import load
    from backend.guardrails.knowledge.ingest import search_with_rerank

    corpus.add(Document(id="d", title="D", source="test", kind="txt", chars=50,
                        chunks=["Trade licence renewal needs Form 4B."],
                        status="indexed", verdict="pass"))
    policy = load(REPO / "config" / "policy.yaml")
    assert str(policy.get("retrieval.engine")) == "bm25"

    called = []
    import backend.guardrails.rails.embedding_rerank as er
    monkeypatch.setattr(er, "rerank", lambda *a, **k: called.append(1) or None)

    hits = search_with_rerank(corpus, "trade licence renewal", 4, 0.15, policy)
    assert hits and not called


def test_coverage_gate_drops_a_weak_match(corpus):

    """A weak match is worse than no match — it gives grounding something

    irrelevant to score against."""

    assert corpus.search("quantum chromodynamics lattice gauge", min_coverage=0.15) == []





def test_removing_a_document_removes_it_from_the_index(tmp_path):

    # A blank corpus, not the shared `ingest_engine` fixture's: this test
    # asserts a *complete absence* of matches after removal, which the real
    # seed document (see knowledge/seed.py) could coincidentally satisfy on
    # its own, unrelated to the thing actually under test.
    engine = Engine(load(REPO / "config" / "policy.yaml"), llm=None,
                    audit=AuditLog(tmp_path / "audit.log"), corpus=Corpus(seed=False))

    result = engine.ingest("Ephemeral notice", "The parking levy is 40 rupees daily.")

    assert engine.corpus.search("parking levy daily")

    engine.corpus.remove(result.document.id)

    assert not engine.corpus.search("parking levy daily")





def test_reset_restores_the_seed_corpus(ingest_engine):

    ingest_engine.ingest("Temporary", "A temporary notice about levies and permits.")

    assert ingest_engine.corpus.stats()["uploaded"] == 1

    ingest_engine.corpus.reset()

    assert ingest_engine.corpus.stats() == {

        "documents": len(CORPUS), "indexed": len(CORPUS), "quarantined": 0,

        "uploaded": 0, "chunks": len(CORPUS), "masked_values": 0,

    }





def test_corpus_survives_a_round_trip_to_disk(tmp_path):

    path = tmp_path / "corpus.json"

    first = Corpus(path)

    engine = Engine(load(REPO / "config" / "policy.yaml"), None,

                    AuditLog(tmp_path / "a.log"), first)

    engine.ingest("Levy notice", "The parking levy is 40 rupees per day in zone C.")



    second = Corpus(path)

    assert second.stats()["uploaded"] == 1

    assert second.search("parking levy zone")





def test_a_deleted_built_in_stays_deleted_across_a_restart(tmp_path, monkeypatch):
    """Deleting a seed is only a real choice if the next process honours it.

    `install_new_builtins` tracks what a store has ever been given by id rather
    than by what it currently holds, so a built-in removed on purpose is not
    reinstalled at the next start. That is what made it safe to stop refusing
    the delete in the API.

    This monkeypatches two throwaway entries in rather than depend on
    `CORPUS`'s real content — `Corpus`'s own methods re-import it fresh each call, but this
    file's own `CORPUS` name was already bound at import time, so the
    patched list itself (not that stale name) is what the assertions below
    compare against. Two, not one: deleting the *only* built-in leaves the
    store empty, which takes `load()`'s "nothing here at all" branch
    (`seed_builtin()`, unconditional) rather than the one this test means to
    exercise (`install_new_builtins()`, which actually checks
    `seeds_installed`) — a real, separate edge case, not what is under test
    here.
    """
    two_builtins = [
        {"id": "test-only-a", "title": "Test-only built-in A",
         "text": "Exists only to prove a deletion survives a restart."},
        {"id": "test-only-b", "title": "Test-only built-in B",
         "text": "Stays behind so the store is never empty after the delete."},
    ]
    monkeypatch.setattr("backend.guardrails.knowledge.seed.CORPUS", two_builtins)

    path = tmp_path / "corpus.json"
    first = Corpus(path)
    doc_id = next(d.id for d in first.all() if d.source == "built-in")
    assert first.remove(doc_id)

    reopened = Corpus(path)
    assert reopened.get(doc_id) is None
    assert reopened.stats()["documents"] == len(two_builtins) - 1

    reopened.reset()
    assert reopened.get(doc_id) is not None, "reset is the way back"


def test_a_corrupt_store_falls_back_to_the_seed_rather_than_crashing(tmp_path):

    path = tmp_path / "corpus.json"

    path.write_text("{not json at all", encoding="utf-8")

    assert Corpus(path).stats()["documents"] == len(CORPUS)





# ── retrieval wiring ───────────────────────────────────────────────

def test_ingested_documents_reach_a_conversation(ingest_engine):

    ingest_engine.ingest(

        "Zone C parking levy",

        "The zone C parking levy is 40 rupees per day and is billed monthly.",

    )

    result = ingest_engine.converse("what is the zone C parking levy")

    assert any("zone C parking levy" in c for c in result.chunks)





def test_tokeniser_drops_stopwords():
    assert tokens("What is the fee for a licence") == ["fee", "licenc"]


def test_tokeniser_stems_so_coverage_measures_topic_not_vocabulary():
    """Regression: "where do I file a grievance" scored 0.143 against a 0.15
    gate — because the document says "filed" and "filing" — and the right
    document was dropped."""
    assert stem("file") == stem("filed") == stem("filing")
    assert stem("renew") == stem("renewal") == stem("renewals")
    assert stem("grievance") == stem("grievances")


def test_the_grievance_query_reaches_its_document(corpus):
    corpus.add(Document(id="grievance-doc", title="Filing a grievance", source="test",
                        kind="txt", chars=60,
                        chunks=["Grievances about service delivery can be filed at "
                               "any office, with a first response within 15 days."],
                        status="indexed", verdict="pass"))
    hits = corpus.search("Where do I file a grievance and how soon will someone respond?")
    assert hits and hits[0].doc_id == "grievance-doc"





# ── spreadsheets, images, scans ────────────────────────────────────

def _workbook() -> bytes:

    """A two-sheet workbook, in memory."""

    import io



    from openpyxl import Workbook



    book = Workbook()

    fees = book.active

    fees.title = "Fees"

    fees.append(["Service", "Amount", "Currency"])

    fees.append(["Trade licence renewal", 1200, "rupees"])

    fees.append([None, None, None])                       # blank rows are dropped

    fees.append(["Birth certificate copy", 100, "rupees"])

    contacts = book.create_sheet("Contacts")

    contacts.append(["Office", "Email"])

    contacts.append(["Chennai", "registrar.chennai@tn.gov.in"])

    buf = io.BytesIO()

    book.save(buf)

    return buf.getvalue()





def test_a_workbook_becomes_markdown_tables():

    result = extract("fees.xlsx", _workbook())

    assert result.method == "sheet"

    assert result.pages == 2

    assert "## Fees" in result.text and "## Contacts" in result.text

    assert "| Trade licence renewal | 1200 | rupees |" in result.text

    # blank rows carry nothing and are not indexed

    assert "|  |  |  |" not in result.text






def test_an_image_without_a_model_refuses_rather_than_indexing_nothing():

    with pytest.raises(OCRUnavailable) as exc:

        extract("scan.png", b"\x89PNG fake")

    assert "ANTHROPIC_API_KEY" in str(exc.value)





def test_an_image_is_transcribed_through_the_injected_hook():

    seen = {}



    def fake_ocr(image, media_type, hint):

        seen.update(media_type=media_type, hint=hint, size=len(image))

        return "The renewal fee is 1,200 rupees."



    result = extract("notice.jpg", b"fake-jpeg-bytes", ocr=fake_ocr)

    assert result.method == "image.ocr"

    assert "1,200 rupees" in result.text

    assert seen["media_type"] == "image/jpeg"      # jpg is jpeg on the wire





def _pdf(pages: list[str]) -> bytes:

    """A real PDF whose pages carry a text layer."""

    import fitz



    doc = fitz.open()

    for body in pages:

        page = doc.new_page()

        if body:

            page.insert_text((72, 96), body, fontsize=11)

    data = doc.tobytes()

    doc.close()

    return data





def test_a_text_pdf_uses_its_text_layer_and_never_calls_the_model():

    called = []



    def fake_ocr(*args):

        called.append(args)

        return "should not happen"



    result = extract("report.pdf", _pdf(["Renewal applications open 60 days before expiry."]),

                     ocr=fake_ocr)

    assert result.method == "pdf.text"

    assert called == []

    assert "60 days" in result.text


def _pdf_with_table(rows: list[list[str]], above: str = "", below: str = "") -> bytes:
    """A real PDF page with a bordered, grid-lined table `find_tables()` can
    actually detect — not just text that happens to line up in columns."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 72
    if above:
        page.insert_text((72, y), above, fontsize=11)
        y += 30
    col_w = [160, 100, 140]
    row_h = 24
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            x = 72 + sum(col_w[:c])
            rect = fitz.Rect(x, y + r * row_h, x + col_w[c], y + (r + 1) * row_h)
            page.draw_rect(rect, color=(0, 0, 0), width=0.5)
            page.insert_text((x + 4, y + r * row_h + 16), text, fontsize=10)
    y += len(rows) * row_h + 20
    if below:
        page.insert_text((72, y), below, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def test_a_bordered_table_is_read_as_a_markdown_table_not_flattened_prose():
    """`pypdf`-style flat extraction has no notion of a table: cells come back
    in whatever order the content stream lists them, which routinely is not
    reading order — a two-column fee schedule can interleave into
    "Fee ScheduleRupees" rather than a usable row. Detecting the table's own
    region and rendering it separately is what keeps the columns lined up."""
    rows = [
        ["Premises size", "Fee (rupees)", "Processing"],
        ["Under 500 sq ft", "1,200", "30 working days"],
        ["500 sq ft and above", "2,400", "30 working days"],
    ]
    data = _pdf_with_table(rows, above="Trade licence renewal fees.",
                           below="Payment is accepted online or at any counter.")

    result = extract("fees.pdf", data)

    assert result.method == "pdf.text"
    assert "| Premises size | Fee (rupees) | Processing |" in result.text
    assert "|---|---|---|" in result.text
    assert "| Under 500 sq ft | 1,200 | 30 working days |" in result.text
    assert "| 500 sq ft and above | 2,400 | 30 working days |" in result.text
    # Reading order preserved: the surrounding prose is not shuffled to the
    # end, and is not itself swallowed into the table.
    assert result.text.index("renewal fees") < result.text.index("Premises size")
    assert result.text.index("500 sq ft and above") < result.text.index("Payment is accepted")


def test_a_pdf_with_no_table_falls_back_to_plain_text():
    """The common case must not regress: no bordered region, no table-shaped
    output, just the page's own text."""
    result = extract("plain.pdf", _pdf(["Just an ordinary paragraph, no table here."]))
    assert result.method == "pdf.text"
    assert "|" not in result.text
    assert "ordinary paragraph" in result.text



def test_a_scanned_pdf_is_transcribed_page_by_page():

    """No text layer — every page is rasterised and sent to the transcriber."""

    pages = []



    def fake_ocr(image, media_type, hint):

        pages.append(hint)

        assert media_type == "image/png"

        assert image[:4] == b"\x89PNG"            # really rendered, not passed through

        return f"transcribed {len(pages)}"



    result = extract("scan.pdf", _pdf(["", ""]), ocr=fake_ocr)

    assert result.method == "pdf.ocr"

    assert result.transcribed == 2

    assert len(pages) == 2





def test_the_transcription_budget_is_reported_not_silently_dropped():

    result = extract("scan.pdf", _pdf(["", "", ""]),

                     ocr=lambda *a: "page text", max_ocr_pages=1)

    assert result.transcribed == 1

    assert "further scanned pages were not transcribed" in result.text





def test_a_scan_without_a_model_says_so(ingest_engine):

    with pytest.raises(OCRUnavailable) as exc:

        extract("scan.pdf", _pdf(["", ""]))

    assert "scan" in str(exc.value).lower()





def test_one_shared_word_is_not_a_topic():
    """Coverage is a ratio, so a short question needs proportionally fewer
    matches. Five terms needed one, which let a fishing-permit question reach a
    trade-licence chunk on the word "applying". Two distinct terms are now the
    floor for anything longer than three.

    A blank corpus with one controlled document, not the shared `corpus`
    fixture's: the real seed document (see knowledge/seed.py) is long and
    general enough to coincidentally share two-plus terms with almost any
    everyday question, which is exactly the false-positive class this test
    means to rule out — on a document actually unrelated to the query.
    """
    blank = Corpus(seed=False)
    blank.add(Document(id="fee-doc", title="Trade licence fees", source="test",
                       kind="txt", chars=40,
                       chunks=["The renewal fee is 1,200 rupees for applying."],
                       status="indexed", verdict="pass"))
    hits = blank.search("How do I apply for a fishing permit on the east coast?", 4, 0.15)
    assert hits == [], f"matched on a coincidence: {[h.doc_id for h in hits]}"


def test_a_short_question_may_still_match_on_one_term(corpus):
    """The floor lifts below four terms: 'renewal fee' is a real question and
    has only two terms to give."""
    corpus.add(Document(id="fee-doc", title="Trade licence fees", source="test",
                        kind="txt", chars=40,
                        chunks=["The renewal fee is 1,200 rupees."],
                        status="indexed", verdict="pass"))
    hits = corpus.search("renewal fee", 4, 0.15)
    assert hits, "a short, legitimate question should still retrieve"


def test_a_genuine_question_still_matches_several_terms(corpus):
    corpus.add(Document(id="trade-licence-renewal", title="Trade licence renewal",
                        source="test", kind="txt", chars=60,
                        chunks=["To renew a trade licence you must submit the "
                               "existing certificate and proof of premises."],
                        status="indexed", verdict="pass"))
    hits = corpus.search("what documents do I need to renew a trade licence", 4, 0.15)
    assert hits and "trade-licence-renewal" in hits[0].doc_id
