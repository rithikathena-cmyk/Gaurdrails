"""Where documents live, and how facts come back out.

    seed.py     the twenty-five built-in documents, and the retrieval entry point
    ingest.py   extraction, chunking, the corpus and its BM25 index

Ingestion is a trust boundary of its own — an uploaded document is
attacker-supplied text the model will later be asked to answer *from* — which
is why it has its own surface in the severity matrix.
"""

from .ingest import (
    Corpus,
    Document,
    Extraction,
    Hit,
    IngestError,
    OCRUnavailable,
    chunk_text,
    extension,
    extract,
    new_document_id,
    stem,
    tokens,
)
from .seed import CORPUS, active, retrieve, use

__all__ = [
    "CORPUS", "Corpus", "Document", "Extraction", "Hit", "IngestError",
    "OCRUnavailable", "active", "chunk_text", "extension", "extract",
    "new_document_id", "retrieve", "stem", "tokens", "use",
]
