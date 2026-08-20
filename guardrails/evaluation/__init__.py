"""Measuring it.

    suite.py      the labelled evaluation — retrieval, rails, answers
    scenarios.py  five end-to-end runs that assert on the real stack

Separate because they answer different questions. The suite asks "is this still
as good as it was"; the scenarios ask "does the thing actually do what the
README says", end to end, with a model in the loop.
"""

from . import scenarios
from .suite import (
    AnswerCase,
    EvalError,
    RailCase,
    Report,
    RetrievalCase,
    Section,
    Suite,
    load_suite,
    run,
)

__all__ = [
    "AnswerCase", "EvalError", "RailCase", "Report", "RetrievalCase",
    "Section", "Suite", "load_suite", "run", "scenarios",
]
