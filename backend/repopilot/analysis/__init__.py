"""Deterministic inspection of a cloned repository.

Nothing in here calls a model. Every fact is measured from the checkout, which is what makes
it safe to use as grounding for the reasoning step and cheap enough to re-run.
"""

from repopilot.analysis.profile import (
    IGNORED_DIRECTORIES,
    ProfileError,
    RepositoryProfile,
    SourceFile,
    build_profile,
    is_ignored,
    iter_source_files,
    looks_like_test,
    read_source,
)
from repopilot.analysis.relevance import (
    CandidateFile,
    rank_candidates,
    score_file,
    tokenize,
)

__all__ = [
    "IGNORED_DIRECTORIES",
    "CandidateFile",
    "ProfileError",
    "RepositoryProfile",
    "SourceFile",
    "build_profile",
    "is_ignored",
    "iter_source_files",
    "looks_like_test",
    "rank_candidates",
    "read_source",
    "score_file",
    "tokenize",
]
