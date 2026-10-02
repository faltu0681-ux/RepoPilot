"""Deterministic inspection of a cloned repository.

Nothing in here calls a model. Every fact is measured from the checkout, which is what makes
it safe to use as grounding for the reasoning step and cheap enough to re-run.
"""

from repopilot.analysis.diffing import (
    MAX_DIFF_LINES_PER_FILE,
    MAX_DIFF_LINES_TOTAL,
    Change,
    CommitMessage,
    DiffError,
    FileDiff,
    assemble_change,
    build_commit_message,
    parse_name_status,
    parse_numstat,
    render_pull_request_body,
    split_diff,
)
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
    "MAX_DIFF_LINES_PER_FILE",
    "MAX_DIFF_LINES_TOTAL",
    "CandidateFile",
    "Change",
    "CommitMessage",
    "DiffError",
    "FileDiff",
    "ProfileError",
    "RepositoryProfile",
    "SourceFile",
    "assemble_change",
    "build_commit_message",
    "build_profile",
    "is_ignored",
    "iter_source_files",
    "looks_like_test",
    "parse_name_status",
    "parse_numstat",
    "rank_candidates",
    "read_source",
    "render_pull_request_body",
    "score_file",
    "split_diff",
    "tokenize",
]
