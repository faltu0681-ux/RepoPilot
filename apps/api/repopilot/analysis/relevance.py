"""Turning a natural-language task into a shortlist of files worth reading.

The shortlist is produced without a model so that the expensive reasoning step only ever sees
plausible candidates, and so that a run can explain *why* a file was considered. The scores are
heuristic on purpose: they are reproducible, free, and good enough to shrink a repository with
thousands of files down to a few dozen. The model then makes the final, grounded call in
:mod:`repopilot.agent.plan`.

Nothing here reads a file's full contents by default. Path and name signals are cheap and cover
most real cases; content is only consulted for the top candidates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from repopilot.analysis.profile import RepositoryProfile, SourceFile, read_source

#: Words that carry no signal about which file to open. Removing them stops "add" from
#: matching every path containing "add".
STOP_WORDS = frozenset(
    {
        "a",
        "add",
        "added",
        "adding",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "could",
        "do",
        "does",
        "each",
        "fix",
        "for",
        "from",
        "get",
        "has",
        "have",
        "how",
        "i",
        "if",
        "implement",
        "implementation",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "make",
        "me",
        "my",
        "need",
        "new",
        "of",
        "on",
        "or",
        "our",
        "please",
        "should",
        "so",
        "some",
        "support",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "this",
        "to",
        "up",
        "use",
        "using",
        "want",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "will",
        "with",
        "you",
        "your",
    }
)

#: Filled in from the task when a word also names a language or a well-known file.
ALIASES: dict[str, str] = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "jsx": "javascript",
    "tsx": "typescript",
    "yml": "yaml",
    "md": "markdown",
    "api": "api",
    "ci": "ci",
    "cd": "cd",
}

#: CamelCase and snake_case splitting, so "RunEngine" also matches the words "run" and "engine".
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*")

#: Cap on candidates handed to the model. Enough to cover a real feature, small enough to read.
DEFAULT_CANDIDATE_LIMIT = 60

#: Cap on how much of one file's content is quoted into the prompt.
MAX_SNIPPET_CHARS = 6_000


@dataclass(frozen=True, slots=True)
class CandidateFile:
    """A file considered relevant, with the evidence for that judgement."""

    path: str
    score: float
    reasons: tuple[str, ...]
    language: str | None
    lines: int
    is_test: bool

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "score": round(self.score, 3),
            "reasons": list(self.reasons),
            "language": self.language,
            "lines": self.lines,
            "is_test": self.is_test,
        }


def tokenize(text: str) -> list[str]:
    """Split a task into lowercase keywords, expanding camelCase and dropping stop words.

    Duplicates are kept: a term repeated three times in the task is a stronger hint than one
    mentioned once, and :func:`score_file` weighs path matches by set size anyway.
    """
    words: list[str] = []
    for chunk in re.split(r"[/\\.\-_\s]+", text):
        for raw in _CAMEL.split(chunk):
            for word in _WORD.findall(raw):
                lowered = word.lower()
                if len(lowered) < 2 or lowered in STOP_WORDS:
                    continue
                words.append(ALIASES.get(lowered, lowered))
    return words


def _tokens_of_path(path: str) -> set[str]:
    """Every searchable token in a path, including its pieces."""
    lowered = path.lower()
    tokens: set[str] = set()
    for chunk in re.split(r"[/\\.\-_\s]+", lowered):
        if not chunk:
            continue
        tokens.add(ALIASES.get(chunk, chunk))
        for part in _CAMEL.split(chunk):
            if len(part) > 1:
                tokens.add(ALIASES.get(part.lower(), part.lower()))
    return tokens


def score_file(
    source: SourceFile, keywords: list[str], *, content_bonus: float = 0.0
) -> tuple[float, tuple[str, ...]]:
    """Score one file against the task's keywords. Returns the score and the reasons."""
    if not keywords:
        return 0.0, ()

    keyword_set = set(keywords)
    path_tokens = _tokens_of_path(source.path)
    name_tokens = _tokens_of_path(source.name)

    score = 0.0
    reasons: list[str] = []

    exact = keyword_set & path_tokens
    if exact:
        score += 3.0 * len(exact)
        reasons.append(f"path mentions {'/'.join(sorted(exact))}")

    in_name = keyword_set & name_tokens
    if in_name:
        score += 2.0 * len(in_name)
        reasons.append(f"filename mentions {'/'.join(sorted(in_name))}")

    if content_bonus:
        score += content_bonus
        reasons.append("keyword appears in the file's contents")

    if source.is_test:
        # A task about a module usually implies its tests; keep them nearby, not on top.
        score -= 0.5
    if source.path.count("/") == 0:
        # Top-level files are broad. They are context, not targets.
        score -= 0.25

    return max(score, 0.0), tuple(reasons)


def rank_candidates(
    profile: RepositoryProfile,
    task: str,
    *,
    root: Path | None = None,
    limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> list[CandidateFile]:
    """Rank every file against the task, best first.

    Two signals are combined. The path and filename are cheap and carry most of the weight. A
    content scan then catches the case the path pass structurally cannot: a task such as "the
    health endpoint times out" names a concept that may appear only inside a file called
    something unrelated. Every file is scanned once, with a hard per-file byte cap, so the cost
    stays bounded on a large repository.
    """
    keywords = tokenize(task)
    if not keywords:
        return []

    scored: list[tuple[float, tuple[str, ...], SourceFile]] = []
    for source in profile.files:
        score, reasons = score_file(source, keywords)
        if root is not None:
            hits = _content_hits(root, source, keywords)
            if hits:
                score += 1.5 * len(hits)
                reasons += (f"defines {'/'.join(hits)}",)
        if score > 0:
            scored.append((score, reasons, source))

    scored.sort(key=lambda item: (-item[0], item[2].path))
    return [
        CandidateFile(
            path=source.path,
            score=score,
            reasons=reasons,
            language=source.language,
            lines=source.lines,
            is_test=source.is_test,
        )
        for score, reasons, source in scored[:limit]
    ]


def _content_hits(root: Path, source: SourceFile, keywords: list[str]) -> list[str]:
    """Which keywords appear in a file's text, capped so a huge file cannot dominate."""
    text = read_source(root, source.path, max_bytes=MAX_SNIPPET_CHARS * 4)
    if not text:
        return []
    lowered = text.lower()
    hits = []
    for keyword in dict.fromkeys(keywords):
        if len(keyword) < 3:
            continue
        if keyword in lowered:
            hits.append(keyword)
        if len(hits) >= 6:
            break
    return hits


def snippet(root: Path, path: str, *, max_chars: int = MAX_SNIPPET_CHARS) -> str | None:
    """The head of a file, for grounding the model in what a candidate actually contains."""
    text = read_source(root, path)
    if text is None:
        return None
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [{len(text) - max_chars} more characters]"
