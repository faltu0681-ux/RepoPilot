"""Describing a change, measured rather than guessed.

Phase 2D needs to say what a run did in a form a person reads: a commit subject, a pull request
body, and a count of what changed. All three come from the same place, ``git``, because that is the
only thing in the project that actually knows what is different.

Nothing here calls a model. Every number is counted from a real diff, so the summary cannot claim
a change is smaller than it is.

The diff itself is bounded twice, and both bounds matter:

* per file, so a run that rewrote a vendored tree cannot produce a pull request body that no
  review tool will accept;
* overall, because a body is prose to a human, not an archive. A truncated diff says it was
  truncated rather than quietly ending, because a reviewer who believes they read the whole change
  is worse off than one who knows there was more.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: At most this many diff lines are kept for one file. Past that the file is summarised by its
#: statistics and the pull request body points at the branch instead of inlining a wall of text.
MAX_DIFF_LINES_PER_FILE = 400

#: At most this many diff lines across the whole change, for the same reason.
MAX_DIFF_LINES_TOTAL = 4000

#: A path as git reports it, which is always forward-slashed even on Windows.
_GIT_PATH = re.compile(r"^(?:diff --git |rename from |rename to )")

#: A header whose path git had to C-quote: ``diff --git "a/x y" "b/x y"``.
_QUOTED_HEADER = re.compile(r'^diff --git "(?P<old>.*?)" "(?P<new>.*?)"\s*$')

#: The usual header, ``diff --git a/x b/x``. Non-greedy because a path may itself contain
#: `` b/``; splitting on the last one instead would truncate a nested path. A path whose name
#: literally contains `` b/`` therefore splits wrongly -- git does not quote for a space, so there
#: is no marker to disambiguate with. ``--name-status`` stays authoritative for the names, and a
#: file this mis-splits loses its quoted patch rather than being reported under a wrong one.
_PLAIN_HEADER = re.compile(r"^diff --git a/(?P<old>.+?) b/(?P<new>.+)$")


def _header_paths(line: str) -> str | None:
    """The ``b/`` side of a ``diff --git`` header, unquoted, or None if this is not one.

    The two sides are separated by reading whichever quoting git chose rather than by splitting
    on a space. A path is quoted precisely when it contains something a plain split would mangle,
    including a space, so the quoted form is the one that cannot be split naively. Git quotes the
    whole ``a/``-prefixed token, so the prefix is stripped before the escapes are decoded --
    otherwise the escapes of the prefix and the path would be decoded together.
    """
    quoted = _QUOTED_HEADER.match(line)
    if quoted:
        return unescape_git_path(_strip_side_prefix(quoted.group("new")))
    plain = _PLAIN_HEADER.match(line.rstrip("\n"))
    if plain:
        # Not quoted, therefore not escaped: a plain path containing a backslash is a path whose
        # backslash is a backslash, and decoding it would invent an escape git never wrote.
        return plain.group("new")
    return None


def _strip_side_prefix(path: str) -> str:
    """Drop git's ``a/`` or ``b/`` marker, which is a diff-ism and not part of the name."""
    for prefix in ("a/", "b/"):
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


class DiffError(RuntimeError):
    """The change could not be described."""


@dataclass(frozen=True, slots=True)
class FileDiff:
    """What happened to one file."""

    path: str
    #: ``A``dded, ``M``odified, ``D``eleted, ``R``enamed, or the raw status git reported.
    status: str
    lines_added: int = 0
    lines_removed: int = 0
    #: The unified diff for this file, possibly empty when it was too large to keep.
    patch: str = ""
    truncated: bool = False

    @property
    def lines_changed(self) -> int:
        return self.lines_added + self.lines_removed

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "status": self.status,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "lines": self.lines_changed,
            "truncated": self.truncated,
        }


@dataclass(frozen=True, slots=True)
class Change:
    """Every file a commit touched, with the diff that explains it.

    ``empty`` is the important field: a run that wrote nothing has a change, and it is an empty
    one. Reporting "no changes" as an absence rather than as a fact would let a caller guess
    between "nothing happened" and "nothing was measured".
    """

    files: tuple[FileDiff, ...] = ()
    base: str | None = None
    head: str | None = None
    #: True when the diff was cut off at a bound and the body will say so.
    truncated: bool = False

    @property
    def empty(self) -> bool:
        return not self.files

    @property
    def lines_added(self) -> int:
        return sum(f.lines_added for f in self.files)

    @property
    def lines_removed(self) -> int:
        return sum(f.lines_removed for f in self.files)

    @property
    def paths(self) -> list[str]:
        return [f.path for f in self.files]

    def as_dict(self) -> dict[str, object]:
        return {
            "base": self.base,
            "head": self.head,
            "files": [f.as_dict() for f in self.files],
            "files_count": len(self.files),
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "empty": self.empty,
            "truncated": self.truncated,
        }


def parse_numstat(raw: str) -> tuple[int, int]:
    """Read git's ``--numstat`` output into (added, removed).

    Git writes ``-`` instead of a number for a binary file, and writing that as 0 would quietly
    report a re-encoded image as an empty change, so a binary file counts as no lines.
    """
    added = removed = 0
    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        if parts[0].isdigit():
            added += int(parts[0])
        if parts[1].isdigit():
            removed += int(parts[1])
    return added, removed


def parse_name_status(raw: str) -> dict[str, str]:
    """Read git's ``--name-status`` output into ``{path: status}``.

    A rename is reported against its destination, because that is the path a reader cares about
    and the path the change is attributed to.

    Git wraps a path in double quotes and escapes it as C string literals whenever the name holds
    anything a plain terminal would mangle -- a non-ASCII character, a quote, a tab. Left quoted,
    such a name would be copied straight into a commit message and a pull request body as
    ``"src/caf\\303\\251.py"``, which names a file that does not exist. Git only quotes for this
    reason, so the quoting is always stripped rather than treated as part of the name.
    """
    statuses: dict[str, str] = {}
    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0][:1] or parts[0]
        statuses[unquote_git_path(parts[-1])] = status
    return statuses


def unescape_git_path(value: str) -> str:
    """Decode the C escapes git writes inside a path it had to quote.

    Git escapes a non-ASCII character as its UTF-8 bytes in octal, so ``caf\\303\\251.py`` is one
    file called ``café.py`` and not six characters of nonsense. Decoded through latin-1 rather
    than utf-8 on purpose: ``unicode_escape`` has already turned each ``\\NNN`` into the code point
    of that byte, and re-encoding as latin-1 recovers the original byte sequence for one real
    decode. Left as-is when there is nothing to decode, so a plain path containing a literal
    backslash is not rewritten by an escape git never wrote.
    """
    if "\\" not in value:
        return value
    try:
        decoded = bytes(value, "utf-8").decode("unicode_escape")
        return decoded.encode("latin-1").decode("utf-8", errors="replace")
    except (UnicodeDecodeError, ValueError):  # pragma: no cover - malformed quoting
        return value


def unquote_git_path(value: str) -> str:
    """Undo git's C-style quoting of a path, leaving an unquoted one untouched.

    Falls back to the original string if the quoting turns out not to be well-formed, because a
    mangled-but-passable path beats an exception for something that is only ever used to build a
    sentence.
    """
    if len(value) < 2 or not (value.startswith('"') and value.endswith('"')):
        return value
    return unescape_git_path(value[1:-1])


def split_diff(raw: str) -> dict[str, list[str]]:
    """Split a combined diff into ``{path: [that file's hunks]}``.

    Keyed by the ``b/`` side of the ``diff --git`` header, which is the path after any rename,
    and unquoted for the same reason :func:`parse_name_status` unquotes: a header git had to
    escape names the same file as the one the status output already named, and the two have to be
    the same string for the hunks to be found.
    """
    per_file: dict[str, list[str]] = {}
    current: str | None = None
    for line in raw.splitlines(keepends=True):
        path = _header_paths(line)
        if path is not None:
            current = path
            per_file.setdefault(current, [])
            continue
        match = _GIT_PATH.match(line)
        if match and match.group(0).strip() != "diff --git":
            # ``rename from``/``rename to``: the path is the rest of the line, and git quotes it
            # under exactly the same rules as the header.
            current = unquote_git_path(line.split(" ", 2)[-1].strip())
            per_file.setdefault(current, [])
            continue
        if current is not None:
            per_file[current].append(line)
    return per_file


def assemble_change(
    *,
    name_status: str,
    numstat: str,
    patch: str,
    base: str | None = None,
    head: str | None = None,
) -> Change:
    """Build a :class:`Change` from git's three views of the same commit range.

    Three because no single one is enough: ``--name-status`` knows a rename from an edit,
    ``--numstat`` has the counts git already computed, and the raw diff has the hunks. Taking the
    counts from git rather than counting ``+`` lines is what keeps a diff line inside a string
    literal from being counted as a change.
    """
    statuses = parse_name_status(name_status)
    # Keyed by the unquoted path, for the same reason ``statuses`` is: a name quoted in one git
    # output and unquoted in another would silently fail to join, and a file that matched with no
    # counts would be reported as an empty edit.
    counts = {}
    for line in numstat.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            counts[unquote_git_path(parts[-1])] = parts[:2]
    hunks = split_diff(patch)

    files: list[FileDiff] = []
    truncated = False
    budget = MAX_DIFF_LINES_TOTAL

    for path, status in statuses.items():
        raw_added, raw_removed = counts.get(path, ("0", "0"))
        added = int(raw_added) if raw_added.isdigit() else 0
        removed = int(raw_removed) if raw_removed.isdigit() else 0

        lines = hunks.get(path, [])
        kept = lines[: min(len(lines), MAX_DIFF_LINES_PER_FILE, max(budget, 0))]
        cut = len(kept) < len(lines)
        truncated = truncated or cut
        budget -= len(kept)

        files.append(
            FileDiff(
                path=path,
                status=status,
                lines_added=added,
                lines_removed=removed,
                patch="".join(kept),
                truncated=cut,
            )
        )

    return Change(files=tuple(files), base=base, head=head, truncated=truncated)


@dataclass(frozen=True, slots=True)
class CommitMessage:
    """The subject and body to commit under.

    Built here rather than in the step so it can be asserted on directly: the phrasing of a commit
    a human reads is a contract with that human, not an implementation detail of the pipeline.
    """

    subject: str
    body: str = ""
    #: Coerced to a single line and length-checked by git anyway, but done here so the failure
    #: names the run rather than arriving as a git error.
    max_length: int = field(default=72)

    def as_args(self) -> list[str]:
        return ["-m", self.subject] + (["-m", self.body] if self.body else [])

    def as_text(self) -> str:
        return f"{self.subject}\n\n{self.body}" if self.body else self.subject


def build_commit_message(
    *,
    task: str,
    verdict: str,
    verdict_reason: str,
    change: Change,
    test_framework: str | None = None,
    test_status: str | None = None,
) -> CommitMessage:
    """The subject and body to commit a run's change under.

    The subject is the task, in the imperative and on one line, because that is what shows in
    ``git log`` and in a pull request's history. The verdict is deliberately *not* in the subject:
    a commit message that says "VERIFIED" invites a reader to trust the message instead of the
    suite, which is the opposite of what the run actually established.

    Attribution is appended as a trailer rather than woven into the prose, so the message stays
    readable and the fact is machine-findable.
    """
    subject = _subject_for(task)
    # The same three distinct labels the pull request body uses. "Not verified" is deliberately
    # not one of them: it reads equally well for a change that broke the suite and for a change
    # nobody could test, and those two outcomes mean opposite things to a reviewer.
    verdict_label = {
        "verified": "VERIFIED",
        "failed": "FAILED",
        "unverified": "UNVERIFIED",
    }[verdict]

    lines = [task.strip() or f"RepoPilot run for {verdict_label.lower()} change", ""]

    if change.empty:
        lines.append("No files were changed.")
    else:
        lines.append(
            f"{len(change.files)} file(s) changed, +{change.lines_added} "
            f"/ -{change.lines_removed} lines."
        )
        for entry in change.files:
            lines.append(f"- {entry.status} {entry.path}")

    if verdict_reason:
        lines += ["", verdict_reason]

    # The verdict is stated unconditionally, on its own line, whatever else is known. It used to
    # ride along with the test line, which meant a run with no detected runner committed a message
    # saying nothing at all about whether anything was verified -- the one case where the reader
    # most needs to be told.
    if test_framework and test_status:
        lines += ["", f"{verdict_label}: ran the {test_framework} suite, {test_status}."]
    else:
        lines += ["", f"{verdict_label}: no test suite was run against this change."]

    lines += ["", "Produced-by: RepoPilot"]
    return CommitMessage(subject=subject, body="\n".join(lines))


def _subject_for(task: str) -> str:
    """One line, first line, no trailing full stop, as a commit subject wants."""
    line = " ".join(task.strip().split())
    if not line:
        return "RepoPilot run"
    if line.endswith((".", "!", "?")):
        line = line[:-1]
    # Cut on a word boundary rather than mid-word: a subject truncated through a word is both
    # unreadable and misleading about what the commit does.
    if len(line) > 72:
        line = line[:72].rsplit(" ", 1)[0].rstrip(" ,;:")
    return line or "RepoPilot run"


def render_pull_request_body(
    *,
    task: str,
    verdict: str,
    verdict_reason: str,
    plan_summary: str,
    change: Change,
    test_framework: str | None,
    test_command: list[str],
    test_status: str | None,
    tests_passed: int | None,
    tests_failed: int | None,
    tests_skipped: int | None,
    unverified_reason: str | None,
    diagnosis_summary: str = "",
    repair_rounds: int = 0,
    pre_existing_failures: int = 0,
) -> str:
    """The pull request body.

    Written for the reviewer who has to decide whether to merge this, so it leads with what the
    change is and whether it was proved, then gives the evidence. The diff is inlined only when it
    is small enough to be readable; otherwise the body says how big it is and leaves the branch to
    GitHub's own diff view rather than pasting something that will not render.
    """
    # Upper case, because this is the word a reviewer scans for and the three are genuinely
    # different promises. In particular FAILED and UNVERIFIED must not render alike: one says
    # the code is wrong, the other says nobody found out.
    heading = {
        "verified": "VERIFIED",
        "failed": "FAILED",
        "unverified": "UNVERIFIED",
    }[verdict]

    lines: list[str] = [f"## {heading}", "", verdict_reason or "No verdict was recorded.", ""]

    lines += ["## Task", "", task or "_No task was recorded._", ""]

    if plan_summary:
        lines += ["## Approach", "", plan_summary, ""]

    if change.empty:
        lines += ["## Changes", "", "_No files were changed._", ""]
    else:
        lines += [
            "## Changes",
            "",
            f"{len(change.files)} file(s), +{change.lines_added} / -{change.lines_removed} lines.",
            "",
        ]
        for entry in change.files:
            mark = {"A": "new", "M": "edited", "D": "deleted", "R": "renamed"}.get(
                entry.status, entry.status
            )
            suffix = " (diff too large to inline)" if entry.truncated else ""
            lines.append(
                f"- `{entry.path}` - {mark}, +{entry.lines_added} / -{entry.lines_removed}{suffix}"
            )
        lines.append("")

    lines += ["## Verification", ""]
    if test_framework and test_status:
        counts = []
        if tests_passed is not None:
            counts.append(f"{tests_passed} passed")
        if tests_failed is not None:
            counts.append(f"{tests_failed} failed")
        if tests_skipped is not None:
            counts.append(f"{tests_skipped} skipped")
        tally = f" ({', '.join(counts)})" if counts else ""
        command = " ".join(test_command)
        lines += [f"Ran the {test_framework} suite: `{command}` -> **{test_status}**{tally}.", ""]
    elif unverified_reason:
        lines += [f"**Not verified.** {unverified_reason}", ""]
    else:
        lines += ["**Not verified.** No test suite was run for this change.", ""]

    if repair_rounds or diagnosis_summary or pre_existing_failures:
        lines += ["## Diagnosis", ""]
        if diagnosis_summary:
            lines += [diagnosis_summary, ""]
        if repair_rounds:
            lines += [f"The run made {repair_rounds} repair round(s) before this result.", ""]
        if pre_existing_failures:
            lines += [
                f"{pre_existing_failures} failure(s) were judged pre-existing rather than caused "
                "by this change.",
                "",
            ]

    # Only inline the diff when it is small enough to read. A reviewer reads the branch's own diff
    # view for anything larger, and a body that truncates silently is worse than one that does not.
    inlineable = (
        not change.empty
        and not change.truncated
        and change.lines_added + change.lines_removed <= 400
    )
    if inlineable:
        lines += ["## Diff", "", "```diff"]
        for entry in change.files:
            if entry.patch:
                lines.append(entry.patch.rstrip("\n"))
        lines += ["```", ""]
    elif not change.empty:
        lines += [
            "## Diff",
            "",
            f"Not inlined: {change.lines_added + change.lines_removed} changed lines. "
            "Review the branch's diff on GitHub.",
            "",
        ]

    lines += [
        "---",
        "",
        "Opened by RepoPilot. The change was produced and verified in an isolated checkout; "
        "this branch is the whole of it.",
    ]
    return "\n".join(lines)


__all__ = [
    "MAX_DIFF_LINES_PER_FILE",
    "MAX_DIFF_LINES_TOTAL",
    "Change",
    "CommitMessage",
    "DiffError",
    "FileDiff",
    "assemble_change",
    "build_commit_message",
    "parse_name_status",
    "parse_numstat",
    "render_pull_request_body",
    "split_diff",
]
