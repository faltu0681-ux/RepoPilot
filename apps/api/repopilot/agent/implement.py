"""Turning one planned file into a real file: Phase 2B.

Two things have to be true for a write to be allowed, and both are enforced here rather than
promised in a prompt:

* **The path is inside this run's checkout.** A plan can propose any path, and a model can
  invent one, so every write goes through :func:`resolve_write_target`, which rejects absolute
  paths, ``..``, anything under ``.git``, and credential files, then re-checks containment after
  resolving symlinks. There is no code path that writes outside ``root``.
* **The content is the whole file.** What the model returns is what lands on disk, so it is
  parsed to a body, checked for the usual ways a reply stops being a file (a preamble, a code
  fence left in, a "rest unchanged" elision), and rejected with a reason rather than written.

The model is asked for one file per call. Splitting the work this way keeps each request
small enough to answer completely, and it means a file the model gets wrong costs one retry
instead of the whole change.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from repopilot.agent.llm import ChatClient, ChatMessage, LLMError, extract_json_object
from repopilot.agent.prompts import build_implementer_prompt
from repopilot.agent.schema import FileChange, ImplementationPlan
from repopilot.analysis.profile import read_source
from repopilot.core.errors import RepopilotError

logger = logging.getLogger("repopilot.agent.implement")

#: How many times to re-ask after a reply that is not usable file content.
MAX_CHANGE_ATTEMPTS = 2

#: Refuse to write a file larger than this. Beyond it, a reply is more likely to be a
#: truncated or looping generation than a real source file.
MAX_CONTENT_CHARS = 200_000

#: A file bigger than this is not rewritten. Truncating the input to fit would invite the
#: model to return a truncated file, which would destroy the rest of it, so the file is left
#: alone and the run says so.
MAX_SOURCE_CHARS = 60_000

#: Files that hold credentials. A plan that names one is refused rather than written: a run has
#: no reason to create or overwrite a key, and a model that proposes doing so has misread the
#: task.
SECRET_NAMES = frozenset(
    {
        ".env",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "credentials",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "secrets.json",
        "secrets.yaml",
        "secrets.yml",
    }
)
SECRET_SUFFIXES = frozenset({".key", ".p12", ".pem", ".pfx"})

#: The markdown fence models wrap file contents in.
_FENCE = "```"

#: A fence's language tag: one short word, no spaces or punctuation a line of code would have.
_LANGUAGE_TAG = re.compile(r"^[A-Za-z0-9_+#.-]{1,20}$")


class UnsafePathError(RepopilotError):
    """A path was refused because writing it would leave the run's checkout."""


class ChangeRejectedError(RuntimeError):
    """The model did not return usable file content."""


@dataclass(frozen=True, slots=True)
class WriteResult:
    """What a successful write actually did on disk."""

    path: str
    target: Path
    created: bool
    bytes: int
    lines: int


# --------------------------------------------------------------------- path safety


def resolve_write_target(root: Path, relative: str) -> Path:
    """Return the absolute path to write, or refuse.

    Every check is on the string the plan supplied, because a path is rejected for what it
    *says* as well as for where it lands: ``a/../b`` inside the checkout is still a path that
    was written to escape the check, and the containment test below would accept it.
    """
    normalised = relative.replace("\\", "/").strip()
    if not normalised:
        raise UnsafePathError("refusing to write an empty path")

    parts = PurePosixPath(normalised).parts
    if normalised.startswith("/") or PureWindowsPath(normalised).drive:
        raise UnsafePathError(f"refusing to write the absolute path '{relative}'")
    if ".." in parts:
        raise UnsafePathError(f"refusing to write '{relative}': it leaves the workspace")
    if any(part == ".git" for part in parts):
        raise UnsafePathError(f"refusing to write '{relative}': it is inside the git directory")

    name = parts[-1]
    suffix = PurePosixPath(name).suffix.lower()
    if name in SECRET_NAMES or name.startswith(".env.") or suffix in SECRET_SUFFIXES:
        raise UnsafePathError(f"refusing to write '{relative}': it is a credentials file")

    target = (root / normalised).resolve()
    if not target.is_relative_to(root.resolve()):
        raise UnsafePathError(f"refusing to write '{relative}': it resolves outside the workspace")
    return target


def read_for_edit(root: Path, relative: str, *, max_chars: int = MAX_SOURCE_CHARS) -> str | None:
    """Read the current contents of a planned file, or ``None`` if it is not safe to rewrite.

    ``None`` covers all three reasons the caller must not proceed: the path is not a file, it is
    refused, or it is too large to hand to a model without risking a truncated rewrite.
    """
    try:
        target = resolve_write_target(root, relative)
    except UnsafePathError:
        return None
    if not target.is_file():
        return None
    text = read_source(root, relative)
    if text is None or len(text) > max_chars:
        return None
    return text


def write_file(root: Path, relative: str, content: str) -> WriteResult:
    """Write one file inside the checkout, creating parent directories as needed."""
    if len(content) > MAX_CONTENT_CHARS:
        raise ChangeRejectedError(
            f"refusing to write {len(content)} characters to {relative}; the limit is "
            f"{MAX_CONTENT_CHARS}"
        )

    target = resolve_write_target(root, relative)
    created = not target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    # ``newline=""`` because the model's content is a file, not a string to be re-wrapped: on
    # Windows the default translation would rewrite every line ending in the source.
    target.write_text(content, encoding="utf-8", newline="")

    return WriteResult(
        path=PurePosixPath(relative.replace("\\", "/")).as_posix(),
        target=target,
        created=created,
        bytes=len(content.encode("utf-8")),
        lines=len(content.splitlines()),
    )


# ------------------------------------------------------------------ model content


async def generate_file_change(
    client: ChatClient,
    *,
    task: str,
    repository: str,
    plan: ImplementationPlan,
    path: str,
    role: str,
    current: str,
    siblings: list[str] | None = None,
    temperature: float = 0.0,
) -> FileChange:
    """Ask the model for the new contents of one file.

    Raises :class:`ChangeRejectedError` when the reply cannot be turned into a file body, and
    ``LLMError`` when the backend itself failed.
    """
    turns = build_implementer_prompt(
        task=task,
        repository=repository,
        plan=plan,
        path=path,
        role=role,
        current=current,
        siblings=siblings,
    )
    messages = [ChatMessage(role=role_, content=content) for role_, content in turns]

    last_error = ""
    for attempt in range(1, MAX_CHANGE_ATTEMPTS + 1):
        try:
            completion = await client.complete(messages, temperature=temperature)
        except LLMError:
            raise  # transport and auth problems are not fixed by asking again

        try:
            content = extract_file_content(completion.text, current=current)
        except ChangeRejectedError as exc:
            last_error = str(exc)
            logger.warning("change for %s rejected (attempt %d): %s", path, attempt, last_error)
            messages = [
                *messages,
                ChatMessage(role="assistant", content=completion.text[:2_000]),
                ChatMessage(
                    role="user",
                    content=(
                        f"That was not a usable file ({last_error}). Reply with the complete "
                        "contents of the file and nothing else."
                    ),
                ),
            ]
            continue

        return FileChange(path=path, role=role, content=content)  # type: ignore[arg-type]

    raise ChangeRejectedError(
        f"the model did not return usable contents for {path} after {MAX_CHANGE_ATTEMPTS} "
        f"attempts ({last_error})"
    )


def extract_file_content(text: str, *, current: str = "") -> str:
    """Pull a file body out of a model reply.

    Models reply in a handful of shapes: a JSON object with the body in it, a fenced block, or
    the bare file. All three are accepted. What is not accepted is a reply that also contains
    prose, because that prose is what would end up as the first line of the file.
    """
    stripped = text.strip()
    if not stripped:
        raise ChangeRejectedError("the model returned an empty response")

    if stripped[0] in "{[":
        try:
            payload = extract_json_object(stripped)
        except LLMError as exc:
            raise ChangeRejectedError(str(exc)) from exc
        for key in ("content", "file", "code", "text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return _reject_elision(value, current)
        raise ChangeRejectedError("the JSON reply had no 'content' field holding the file")

    if stripped.startswith(_FENCE):
        return _reject_elision(_unfence(stripped), current)

    if "\n" in stripped:
        first = stripped.split("\n", 1)[0].strip().lower()
        if _looks_like_preamble(first):
            raise ChangeRejectedError(
                f"the reply started with prose ('{first[:60]}') instead of the file"
            )

    return _reject_elision(stripped, current)


def _unfence(text: str) -> str:
    """Strip one surrounding markdown fence, or refuse when the block is not closed."""
    body = text[len(_FENCE) :]
    head, newline, rest = body.partition("\n")
    if newline and _LANGUAGE_TAG.match(head.strip()):
        body = rest  # drop the language tag on the opening fence
    closing = body.rfind(_FENCE)
    if closing == -1:
        raise ChangeRejectedError("the reply opened a code fence and never closed it")
    return body[:closing].strip("\n")


#: Replies that open with a sentence addressed to the reader instead of code. Deliberately
#: narrow: a marker broad enough to catch "this module..." would also catch a documentation
#: file that legitimately starts that way, and a refused write is a better failure than a
#: source file with a sentence on line one.
_PREAMBLE_STARTS = (
    "here is",
    "here's",
    "heres",
    "here are",
    "sure",
    "certainly",
    "of course",
    "as requested",
    "below is",
    "below are",
    "i've ",
    "i have ",
    "i can ",
    "i will ",
    "i cannot",
    "the updated",
    "updated file",
    "updated version",
    "note:",
)

#: The elisions a model reaches for when it is running out of room. Writing one of these into a
#: file would silently delete everything after it.
_ELISION_MARKERS = (
    "// rest of the file unchanged",
    "# rest of the file unchanged",
    "/* rest of the file unchanged */",
    "... rest of the file",
    "...rest of the file",
    "rest of file unchanged",
    "rest of the file unchanged",
    "[... unchanged]",
    "// ... rest unchanged",
)


def _looks_like_preamble(line: str) -> bool:
    return any(line.startswith(marker) for marker in _PREAMBLE_STARTS)


def validate_whole_file(content: str, *, current: str = "", path: str = "the file") -> str:
    """Check that ``content`` really is the whole file, and return it normalised.

    The same rules the implementer's replies go through, exposed for Phase 2C: a repair arrives as
    one field of a larger JSON object rather than as the whole reply, but it lands on disk exactly
    the same way, so it is held to exactly the same standard. A partial file is a syntax error at
    best and a silent deletion at worst.
    """
    try:
        return _reject_elision(content, current)
    except ChangeRejectedError as exc:
        raise ChangeRejectedError(f"refusing to write {path}: {exc}") from exc


def _reject_elision(content: str, current: str) -> str:
    """Reject a body that claims to be a summary of a file rather than the file."""
    if not content.strip():
        raise ChangeRejectedError("the model returned no file content")
    if len(content) > MAX_CONTENT_CHARS:
        raise ChangeRejectedError(
            f"the reply is {len(content)} characters, over the {MAX_CONTENT_CHARS} limit"
        )

    lowered = content.lower()
    for marker in _ELISION_MARKERS:
        if marker in lowered:
            raise ChangeRejectedError(
                f"the reply elides the rest of the file ('{marker}'); a partial file is never "
                "written"
            )
    if "```" in content:
        raise ChangeRejectedError("the reply contains a code fence, so it is not a bare file")

    if current and content.strip() == current.strip():
        raise ChangeRejectedError("the reply is identical to the file it was asked to change")

    if not content.endswith("\n"):
        content += "\n"
    return content
