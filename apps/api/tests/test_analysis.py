"""Tests for the deterministic half of Phase 2A.

Everything here runs against real files on disk in a temporary directory. There is no model
involved and nothing is stubbed, because this layer is supposed to be exactly reproducible.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from repopilot.analysis.profile import (
    ProfileError,
    build_profile,
    is_ignored,
    iter_source_files,
    looks_like_test,
    read_source,
)
from repopilot.analysis.relevance import rank_candidates, snippet, tokenize


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """A small multi-language repository with the layout details that matter."""
    root = tmp_path / "repo"
    (root / "src" / "billing").mkdir(parents=True)
    (root / "src" / "auth").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "node_modules" / "left-pad").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / ".venv" / "lib").mkdir(parents=True)

    (root / "README.md").write_text("# billing service\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "billing"\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n',
        encoding="utf-8",
    )
    (root / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (root / "src" / "billing" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "billing" / "invoice.py").write_text(
        "def create_invoice(total):\n    return {'total': total}\n", encoding="utf-8"
    )
    (root / "src" / "billing" / "refunds.py").write_text(
        "def refund(invoice):\n    return 'refunded'\n", encoding="utf-8"
    )
    (root / "src" / "auth" / "session.py").write_text(
        "def login(user, password):\n    return True\n", encoding="utf-8"
    )
    (root / "src" / "auth" / "index.ts").write_text(
        "export const login = () => true;\n", encoding="utf-8"
    )
    (root / "tests" / "test_invoice.py").write_text(
        "def test_create_invoice():\n    pass\n", encoding="utf-8"
    )
    (root / "node_modules" / "left-pad" / "index.js").write_text("x=1\n", encoding="utf-8")
    (root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (root / ".venv" / "lib" / "big.py").write_text("y = 2\n", encoding="utf-8")
    (root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return root


class TestBuildProfile:
    def test_measures_real_source_files(self, sample_repo: Path) -> None:
        profile = build_profile(sample_repo, default_branch="main")

        paths = {f.path for f in profile.files}
        assert "src/billing/invoice.py" in paths
        assert "src/billing/refunds.py" in paths
        assert "src/auth/index.ts" in paths
        assert profile.default_branch == "main"

    def test_counts_languages_by_source_file(self, sample_repo: Path) -> None:
        profile = build_profile(sample_repo)

        # 5 Python source files (4 under src, 1 test) and 1 TypeScript.
        # Manifests and configs are context, not source.
        assert profile.languages == (("Python", 5), ("TypeScript", 1))
        assert profile.primary_language == "Python"

    def test_records_line_counts(self, sample_repo: Path) -> None:
        profile = build_profile(sample_repo)

        invoice = profile.by_path("src/billing/invoice.py")
        assert invoice is not None
        assert invoice.lines == 2
        assert invoice.language == "Python"

    def test_never_descends_into_noise_directories(self, sample_repo: Path) -> None:
        profile = build_profile(sample_repo)

        paths = {f.path for f in profile.files}
        assert not any(p.startswith("node_modules/") for p in paths)
        assert not any(p.startswith(".git/") for p in paths)
        assert not any(p.startswith(".venv/") for p in paths)

    def test_manifests_and_configs_are_recorded_separately(self, sample_repo: Path) -> None:
        profile = build_profile(sample_repo)

        assert "pyproject.toml" in profile.manifests
        assert "pytest.ini" in profile.test_configs
        assert "README.md" in profile.docs
        # They are context, not edit targets, so they are not source files.
        assert "pyproject.toml" not in {f.path for f in profile.files}

    def test_identifies_test_files(self, sample_repo: Path) -> None:
        profile = build_profile(sample_repo)

        assert "tests/test_invoice.py" in profile.test_files
        assert "src/billing/invoice.py" not in profile.test_files

    def test_lists_top_level_entries(self, sample_repo: Path) -> None:
        profile = build_profile(sample_repo)

        assert "src" in profile.top_level
        assert "README.md" in profile.top_level
        assert not any(name.startswith(".") for name in profile.top_level)

    def test_rejects_a_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ProfileError, match="does not exist"):
            build_profile(tmp_path / "nope")

    def test_rejects_a_directory_with_nothing_in_it(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(ProfileError, match="no source files or manifests"):
            build_profile(empty)

    def test_accepts_a_manifest_only_repository(self, tmp_path: Path) -> None:
        root = tmp_path / "docs-only"
        root.mkdir()
        (root / "Cargo.toml").write_text('[package]\nname = "x"\n', encoding="utf-8")

        profile = build_profile(root)

        # The recorded name must be the one on disk, not the lowercase table entry.
        assert profile.manifests == ("Cargo.toml",)
        assert profile.files == ()

    def test_respects_the_scan_limit(self, sample_repo: Path) -> None:
        files, _ = iter_source_files(sample_repo, limit=2)

        assert len(files) == 2

    def test_by_path_returns_none_for_unknown_paths(self, sample_repo: Path) -> None:
        profile = build_profile(sample_repo)

        assert profile.by_path("src/billing/nope.py") is None


class TestPathHelpers:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("tests/test_core.py", True),
            ("src/tests/test_core.py", True),
            ("src/__tests__/core.ts", True),
            ("core.test.ts", True),
            ("core_spec.rb", True),
            ("internal/thing_test.go", True),
            ("spec_helper.rb", True),
            ("src/billing/invoice.py", False),
            ("src/testing/helper.py", False),  # "testing" is not a test directory
        ],
    )
    def test_looks_like_test(self, path: str, expected: bool) -> None:
        assert looks_like_test(path) is expected

    def test_is_ignored_looks_at_parents_only(self, sample_repo: Path) -> None:
        assert is_ignored(sample_repo / "node_modules" / "left-pad" / "index.js", sample_repo)
        assert not is_ignored(sample_repo / "src" / "billing" / "invoice.py", sample_repo)

    def test_read_source_refuses_to_escape_the_workspace(self, sample_repo: Path) -> None:
        """The containment check is what keeps analysis off the user's own machine."""
        secret = sample_repo.parent / "outside.py"
        secret.write_text("SECRET = 1\n", encoding="utf-8")

        assert read_source(sample_repo, "../outside.py") is None
        assert read_source(sample_repo, "src/../../outside.py") is None

    def test_read_source_reads_a_real_file(self, sample_repo: Path) -> None:
        text = read_source(sample_repo, "src/billing/invoice.py")

        assert text is not None
        assert "def create_invoice" in text

    def test_read_source_returns_none_for_missing_files(self, sample_repo: Path) -> None:
        assert read_source(sample_repo, "src/billing/absent.py") is None

    def test_snippet_truncates_and_marks_the_cut(self, sample_repo: Path) -> None:
        text = snippet(sample_repo, "src/billing/invoice.py", max_chars=10)

        assert text is not None
        assert text.endswith("more characters]")
        assert text.startswith("def create")


class TestTokenize:
    def test_drops_stop_words(self) -> None:
        assert tokenize("add a new endpoint to the server") == ["endpoint", "server"]

    def test_splits_camel_case(self) -> None:
        assert "engine" in tokenize("please fix RunEngine retry logic")
        assert "retry" in tokenize("please fix RunEngine retry logic")

    def test_splits_paths_and_snake_case(self) -> None:
        tokens = tokenize("edit src/billing/invoice_processor.py")

        assert "billing" in tokens
        assert "invoice" in tokens
        assert "processor" in tokens

    def test_expands_known_aliases(self) -> None:
        assert "javascript" in tokenize("rename the js helper")
        assert "typescript" in tokenize("update the ts types")

    def test_ignores_single_characters(self) -> None:
        assert tokenize("a b c") == []


class TestRankCandidates:
    def test_ranks_the_module_the_task_names_first(self, sample_repo: Path) -> None:
        profile = build_profile(sample_repo)

        ranked = rank_candidates(profile, "fix the invoice calculation", root=sample_repo)

        assert ranked[0].path == "src/billing/invoice.py"
        assert ranked[0].score > 0
        assert any("invoice" in reason for reason in ranked[0].reasons)

    def test_path_and_content_signals_combine(self, sample_repo: Path) -> None:
        """A file named after the task's concept outranks one that merely mentions it."""
        profile = build_profile(sample_repo)

        # Both define a login, but only session.py is *named* for the concept.
        ranked = rank_candidates(profile, "fix the session login", root=sample_repo)
        paths = [c.path for c in ranked]
        scores = {c.path: c.score for c in ranked}

        assert paths[0] == "src/auth/session.py"
        assert scores["src/auth/session.py"] > scores["src/auth/index.ts"]

    def test_breaks_ties_deterministically_by_path(self, sample_repo: Path) -> None:
        """Two files with identical evidence must always order the same way."""
        profile = build_profile(sample_repo)
        first = [c.path for c in rank_candidates(profile, "login", root=sample_repo)]
        second = [c.path for c in rank_candidates(profile, "login", root=sample_repo)]

        assert first == second == sorted(first)

    def test_content_boost_lifts_a_file_whose_name_does_not_match(self, sample_repo: Path) -> None:
        """A concept that only appears inside a file must still surface it."""
        profile = build_profile(sample_repo)
        by_name = {c.path: c.score for c in rank_candidates(profile, "login")}
        with_content = {
            c.path: c.score for c in rank_candidates(profile, "login", root=sample_repo)
        }

        # index.ts never mentions "login" in its path, only in its body.
        assert by_name.get("src/auth/index.ts", 0.0) == 0.0
        assert with_content["src/auth/index.ts"] > 0.0
        assert any(
            "defines login" in reason
            for candidate in rank_candidates(profile, "login", root=sample_repo)
            if candidate.path == "src/auth/index.ts"
            for reason in candidate.reasons
        )

    def test_ranks_related_test_files_below_the_module(self, sample_repo: Path) -> None:
        profile = build_profile(sample_repo)

        ranked = rank_candidates(profile, "change create_invoice", root=sample_repo)
        paths = [c.path for c in ranked]

        assert paths.index("src/billing/invoice.py") < paths.index("tests/test_invoice.py")

    def test_returns_nothing_when_no_keyword_survives(self, sample_repo: Path) -> None:
        profile = build_profile(sample_repo)

        assert rank_candidates(profile, "add a", root=sample_repo) == []

    def test_respects_the_limit(self, sample_repo: Path) -> None:
        profile = build_profile(sample_repo)

        ranked = rank_candidates(profile, "billing auth session login", root=sample_repo, limit=2)

        assert len(ranked) == 2

    def test_every_candidate_exists_in_the_profile(self, sample_repo: Path) -> None:
        profile = build_profile(sample_repo)
        known = {f.path for f in profile.files}

        for candidate in rank_candidates(profile, "billing invoice refund", root=sample_repo):
            assert candidate.path in known
