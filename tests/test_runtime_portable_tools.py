"""Tests for bounded tools shared by orchestration framework adapters."""

from pathlib import Path

import pytest
from lychee_mas.runtime.model.token_budget import bound_text_for_model_context
from lychee_mas.runtime.tools.portable import portable_web_tools, portable_workspace_tools


class _Response:
    def __init__(self, text: str, *, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_workspace_listing_is_bounded_and_skips_git(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "secret").write_text("hidden", encoding="utf-8")
    source = tmp_path / "repo" / "package"
    source.mkdir(parents=True)
    (source / "module.py").write_text("print('ok')", encoding="utf-8")
    tools = portable_workspace_tools(tmp_path)

    listing = tools["list_workspace"](max_depth=2, max_entries=10)

    assert ".git" not in listing
    assert "repo" in listing
    assert "repo/package" in listing
    assert "module.py" not in listing


def test_workspace_reader_supports_segments_and_rejects_escape(tmp_path: Path) -> None:
    (tmp_path / "long.txt").write_text("a" * 3000, encoding="utf-8")
    tools = portable_workspace_tools(tmp_path)

    value = tools["read_text_file"]("long.txt", offset=500, max_chars=1000)

    assert value.startswith("a" * 1000)
    assert "continue at offset=1500" in value
    with pytest.raises(ValueError, match="inside the workspace"):
        tools["read_text_file"]("../outside.txt")


def test_workspace_reader_accepts_one_based_start_line(tmp_path: Path) -> None:
    (tmp_path / "lines.txt").write_text(
        "first\nsecond\nthird\n", encoding="utf-8"
    )
    tools = portable_workspace_tools(tmp_path)

    value = tools["read_text_file"]("lines.txt", start_line=2, max_chars=1000)

    assert value.startswith("second\nthird\n")


def test_workspace_reader_accepts_container_workspace_prefix(tmp_path: Path) -> None:
    (tmp_path / "case.txt").write_text("portable", encoding="utf-8")
    tools = portable_workspace_tools(tmp_path)

    assert tools["read_text_file"]("/workspace/case.txt") == "portable"


def test_workspace_listing_treats_slash_as_virtual_workspace_root(tmp_path: Path) -> None:
    (tmp_path / "case.txt").write_text("portable", encoding="utf-8")
    tools = portable_workspace_tools(tmp_path)

    assert tools["list_workspace"]("/") == "case.txt"


def test_workspace_exact_replace_is_atomic_and_rejects_ambiguous_edits(
    tmp_path: Path,
) -> None:
    source = tmp_path / "repo" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\nvalue = 2\n", encoding="utf-8")
    tools = portable_workspace_tools(tmp_path)

    result = tools["replace_text_file"](
        "repo/module.py",
        "value = 2",
        "value = 3",
    )

    assert result == "replaced 1 occurrence(s) in repo/module.py"
    assert source.read_text(encoding="utf-8") == "value = 1\nvalue = 3\n"
    with pytest.raises(ValueError, match="found 2"):
        tools["replace_text_file"](
            "repo/module.py",
            "value",
            "item",
        )
    with pytest.raises(ValueError, match="inside the workspace"):
        tools["replace_text_file"]("../outside.py", "old", "new")
    with pytest.raises(ValueError, match="identical"):
        tools["replace_text_file"](
            "repo/module.py",
            "value = 3",
            "value = 3",
        )


def test_model_context_text_bound_is_exact_and_preserves_both_ends() -> None:
    text = "HEAD" + "x" * 2500 + "TAIL"

    bounded, truncated = bound_text_for_model_context(text, 1000)

    assert truncated is True
    assert len(bounded) == 1000
    assert bounded.startswith("HEAD")
    assert bounded.endswith("TAIL")


def test_portable_search_uses_bing_result_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    html = """
    <li class="b_algo"><h2><a href="https://example.test/a">Example A</a></h2>
    <div class="b_caption"><p>First snippet.</p></div></li>
    """

    def fake_get(url: str, **_: object) -> _Response:
        assert url == "https://www.bing.com/search"
        return _Response(html)

    monkeypatch.setattr("requests.get", fake_get)

    result = portable_web_tools()["search_web"]("query", max_results=1)

    assert "[1] Example A" in result
    assert "URL: https://example.test/a" in result
    assert "First snippet." in result


def test_portable_search_falls_back_after_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    html = """
    <div class="result"><a class="result__a" href="https://example.test/b">B</a>
    <div class="result__snippet">Fallback snippet.</div></div>
    """
    called: list[str] = []

    def fake_get(url: str, **_: object) -> _Response:
        called.append(url)
        if "bing.com" in url:
            raise OSError("temporary provider failure")
        return _Response(html)

    monkeypatch.setattr("requests.get", fake_get)

    result = portable_web_tools()["search_web"]("query", max_results=1)

    assert called[:2] == [
        "https://www.bing.com/search",
        "https://html.duckduckgo.com/html/",
    ]
    assert "URL: https://example.test/b" in result


def test_portable_web_tools_expose_both_search_name_conventions() -> None:
    tools = portable_web_tools()

    assert tools["web_search"] is tools["search_web"]


def test_portable_fetch_rejects_workspace_file_urls() -> None:
    tools = portable_web_tools()

    with pytest.raises(ValueError, match="use read_text_file"):
        tools["fetch_url"]("file:///workspace/case.txt")
