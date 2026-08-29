"""Framework-neutral tools shared by every RuntimeAdapter."""

from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, unquote, urlparse


def portable_workspace_tools(workspace: str | Path | None) -> dict[str, Callable]:
    """Return bounded, case-local file tools with identical framework semantics."""

    if workspace is None:
        return {}
    root = Path(workspace).resolve()

    def resolve(path: str) -> Path:
        portable_path = str(path).strip()
        if portable_path in {"/", "/workspace"}:
            portable_path = "."
        elif portable_path.startswith("/workspace/"):
            portable_path = portable_path.removeprefix("/workspace/")
        target = (root / portable_path).resolve()
        if root not in target.parents and target != root:
            raise ValueError("path must stay inside the workspace")
        return target

    def list_workspace(
        path: str = ".",
        max_depth: int = 2,
        max_entries: int = 500,
    ) -> str:
        """List a bounded case-workspace tree; '.', '/', and '/workspace' mean its root."""

        target = resolve(path)
        if not target.is_dir():
            raise NotADirectoryError(path)
        depth_limit = max(0, min(int(max_depth), 8))
        entry_limit = max(1, min(int(max_entries), 2000))
        rows: list[str] = []
        for current, directories, files in os.walk(target):
            current_path = Path(current)
            relative_dir = current_path.relative_to(root)
            depth = len(current_path.relative_to(target).parts)
            directories[:] = sorted(
                name for name in directories if name not in {".git", "__pycache__"}
            )
            if depth >= depth_limit:
                directories[:] = []
            visible_directories = [
                f"{item}/" for item in directories if depth + 1 <= depth_limit
            ]
            visible_files = sorted(files) if depth + 1 <= depth_limit else []
            for name in [*visible_directories, *visible_files]:
                rows.append(str(relative_dir / name))
                if len(rows) >= entry_limit:
                    rows.append(
                        f"... truncated after {entry_limit} entries; narrow path or depth"
                    )
                    return "\n".join(rows)
        return "\n".join(rows)

    def read_text_file(
        path: str,
        offset: int = 0,
        max_chars: int = 50000,
        start_line: int = 0,
    ) -> str:
        """Read a local workspace file; use fetch_url only for HTTP(S) resources."""

        target = resolve(path)
        text = target.read_text(encoding="utf-8", errors="replace")
        line_number = max(0, int(start_line))
        if line_number:
            lines = text.splitlines(keepends=True)
            start = sum(len(item) for item in lines[: max(0, line_number - 1)])
        else:
            start = max(0, int(offset))
        limit = max(1000, min(int(max_chars), 200000))
        segment = text[start : start + limit]
        suffix = ""
        if start + len(segment) < len(text):
            suffix = f"\n... truncated; continue at offset={start + len(segment)}"
        return segment + suffix

    def replace_text_file(
        path: str,
        old_text: str,
        new_text: str,
        expected_replacements: int = 1,
    ) -> str:
        """Atomically replace an exact text fragment in a workspace file."""

        target = resolve(path)
        if not target.is_file():
            raise FileNotFoundError(path)
        if not old_text:
            raise ValueError("old_text must not be empty")
        if old_text == new_text:
            raise ValueError(
                "old_text and new_text are identical; no file change would be made. "
                "Do not retry the same edit: inspect the current file and provide a real change."
            )
        if len(old_text) > 200000 or len(new_text) > 200000:
            raise ValueError("old_text and new_text must each be at most 200000 characters")
        expected = int(expected_replacements)
        if expected < 1:
            raise ValueError("expected_replacements must be >= 1")
        text = target.read_text(encoding="utf-8", errors="strict")
        actual = text.count(old_text)
        if actual != expected:
            raise ValueError(
                f"expected {expected} occurrence(s) of old_text in {path!r}, found {actual}"
            )
        updated = text.replace(old_text, new_text)
        temporary = target.with_name(f".{target.name}.lychee-edit-{os.getpid()}")
        try:
            temporary.write_text(updated, encoding="utf-8")
            temporary.chmod(target.stat().st_mode)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return f"replaced {actual} occurrence(s) in {path}"

    return {
        "list_workspace": list_workspace,
        "read_text_file": read_text_file,
        "replace_text_file": replace_text_file,
    }


def portable_web_tools(proxy_url: str | None = None) -> dict[str, Callable]:
    """Return small search/fetch tools with identical semantics across frameworks."""

    import requests
    from bs4 import BeautifulSoup

    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "Chrome/124.0 Safari/537.36"
        )
    }

    def search_web(query: str, max_results: int = 8) -> str:
        """Search the public web and return result titles, URLs, and snippets."""

        limit = max(1, min(int(max_results), 20))
        providers = (
            (
                "bing",
                "https://www.bing.com/search",
                {"q": query, "setlang": "en-US", "cc": "US"},
            ),
            ("duckduckgo_html", "https://html.duckduckgo.com/html/", {"q": query}),
            ("duckduckgo_lite", "https://lite.duckduckgo.com/lite/", {"q": query}),
        )
        errors: list[str] = []
        for provider, url, params in providers:
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers={**headers, "Accept-Language": "en-US,en;q=0.9"},
                    proxies=proxies,
                    timeout=20,
                )
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                rows: list[str] = []
                if provider == "bing":
                    results = soup.select("li.b_algo")
                    for result in results:
                        link = result.select_one("h2 a")
                        if link is None:
                            continue
                        snippet_node = result.select_one(".b_caption p")
                        snippet = (
                            " ".join(snippet_node.stripped_strings)
                            if snippet_node
                            else ""
                        )
                        title = " ".join(link.stripped_strings)
                        href = str(link.get("href") or "")
                        if title and href:
                            rows.append(
                                f"[{len(rows) + 1}] {title}\nURL: {href}\n{snippet}"
                            )
                        if len(rows) >= limit:
                            break
                else:
                    selectors = (
                        (".result", ".result__a", ".result__snippet"),
                        ("tr", "a.result-link", ".result-snippet"),
                    )
                    for result_selector, link_selector, snippet_selector in selectors:
                        for result in soup.select(result_selector):
                            link = result.select_one(link_selector)
                            if link is None:
                                continue
                            href = str(link.get("href") or "")
                            parsed = parse_qs(urlparse(href).query).get("uddg")
                            result_url = unquote(parsed[0]) if parsed else href
                            snippet_node = result.select_one(snippet_selector)
                            snippet = (
                                " ".join(snippet_node.stripped_strings)
                                if snippet_node
                                else ""
                            )
                            title = " ".join(link.stripped_strings)
                            if title and result_url:
                                rows.append(
                                    f"[{len(rows) + 1}] {title}\n"
                                    f"URL: {result_url}\n{snippet}"
                                )
                            if len(rows) >= limit:
                                break
                        if rows:
                            break
                if rows:
                    return "\n\n".join(rows)
                errors.append(f"{provider}: no parseable results")
            except Exception as exc:
                errors.append(f"{provider}: {type(exc).__name__}: {exc}")
        raise RuntimeError("all search providers failed; " + " | ".join(errors))

    def fetch_url(url: str, max_chars: int = 30000) -> str:
        """Fetch one HTTP(S) URL; use read_text_file for case-workspace files."""

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(
                "fetch_url supports http:// and https:// URLs only; use "
                "read_text_file for files in the case workspace"
            )
        limit = max(1000, min(int(max_chars), 100000))
        response = requests.get(
            url,
            headers=headers,
            proxies=proxies,
            timeout=45,
            allow_redirects=True,
        )
        response.raise_for_status()
        content_type = str(response.headers.get("content-type") or "").lower()
        if "pdf" in content_type or urlparse(response.url).path.lower().endswith(".pdf"):
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(response.content))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            soup = BeautifulSoup(response.text, "html.parser")
            for node in soup(["script", "style", "noscript", "svg"]):
                node.decompose()
            lines = (
                re.sub(r"\s+", " ", item).strip() for item in soup.stripped_strings
            )
            text = "\n".join(line for line in lines if line)
        return f"Final URL: {response.url}\n\n{text[:limit]}"

    return {
        "search_web": search_web,
        "web_search": search_web,
        "fetch_url": fetch_url,
    }
