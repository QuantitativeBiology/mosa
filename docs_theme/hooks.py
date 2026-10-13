"""Keep source links usable in both GitHub Markdown and the published site."""

import re
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

from mkdocs.utils import get_relative_url


def on_page_markdown(markdown, *, page, config, **kwargs):
    docs_root = Path(config.docs_dir).resolve()
    repo_root = docs_root.parent
    source_dir = Path(page.file.abs_src_path).parent

    def repository_link(match):
        href = match.group(2)
        parts = urlsplit(href)
        if parts.scheme or parts.netloc or not parts.path or parts.path.startswith("/"):
            return match.group(0)
        target = (source_dir / parts.path).resolve()
        if target.is_relative_to(docs_root):
            return match.group(0)
        if not target.is_relative_to(repo_root) or not target.exists():
            raise ValueError(f"{page.file.src_uri}: invalid repository link {href!r}")
        kind = "tree" if target.is_dir() else "blob"
        repo_path = quote(target.relative_to(repo_root).as_posix(), safe="/")
        url = f"{config.repo_url}/{kind}/main/{repo_path}"
        return (
            match.group(1)
            + urlunsplit((*urlsplit(url)[:3], parts.query, parts.fragment))
            + match.group(3)
        )

    def image_link(match):
        source = match.group(2)
        parts = urlsplit(source)
        if parts.scheme or parts.netloc or source.startswith("/"):
            return match.group(0)
        target = (source_dir / parts.path).resolve()
        if not target.is_relative_to(docs_root) or not target.is_file():
            raise ValueError(f"{page.file.src_uri}: invalid image path {source!r}")
        relative = get_relative_url(target.relative_to(docs_root).as_posix(), page.file.url)
        return match.group(1) + relative + match.group(3)

    markdown = re.sub(r'(<img\b[^>]*\bsrc=")([^"]+)(")', image_link, markdown)
    return re.sub(r"(\]\()([^\s)]+)(\))", repository_link, markdown)


def on_page_content(html, *, page, **kwargs):
    # The theme renders the title and navigation; Markdown retains them on GitHub.
    html = re.sub(r"^\s*<h1\b[^>]*>.*?</h1>\s*", "", html, count=1, flags=re.S)
    html = re.sub(
        r'^<p><a href="[^"]*">Documentation</a>.*?</p>\s*',
        "",
        html,
        count=1,
        flags=re.S,
    )

    def link_diagram(match):
        tag = match.group(0)
        source = re.search(r'src="([^"]+)"', tag)
        if source and "/assets/diagrams/" in source.group(1):
            return f'<a href="{source.group(1)}" title="Open diagram at full size">{tag}</a>'
        return tag

    return re.sub(r"<img\b[^>]*>", link_diagram, html)
