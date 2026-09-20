#!/usr/bin/env python3
"""Build both documentation languages from the repository's Markdown files."""

import os
import json
from html import escape
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
BUILD = DOCS / "_build"
SITE = BUILD / "html"
LANGUAGES = {"en-US": "en", "zh-CN": "zh_CN"}


def write_redirect(path, target):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta http-equiv="refresh" content="0; url={escape(target, quote=True)}">'
        '<title>SatNav Documentation</title>'
        f'<script>location.replace({json.dumps(target)} + location.search + location.hash);</script>'
        f'</head><body><a href="{escape(target, quote=True)}">SatNav Wiki</a></body></html>'
    )


def prepare_markdown(text, source, locale):
    """Adapt repository/asset links in build copies; leave authored Markdown intact."""
    def rewrite(url):
        parsed = urlsplit(url)
        if parsed.scheme or parsed.netloc or not parsed.path:
            return url
        target = (source.parent / unquote(parsed.path)).resolve()
        suffix = ("?" + parsed.query if parsed.query else "") + ("#" + parsed.fragment if parsed.fragment else "")
        if target.is_relative_to(DOCS / "assets"):
            # Keep raw HTML media paths relative to each generated page.
            relative = os.path.relpath(Path("assets") / target.relative_to(DOCS / "assets"), source.relative_to(DOCS / locale).parent)
            return Path(relative).as_posix() + suffix
        if target.is_relative_to(DOCS / locale):
            return url
        if target.is_relative_to(ROOT):
            kind = "tree" if target.is_dir() else "blob"
            return f"https://github.com/Eku127/SatNav/{kind}/master/{quote(target.relative_to(ROOT).as_posix())}" + suffix
        return url

    # Do not rewrite examples inside fenced code blocks.
    parts = re.split(r"(^```[^\n]*\n.*?^```\s*$)", text, flags=re.M | re.S)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"(\]\()([^\s)]+)(\))", lambda m: m[1] + rewrite(m[2]) + m[3], parts[i])
        parts[i] = re.sub(r'((?:src|href)=")([^"]+)(")', lambda m: m[1] + rewrite(m[2]) + m[3], parts[i])
    return "".join(parts)


def main():
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir(parents=True, exist_ok=True)
    for locale, language in LANGUAGES.items():
        source_dir = BUILD / "sources" / locale
        output_dir = SITE / "wiki" / locale
        if source_dir.exists():
            shutil.rmtree(source_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        source_dir.mkdir(parents=True)
        for source in (DOCS / locale).rglob("*.md"):
            destination = source_dir / source.relative_to(DOCS / locale)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                prepare_markdown(source.read_text(encoding="utf-8"), source, locale),
                encoding="utf-8",
            )
        shutil.copytree(DOCS / "assets", source_dir / "assets")
        subprocess.run([
            sys.executable, "-m", "sphinx", "-b", "html", "-W", "--keep-going",
            "-c", str(DOCS), "-D", f"language={language}",
            str(source_dir), str(output_dir),
        ], check=True)
        # Raw HTML video elements need the original asset paths as well.
        shutil.copytree(DOCS / "assets", output_dir / "assets", dirs_exist_ok=True)
        for page in output_dir.rglob("*.html"):
            old_path = SITE / locale / page.relative_to(output_dir)
            write_redirect(old_path, Path(os.path.relpath(page, old_path.parent)).as_posix())
    write_redirect(SITE / "wiki" / "index.html", "en-US/index.html")
    write_redirect(SITE / "index.html", "wiki/")
    print(f"\nDocumentation ready: {SITE}")


if __name__ == "__main__":
    main()
