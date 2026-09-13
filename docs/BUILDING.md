# Build the documentation

The site uses Sphinx, MyST Markdown, and the Read the Docs theme. English and
Chinese pages are built from `docs/en-US/` and `docs/zh-CN/` respectively.

Run these commands from the repository root. The documentation environment
does not require SatNav, PyTorch, or model dependencies.

```bash
python3 -m venv .local/docs-venv
.local/docs-venv/bin/python -m pip install -r docs/requirements.txt
.local/docs-venv/bin/python scripts/build_docs.py
.local/docs-venv/bin/python -m http.server 8000 --bind 127.0.0.1 --directory docs/_build/html
```

Open <http://localhost:8000/>. English and Chinese builds live at
`/en-US/index.html` and `/zh-CN/index.html`. The language link opens the same
page in the other language.

## Edit pages

- Edit the existing Markdown under the corresponding language directory.
- Add new pages to the `toctree` in that language's `index.md`.
- Keep shared images and videos in `docs/assets/`.
- Re-run the build command and refresh the browser after edits.

The build script prepares temporary copies under `docs/_build/sources/`,
adapts repository links for the website, and builds both languages with
warnings treated as errors. Authored Markdown keeps its GitHub-relative links.
Generated files and the local Python environment are ignored by Git.

## Publish later

`docs/_build/html/` is the complete static site, including search indexes,
styles, images, and videos. It can be served at a domain root or a project
subpath. Publish that directory when ready; no application server is needed.
