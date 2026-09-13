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

## Publish on GitHub Pages

The `Deploy documentation` GitHub Actions workflow builds and publishes
`docs/_build/html/` when documentation, its build script, or the workflow
changes on `master`. Generated HTML is uploaded as an artifact; no `gh-pages`
branch is needed.

1. Edit the Markdown and preview it locally using the commands above.
2. Commit and push your changes to `master`, or merge your documentation branch
   into `master` and push.
3. Check **Actions → Deploy documentation** for the build and deployment result.

To publish again without changing files, open **Actions → Deploy documentation
→ Run workflow**, select `master`, and run it.

Repository setup: **Settings → Pages → Build and deployment → Source** must be
set to **GitHub Actions**. The Pages settings and deployment result show the
published URL. For `Eku127/SatNav` without a custom domain, the default is
<https://eku127.github.io/SatNav/>.

The output is a complete static site, including both languages, search indexes,
styles, images, and videos. It can also be served by another static host at a
domain root or project subpath; no application server is needed.
