# Build the documentation

The site uses Python 3.11 or later, Sphinx, MyST Markdown, and the Read the Docs theme. English and
Chinese pages are built from `docs/en-US/` and `docs/zh-CN/` respectively.

Run these commands from the repository root. The documentation environment
does not require SatNav, PyTorch, or model dependencies.

```bash
python3 -m venv .local/docs-venv
.local/docs-venv/bin/python -m pip install -r docs/requirements.txt
.local/docs-venv/bin/python scripts/build_docs.py
.local/docs-venv/bin/python -m http.server 8000 --bind 127.0.0.1 --directory docs/_build/html
```

Open <http://localhost:8000/wiki/>. English and Chinese builds live at
`/wiki/en-US/index.html` and `/wiki/zh-CN/index.html`. The language link opens the same
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

On Windows PowerShell, use `.local/docs-venv/Scripts/python.exe` for the virtual
environment's interpreter. Source Markdown is read and written as UTF-8; generated
asset and redirect URLs use forward slashes on every platform.

## Maintain concept figures

The four system-principle pages live in `docs/<language>/concepts/`. Add or update
both languages together, including their home-page toctrees and cross-links to
the operation guides.

Explain concepts, mechanisms, steps, and conclusions directly. Place model
assumptions and units beside the relevant explanation or formula. Keep figure
provenance, regeneration commands, and code-review baselines in maintenance
material. The initial concept pages were checked against code baseline `402027e`.

Eight mechanism diagrams have matching SVG and editable draw.io sources under
`docs/assets/concepts/diagrams/`.

Keep editable `.drawio` files in the repository. Wiki pages show the diagrams and
explanatory captions; source-file download links belong in maintenance material.

Center figure captions with `<p class="figure-caption" align="center"><em>Caption</em></p>`.
The class applies the Wiki style, and the alignment attribute also supports
repository Markdown rendering. Keep body paragraphs left-aligned.

Regenerate the diagrams with the standard-library script:

```bash
python scripts/docs/generate_concept_diagrams.py
```

The real-scene camera comparisons use the repository's `SatelliteCamera`, with
Matplotlib providing layout and annotations. Install `rasterio`,
`opencv-python-headless`, `pyproj`, `matplotlib`, `numpy`, and `Pillow` in a figure
environment, then run:

```bash
python scripts/docs/render_camera_examples.py --scene /path/to/Amsterdam-1.tif
```

The script generates both languages and records the source digest, center, and
parameters in `docs/assets/concepts/README.md`. When choosing a different scene,
update the scene description in both `concepts/SATSIM.md` pages as well.
After regenerating, inspect text and connectors in every diagram, then rebuild
both languages to check page layout, equations, images, and downloadable sources.

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
published base URL. The documentation lives under `wiki/`:
<https://eku127.github.io/SatNav/wiki/>. The old root and language-page URLs
redirect to the new location.

The output is a complete static site, including both languages, search indexes,
styles, images, and videos. It can also be served by another static host at a
domain root or project subpath; no application server is needed.
