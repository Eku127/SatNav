# Combined website and Wiki deployment

The `Deploy website and Wiki` workflow publishes a single Pages artifact:

- `/SatNav/`: project website from private `Eku127/SatNav-website`, branch `main`.
- `/SatNav/wiki/`: multilingual documentation built from this repository.
- Existing `/SatNav/en-US/` and `/SatNav/zh-CN/` redirects are preserved.

Website images and videos remain in the separate website repository.
Actions reads that repository using the `WEBSITE_DEPLOY_KEY` secret, paired
with its read-only deploy key. No personal access token is needed in the workflow.

After pushing website changes, run `docs.yml` manually on `master`.
Documentation changes matching its push paths publish both automatically.
GitHub Pages must use **GitHub Actions** as its publishing source.

Build steps:

```sh
python scripts/build_docs.py
python _website/build_site.py
python scripts/assemble_pages.py --website _website/_site
```

`_website` is the separate checkout created by Actions. Only its generated
`_site` directory is copied into `docs/_build/html`; the Wiki paths are reserved.
