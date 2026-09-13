"""Shared configuration for the English and Chinese documentation builds."""

from pathlib import Path

project = "SatNav"
author = "SatNav contributors"
extensions = ["myst_parser", "sphinx_copybutton"]
source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["assets/**"]
myst_enable_extensions = ["colon_fence", "dollarmath", "html_image"]
myst_heading_anchors = 6
html_theme = "sphinx_rtd_theme"
html_title = "SatNav Documentation"
html_static_path = [str(Path(__file__).parent / "_static")]
templates_path = [str(Path(__file__).parent / "_templates")]
html_css_files = ["satnav.css"]
html_theme_options = {
    "style_nav_header_background": "#2475a5",
    "collapse_navigation": False,
    "navigation_depth": 2,
    "titles_only": True,
    "sticky_navigation": True,
    "prev_next_buttons_location": "bottom",
}
html_show_sourcelink = False
html_show_copyright = False
html_show_sphinx = True
html_copy_source = False
copybutton_prompt_text = r"\$ |>>> |\.\.\. "
copybutton_prompt_is_regexp = True


def page_context(app, pagename, templatename, context, doctree):
    """Keep language switching on the corresponding page, at any URL depth."""
    current = "zh-CN" if app.config.language == "zh_CN" else "en-US"
    other = "en-US" if current == "zh-CN" else "zh-CN"
    context["satnav_other_language"] = "English" if other == "en-US" else "简体中文"
    context["satnav_language_url"] = "../" * (pagename.count("/") + 1) + other + "/" + pagename + ".html"


def setup(app):
    app.connect("html-page-context", page_context)
