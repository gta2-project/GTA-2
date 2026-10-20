"""Static documentation configuration: never import application packages."""

project = "GTA-2"
author = "GTA-2 contributors"
copyright = "2026, GTA-2 contributors"
extensions = ["myst_parser", "sphinx_design", "sphinx_copybutton"]
source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "tools", "examples", "requirements*", "**/.DS_Store"]
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3
nitpicky = True

html_theme = "furo"
html_title = "GTA-2 documentation"
html_logo = "_static/logo.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#087f70",
        "color-brand-content": "#087568",
    },
    "dark_css_variables": {
        "color-brand-primary": "#63dfc4",
        "color-brand-content": "#83e8d1",
    },
}
# Do not expose filesystem locations or copy Markdown sources into the website.
html_show_sourcelink = False
html_copy_source = False
html_show_sphinx = False
html_last_updated_fmt = None
html_permalinks_icon = "#"
html_search_language = "en"
copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True
linkcheck_timeout = 20
linkcheck_retries = 2
# Several upstream sites generate anchors in JavaScript. Local anchors are
# checked by Sphinx and tools/check_site.py instead.
linkcheck_anchors = False
