# Build and maintain the documentation

**Prerequisites:** the release source and Python 3.12 with virtual-environment
support. The documentation environment is separate from the GTA application environment.
**Outcome:** a local searchable HTML site with light/dark themes.

## 1. Create an isolated documentation environment

From the repository root:

```bash
python3.12 -m venv .venv-docs
.venv-docs/bin/python -m pip install -r docs/requirements.txt
```

`docs/requirements.txt` pins the complete documentation dependency set.
`requirements.in` records the direct dependencies for lock updates.
Sphinx reads selected application source as text.

The site uses [Sphinx](https://www.sphinx-doc.org/en/master/usage/markdown.html),
[MyST Markdown](https://myst-parser.readthedocs.io/en/latest/), and
[Furo](https://pradyunsg.me/furo/). Furo supplies navigation, local search,
and light/dark modes; styling is confined to `docs/_static/`.

## 2. Build strictly and preview

```bash
.venv-docs/bin/python -I -m sphinx -M html docs docs/_build -n -W --keep-going
.venv-docs/bin/python docs/tools/check_site.py docs/_build/html
.venv-docs/bin/python -m http.server 8765 --bind 127.0.0.1 --directory docs/_build/html
```

Open `http://127.0.0.1:8765`. Stop the preview with Ctrl+C. Serving over
localhost is preferable to opening HTML directly, especially for search.
Build output and local environments are ignored; sources, assets, and example
code are editable.

The `-I` flag isolates the build from inherited Python paths, including a
sourced ROS environment.
The `-M` build keeps Sphinx's caches outside the HTML folder. Cache files
can contain local source paths and are not website assets.

External-link checking is separate:

```bash
.venv-docs/bin/python -I -m sphinx -b linkcheck docs docs/_build/linkcheck
```

Review failures individually: upstream blocking, redirects, and real missing
pages require different responses. The local checker verifies HTML anchors
and local asset targets.

## 3. Maintain examples and screenshots

Tutorial code lives in `docs/examples/` and is included with
`literalinclude`; edit that source rather than pasting a second copy.
The cube plan and keypoints are synthetic fixtures.

The screenshot tool serves the actual built frontend through intercepted
browser requests. It substitutes synthetic state, disables WebSockets, and
refuses unexpected requests. Use a separate
optional visual-QA environment:

```bash
python3.12 -m venv .venv-docs-qa
.venv-docs-qa/bin/python -m pip install playwright==1.58.0 Pillow==12.1.1
.venv-docs-qa/bin/python -m playwright install chromium
# Build frontend/ first, as in the installation guide.
.venv-docs-qa/bin/python docs/tools/capture_screenshots.py
```

Check all pages at desktop/mobile widths and in both themes. This command
temporarily serves only the built site on loopback and stops that server when
finished:

```bash
.venv-docs-qa/bin/python docs/tools/review_site.py
```

Editable diagrams are SVGs in `docs/_static/diagrams/`. Keep their text,
captions, and alt text accurate. The example image is
`docs/_static/images/cube-scene.jpg`; keep runtime captures in their configured
workspace paths.

## 4. Review before sharing

Check all navigation sections, search, narrow layouts, code scrolling, and
figures in both themes. Run [example tests](testing.md), the strict build,
and the local link checker. Review image metadata and generated HTML using
the [sensitive-data guidance](../reference/artifacts.md#sensitive-data).
Describe current behavior in the documentation.

**Verify:** the HTML site can be built in a clean docs-only environment and
all local links resolve.
