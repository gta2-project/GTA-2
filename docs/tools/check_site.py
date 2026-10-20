"""Check every local HTML link, anchor, image, script, and download without network."""

from html.parser import HTMLParser
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.ids = set()
        self.links = []
        self.feed(source)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        for name in ("id", "name" if tag == "a" else "id"):
            if attrs.get(name):
                self.ids.add(attrs[name])
        for name in ("href", "src", "poster"):
            if attrs.get(name):
                self.links.append(attrs[name])


def check(root):
    root = root.resolve()
    pages = {path: Page(path.read_text(encoding="utf-8"))
             for path in root.rglob("*.html")}
    errors = []
    if any(root.rglob("*.doctree")) or any(root.rglob("*.pickle")):
        errors.append("Sphinx caches are inside the website; use the documented -M build.")
    count = 0
    for source, page in pages.items():
        for raw in page.links:
            link = urlsplit(raw)
            if link.scheme or link.netloc:
                continue
            target = (root / unquote(link.path).lstrip("/") if link.path.startswith("/")
                      else source.parent / unquote(link.path)).resolve()
            if not link.path:
                target = source
            if not target.is_relative_to(root):
                errors.append(f"{source.relative_to(root)}: link leaves site: {raw}")
                continue
            if target.is_dir():
                target /= "index.html"
            count += 1
            if not target.is_file():
                errors.append(f"{source.relative_to(root)}: missing target: {raw}")
            elif link.fragment and target in pages:
                if unquote(link.fragment) not in pages[target].ids:
                    errors.append(f"{source.relative_to(root)}: missing anchor: {raw}")
    return pages, count, errors


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/_build/html")
    if not (root / "index.html").is_file():
        raise SystemExit("Build the HTML site first.")
    pages, count, errors = check(root)
    for error in errors:
        print(error)
    print(f"Checked {len(pages)} HTML pages and {count} local references: "
          f"{len(errors)} errors")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
