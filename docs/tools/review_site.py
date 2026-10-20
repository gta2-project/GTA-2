"""Browser QA for the built static documentation. Only a loopback site is allowed."""

import argparse
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def local_preview():
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    handler = partial(QuietHandler, directory=str(ROOT / "docs/_build/html"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Existing loopback preview; otherwise serve the built site temporarily.")
    args = parser.parse_args()
    if args.url:
        review(args.url.rstrip("/"))
    else:
        with local_preview() as origin:
            review(origin)


def review(origin):
    if urlparse(origin).hostname not in {"127.0.0.1", "localhost"}:
        raise SystemExit("Only a loopback documentation preview may be inspected.")
    built = ROOT / "docs/_build/html"
    output = ROOT / "docs/_build/qa"
    output.mkdir(parents=True, exist_ok=True)
    pages = sorted(path.relative_to(built).as_posix() for path in built.rglob("*.html"))
    problems = []
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        for width, height in ((1440, 1000), (390, 844)):
            for theme in ("light", "dark"):
                context = browser.new_context(
                    viewport={"width": width, "height": height}, color_scheme=theme,
                    reduced_motion="reduce", service_workers="block")

                def local_only(route):
                    if not route.request.url.startswith(origin + "/"):
                        problems.append("Unexpected external request: " + route.request.url)
                        route.abort()
                    else:
                        route.continue_()

                context.route("**/*", local_only)
                page = context.new_page()
                page.on("pageerror", lambda error: problems.append(str(error)))
                page.on("response", lambda response: problems.append(
                    f"HTTP {response.status}: {response.url}") if response.status >= 400 else None)
                for name in pages:
                    page.goto(origin + "/" + name, wait_until="networkidle")
                    expect(page.locator('[role="main"]')).to_be_attached()
                    if name == "index.html":
                        expect(page.locator('[role="main"] h1')).to_be_hidden()
                        expect(page.locator('[role="main"] .hero')).to_be_visible()
                    elif name not in {"genindex.html", "search.html"}:
                        expect(page.locator('[role="main"] h1').first).to_be_visible()
                    measurements = page.evaluate("""() => ({
                        overflow: document.documentElement.scrollWidth > innerWidth + 2,
                        broken: [...document.images].filter(i => !i.complete || i.naturalWidth === 0)
                            .map(i => i.getAttribute('src'))
                    })""")
                    if measurements["overflow"] or measurements["broken"]:
                        problems.append(f"{width}/{theme}/{name}: {measurements}")
                    if name in {"index.html", "extend/controller.html",
                                "hardware/tool-geometry.html", "using/vision.html"}:
                        page.screenshot(path=str(output / f"{width}-{theme}-{name.replace('/', '-')}.png"),
                                        full_page=name != "extend/controller.html")
                page.goto(origin + "/search.html?q=TCP", wait_until="networkidle")
                expect(page.locator("#search-results li").first).to_be_visible(timeout=10000)
                # Exercise the actual theme button, not just media emulation.
                before = page.locator("body").evaluate("(e) => getComputedStyle(e).backgroundColor")
                toggle = page.locator(".theme-toggle:visible").first
                for _ in range(3):
                    toggle.click()
                    page.wait_for_timeout(50)
                    after = page.locator("body").evaluate("(e) => getComputedStyle(e).backgroundColor")
                    if before != after:
                        break
                else:
                    problems.append(f"{width}/{theme}: theme toggle did not change colors")
                # Exercise Furo's real mobile navigation drawer.
                if width == 390:
                    page.goto(origin + "/index.html")
                    page.locator("label.nav-overlay-icon[for=__navigation]").click()
                    expect(page.locator(".sidebar-tree")).to_be_visible()
                    page.locator(".sidebar-tree a").filter(has_text="Install GTA").click()
                    expect(page).to_have_url(origin + "/start/installation.html")
                context.close()
        browser.close()
    if problems:
        raise AssertionError("\n".join(problems))
    print(f"Reviewed {len(pages)} pages at two widths in both themes; search and mobile navigation passed.")


if __name__ == "__main__":
    main()
