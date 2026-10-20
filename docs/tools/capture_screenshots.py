"""Capture the real frontend with synthetic fixtures; all requests are intercepted.

Never launch main.py/library.py here. No server, provider, ROS, or RTDE is used.
Run after building frontend/dist; output is limited to docs/_static/screenshots.
"""

import json
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "frontend" / "dist"
ASSETS = ROOT / "docs" / "_static"
STAMP = "2026-01-01T12:00:00Z"
ORIGIN = "http://gta-docs.invalid"


def fixtures():
    plan = json.loads((ROOT / "docs/examples/cube-plan.json").read_text())
    points = json.loads((ROOT / "docs/examples/cube-keypoints.json").read_text())
    checkpoint = {
        "id": "a" * 32, "name": "Cube pickup",
        "notes": "Illustrative documentation fixture. Place a visible cube on a clear surface.",
        "task": "Pick up the yellow cube and lift it above the table.",
        "visible": True, "thumbnail_url": "/fixture/cube-scene.jpg", "created_at": STAMP,
    }
    session = {"id": "b" * 32, "checkpoint_id": checkpoint["id"], "checkpoint": checkpoint}
    state = {
        "task": checkpoint["task"], "subtasks": [
            {"id": item["subtask_id"], "subtask_prompt": item["task"],
             "goal_check_prompt": item["goal"]} for item in plan],
        "skills": plan, "parameters": plan, "keypoints": points,
        "messages": [], "events": [], "busy_stage": None, "last_error": None,
        "artifacts": {"environment.jpg": {
            "name": "environment.jpg", "exists": True, "url": "/fixture/cube-scene.jpg",
            "size": None, "mtime": STAMP}},
        "script": "# Illustrative documentation fixture — not an executable skill.\n"
                  "# Actual scripts are generated from current grounding.\n\n"
                  "# 1. Prepare RGB and aligned 3D points.\n"
                  "# 2. Review the required Open3D preview.\n"
                  "# 3. Execute reviewed controller subtasks.\n"
                  "# 4. Stop and verify hardware cleanup.\n",
        "script_path": "robot_script.py", "revision": 1, "image_revision": 1,
        "grounding_ready": True, "script_ready": True, "active_handlers": 0,
    }
    health = {
        "ready": True, "initialized": True, "init_error": None,
        "camera_backend": "file", "gta_backend": "ros2",
        "camera_image_path": "environment.jpg",
        "agent_models": {name: {"provider": "fixture", "model": "illustrative"}
                         for name in ("TaskDecomposer", "SkillGenerator",
                                      "ParameterSetter", "VisionModule")},
        "robot_dependencies": {"rclpy": False, "rospy": False, "rtde_receive": False},
    }
    execution = {
        "status": "idle", "operation": None, "run_id": None, "session_id": None,
        "subtask": None, "subtasks": [], "logs": [], "error": None, "skip_pending": False,
    }
    return checkpoint, session, state, health, execution


def main():
    if not (DIST / "library.html").is_file():
        raise SystemExit("Build frontend/dist first; this tool never starts the application.")
    output = ASSETS / "screenshots"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint, session, state, health, execution = fixtures()
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        context = browser.new_context(viewport={"width": 1440, "height": 1020},
                                      device_scale_factor=1, locale="en-US",
                                      timezone_id="UTC", reduced_motion="reduce",
                                      service_workers="block")
        # The frontend sees a connected event stream, but no socket is constructed.
        context.add_init_script("""
            window.WebSocket = class {
                constructor() { setTimeout(() => this.onopen?.({}), 0); }
                close() {}
            };
        """)
        failures, requests = [], []

        def route(request):
            url = urlparse(request.request.url)
            path = url.path
            requests.append((request.request.method, path))
            if url.netloc != "gta-docs.invalid" or request.request.method != "GET":
                failures.append("Unexpected request: " + request.request.method + " " + path)
                request.abort()
                return
            if path == "/fixture/cube-scene.jpg":
                request.fulfill(body=(ASSETS / "images/cube-scene.jpg").read_bytes(),
                                content_type="image/jpeg")
                return
            if path.startswith("/api/"):
                if path == "/api/checkpoints":
                    data = [checkpoint]
                elif path == "/api/library/session":
                    data = session
                elif path == "/api/execution":
                    data = execution
                elif path.endswith("/health"):
                    data = health
                elif path.endswith("/state"):
                    data = state
                else:
                    failures.append("Unmocked API path: " + path)
                    request.abort()
                    return
                request.fulfill(body=json.dumps(data), content_type="application/json")
                return
            if path in {"/", "/admin", "/library", "/run"}:
                file = DIST / ("index.html" if path == "/" else "library.html")
            elif path.startswith("/assets/"):
                file = (DIST / path.lstrip("/")).resolve()
            else:
                failures.append("Unmocked asset: " + path)
                request.abort()
                return
            if not file.is_relative_to(DIST) or not file.is_file():
                failures.append("Missing or out-of-root asset: " + path)
                request.abort()
                return
            request.fulfill(body=file.read_bytes(),
                            content_type=mimetypes.guess_type(file.name)[0] or "application/octet-stream")

        context.route("**/*", route)
        page = context.new_page()
        page.on("pageerror", lambda error: failures.append(str(error)))

        def capture(name, target=None):
            # Focusing a carousel textarea can scroll an overflow-hidden ancestor.
            # Restore normal horizontal scroll without altering the frontend/CSS.
            page.evaluate("""() => {
                for (const element of document.querySelectorAll('*')) {
                    if (element.scrollLeft) element.scrollLeft = 0;
                }
                document.activeElement?.blur();
            }""")
            page.evaluate("document.fonts.ready")
            page.locator("img").evaluate_all(
                "(images) => Promise.all(images.map(i => i.decode().catch(() => {})))")
            page.wait_for_timeout(350)  # Allow the actual panel transition to settle.
            (target or page).screenshot(path=str(output / name),
                                       **({} if target else {"full_page": True}))

        page.goto(ORIGIN + "/")
        expect(page.get_by_role("heading", name="GTA Sessions", exact=True)).to_be_visible()
        page.locator(".stage-tab").filter(has_text="Task Decomposer").click()
        page.locator("#task").fill(checkpoint["task"])  # No form submission.
        capture("dashboard.png")
        page.locator(".stage-tab").filter(has_text="Vision Module").click()
        visible_feedback = page.locator(".stage-browser-slide").nth(3).locator("textarea")
        visible_feedback.fill("Place the cube point at the center of its visible top face.")
        capture("vision.png")

        page.set_viewport_size({"width": 1440, "height": 620})
        page.goto(ORIGIN + "/library")
        expect(page.get_by_role("button", name="Load in Run", exact=True)).to_be_visible()
        capture("library.png")

        page.set_viewport_size({"width": 1440, "height": 1020})
        execution.update(status="preview", operation="run", run_id="illustrative-run",
                         session_id=session["id"],
                         logs=[{"role": "script", "text": "Illustrative preview state; no hardware run."}])
        page.goto(ORIGIN + "/run")
        page.locator(".stage-tab").filter(has_text="Robot Script").click()
        panel = page.locator(".domain-panel").filter(
            has=page.get_by_role("heading", name="Robot Script", exact=True))
        expect(panel.get_by_role("button", name="Cancel", exact=True)).to_be_enabled()
        capture("execution.png", panel)
        browser.close()
        if failures:
            raise AssertionError("\n".join(failures))
        assert all(method == "GET" for method, _ in requests)
        print(f"Captured four screenshots; intercepted {len(requests)} GETs, no live service.")


if __name__ == "__main__":
    main()
