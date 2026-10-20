# Offline tests

**Prerequisites:** the application Python dependencies and pytest. Optional
browser tests need a built frontend, Playwright, and Chromium.
**Outcome:** verify software behavior without hardware or provider requests.

## 1. Use the audited discovery policy

`pytest.ini` uses `tests/` and an explicit filename allowlist. It excludes
standalone hardware scripts and generated workspaces. A filename beginning
with `test_` is not enough to join the default suite.

From the repository root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --collect-only -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Explicit file arguments override the discovery allowlist. Keep
`scripts/ur5e/` out of test fixtures: some utilities connect hardware on import.

## 2. Test the documentation examples

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_documentation_examples.py
```

These tests use memory-only camera, robot, gripper, and controller objects.
They verify finite inputs, freshness, projection, cancellation, cleanup, and
the synthetic cube fixture's compatibility with current plan validation and
script generation. Generated scripts are checked by parsing their syntax.

The documentation build itself uses a separate environment with no GTA
dependencies. Example behavior tests intentionally use the application
environment because they import the actual abstract contracts and generator.

## 3. Run optional browser checks

```bash
python -m pip install playwright
python -m playwright install chromium
cd frontend
npm ci
npm run build
cd ..
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_library_ui.py
```

The browser tests intercept network requests and use fake API responses.
Missing browser dependencies can skip these checks; inspect the test summary
for skips.

## 4. Add an offline test

Audit imports as well as test bodies. Replace ROS, RTDE, camera, provider, and
subprocess interactions with controlled fakes before those calls can run.
Use temporary directories for outputs. Add the exact reviewed filename to
the sorted `python_files` list in `pytest.ini`.

**Verify:** collect the suite, inspect unexpected imports/skips, and run the
relevant modules. For hardware validation, use the
[hardware checklist](../hardware/checklist.md).
For environment issues see [installation troubleshooting](../troubleshooting/installation.md).
