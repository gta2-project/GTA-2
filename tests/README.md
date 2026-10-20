# Offline automated tests

See [offline testing](../docs/development/testing.md) for environment setup,
browser checks, and documentation examples.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --collect-only -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

`pytest.ini` restricts default discovery to an explicit allowlist of reviewed
offline modules. Tests use fake devices/providers, temporary workspaces, and
mocked browser requests. Audit imports as well as test bodies before adding
a module's exact filename to the allowlist.

The supporting simulation module and `fixtures/library_child.py` provide
pure calculations and test child processes. Browser tests may skip if
Playwright/Chromium or the built frontend is absent.

Standalone [UR5e utilities](../scripts/ur5e/README.md) are not tests. Some perform
hardware I/O on import; keep them out of test fixtures. Explicit file arguments
override the discovery allowlist.
