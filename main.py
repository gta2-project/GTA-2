"""Web dashboard entry point for GTA Sessions."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
import uvicorn


ROOT_DIR = Path(__file__).resolve().parent


def main() -> None:
    """Start the local FastAPI web dashboard."""
    os.chdir(ROOT_DIR)
    load_dotenv(ROOT_DIR / ".env")
    host = os.getenv("GTA_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("GTA_WEB_PORT", "8000"))
    reload = os.getenv("GTA_WEB_RELOAD", "false").lower() in {"1", "true", "yes"}

    print(f"GTA Sessions web dashboard: http://{host}:{port}")
    uvicorn.run("api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
