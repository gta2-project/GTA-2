"""Separate Prepare / Library / Run application; never mounted by main.py."""
from pathlib import Path
from typing import Optional

from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.app import ROOT_DIR, create_app
from api.checkpoints import LibraryManager
from api.library_routes import install_library_routes
from api.service import GtaWebService


def create_library_app(service: Optional[GtaWebService] = None,
                       initialize_on_startup: bool = True, root: Path = ROOT_DIR):
    root = Path(root)
    # Keep library preparation artifacts separate from the normal dashboard too.
    admin = service or GtaWebService(root / 'library_sessions' / 'admin', scoped=True)
    manager = LibraryManager(root, admin)

    class AdminProxy:
        @property
        def event_service(self):
            return manager.admin

        def __getattr__(self, name):
            def invoke(*args, **kwargs):
                with manager.lock:
                    return getattr(manager.admin, name)(*args, **kwargs)
            return invoke

    app = create_app(AdminProxy(), initialize_on_startup=False, serve_frontend=False,
                     enable_execution=False)
    app.title = 'GTA-2 Library'
    app.state.gta_service = admin
    app.state.library_manager = manager
    install_library_routes(app, manager)

    if initialize_on_startup:
        @app.on_event('startup')
        def startup():
            manager.admin.root_dir.mkdir(parents=True, exist_ok=True)
            manager.admin.initialize()

    @app.on_event('shutdown')
    def shutdown():
        manager.shutdown()

    @app.get('/')
    def home():
        return RedirectResponse('/admin', status_code=303)

    dist = ROOT_DIR / 'frontend' / 'dist'

    @app.get('/admin')
    @app.get('/library')
    @app.get('/run')
    def interface():
        if (dist / 'library.html').is_file():
            return FileResponse(dist / 'library.html')
        return {'message': 'Build the frontend with npm run build, then open /admin, /library or /run.'}

    if (dist / 'assets').is_dir():
        app.mount('/assets', StaticFiles(directory=str(dist / 'assets')), name='library-assets')
    return app


app = create_library_app()
