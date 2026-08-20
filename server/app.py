"""App factory and wiring. No guardrail logic lives here."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Cookie, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .auth import directory
from .routes import api
from .state import state

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
DEMO = ROOT / "demo"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state.try_reload()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Guardrail Console", version="1.0.0", lifespan=lifespan)
    app.include_router(api)

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        """The landing page: what the stack is, drawn as one pipeline."""
        return FileResponse(WEB / "home.html")

    @app.get("/login", include_in_schema=False)
    def login_page() -> FileResponse:
        return FileResponse(WEB / "login.html")

    def _gate(cookie: str | None, target: str, permission: str = ""):
        """Send an unauthenticated caller to sign in, remembering where they were.

        The page gate is a courtesy — it stops someone landing on a screen that
        would only 401 at them. The enforcement is on the API routes.
        """
        user = directory.resolve(cookie)
        if user is None:
            return RedirectResponse(f"/login?next={target}", status_code=303)
        if permission and not user.can(permission):
            return RedirectResponse("/console", status_code=303)
        return None

    @app.get("/console", include_in_schema=False)
    def console(gc_session: str | None = Cookie(default=None)):
        return _gate(gc_session, "/console") or FileResponse(WEB / "index.html")

    # The request-lifecycle chart. Declared before the static mount below, so
    # it wins the path; the page itself calls /api/chat to overlay a real trace.
    @app.get("/demo", include_in_schema=False)
    def demo(gc_session: str | None = Cookie(default=None)):
        # The architecture view runs scenarios and ingests documents, so it is
        # an operator's tool rather than a citizen's.
        return _gate(gc_session, "/demo", "scenarios") or FileResponse(DEMO / "index.html")

    @app.get("/demo/stages", include_in_schema=False)
    def demo_stages(gc_session: str | None = Cookie(default=None)):
        """The same pipeline stage by stage, in depth."""
        return _gate(gc_session, "/demo/stages", "traces") or FileResponse(DEMO / "stages.html")

    @app.exception_handler(HTTPException)
    async def http_error(_req, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content={"error": detail})

    if WEB.exists():
        app.mount("/", StaticFiles(directory=WEB, html=True), name="web")

    return app


app = create_app()
