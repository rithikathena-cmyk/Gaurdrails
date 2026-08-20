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

    @app.get("/login", include_in_schema=False)
    def login_page(gc_session: str | None = Cookie(default=None)):
        """Signing in while already signed in is a dead end — "Start app" on the
        home page points here, and somebody who has a session means "take me in",
        not "show me the form again". Sign out first to switch accounts."""
        if directory.resolve(gc_session) is not None:
            return RedirectResponse("/console", status_code=303)
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

    @app.get("/", include_in_schema=False)
    def home():
        """The front door, and public.

        What the stack is, drawn as one pipeline, with the scenarios it is
        checked against. Nothing on it is anybody's data — the cards that once
        led to the control surface are gone — so there is nothing here to sign
        in for.
        """
        return FileResponse(WEB / "home.html")

    @app.get("/console", include_in_schema=False)
    def console(gc_session: str | None = Cookie(default=None)):
        return _gate(gc_session, "/console") or FileResponse(WEB / "index.html")

    # The request-lifecycle chart. Declared before the static mount below, so
    # it wins the path; the page itself calls /api/agent/chat to overlay a real
    # trace.
    @app.get("/demo/stages", include_in_schema=False)
    def demo_stages(gc_session: str | None = Cookie(default=None)):
        """The same pipeline stage by stage, in depth."""
        return _gate(gc_session, "/demo/stages", "traces") or FileResponse(DEMO / "stages.html")

    @app.exception_handler(HTTPException)
    async def http_error(_req, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content={"error": detail})

    @app.middleware("http")
    async def revalidate_assets(request, call_next):
        """Make the browser check before reusing a script or stylesheet.

        StaticFiles sends an ETag and Last-Modified but no Cache-Control, which
        leaves the browser free to heuristically cache — so a deploy can leave
        someone running yesterday's JavaScript against today's API, with no
        error to explain it. `no-cache` does not mean "do not store": it means
        revalidate, so an unchanged file still comes back as a cheap 304.
        """
        response = await call_next(request)
        path = request.url.path
        if path.endswith((".js", ".css", ".html")) or path in ("/", "/console", "/login"):
            response.headers.setdefault("Cache-Control", "no-cache")
        return response

    if WEB.exists():
        app.mount("/", StaticFiles(directory=WEB, html=True), name="web")

    return app


app = create_app()
