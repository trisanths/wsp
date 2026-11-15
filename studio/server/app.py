"""The wsp studio FastAPI app: the API under /api and the web app from studio/web/ at /."""

import asyncio
import json
import os
import time

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from studio.server import builder as builder_mod
from studio.server import catalog as catalog_mod
from studio.server import programs
from studio.server import runs as runs_mod
from studio.server import settings
from studio.server import specs as specs_mod

settings.load_env()

app = FastAPI(title="wsp studio")

STREAM_TICK_S = 0.5
STREAM_HEARTBEAT_S = 5.0
STREAM_LINGER_S = 3.0


def fail(message, status=400, line=None):
    body = {"error": message}
    if line is not None:
        body["line"] = line
    return JSONResponse(body, status_code=status)


@app.exception_handler(StarletteHTTPException)
async def http_error(request, error):
    return fail(str(error.detail), error.status_code)


@app.exception_handler(RequestValidationError)
async def validation_error(request, error):
    return fail("the request is not valid: " + str(error.errors()[0].get("msg", "")), 400)


async def read_json(request):
    """Read a JSON object body. Returns (body, None) or (None, error response)."""
    try:
        body = await request.json()
    except ValueError:
        return None, fail("the body must be JSON")
    if not isinstance(body, dict):
        return None, fail("the body must be a JSON object")
    return body, None


def health_body():
    return {"ok": True, "backend": settings.backend(), "llm": settings.llm(), "stub": settings.is_stub()}


@app.get("/api/health")
def health():
    return health_body()


# Specs.

@app.post("/api/spec")
async def create_spec(request: Request):
    form = await request.form()
    prompt = form.get("prompt")
    if prompt is None or not isinstance(prompt, str):
        prompt = ""
    uploads = []
    for item in form.getlist("files"):
        if isinstance(item, str):
            continue
        data = await item.read()
        uploads.append((item.filename or "upload", data))
    if prompt.strip() == "" and len(uploads) == 0:
        return fail("write a prompt or add a file")
    spec = await run_in_threadpool(specs_mod.draft_spec, prompt, uploads)
    return spec


@app.get("/api/spec/{spec_id}")
def get_spec(spec_id: str):
    spec = specs_mod.get_spec(spec_id)
    if spec is None:
        return fail("unknown spec", 404)
    return spec


@app.put("/api/spec/{spec_id}")
async def put_spec(spec_id: str, request: Request):
    body, problem = await read_json(request)
    if problem is not None:
        return problem
    try:
        spec = specs_mod.update_spec(spec_id, body)
    except ValueError as error:
        return fail(str(error))
    if spec is None:
        return fail("unknown spec", 404)
    return spec


# Runs.

def find_spec_or_error(spec_id):
    if not isinstance(spec_id, str):
        return None, fail("spec_id is required")
    spec = specs_mod.get_spec(spec_id)
    if spec is None:
        return None, fail("unknown spec", 404)
    return spec, None


def start_run_response(spec, program):
    try:
        run_id = runs_mod.start_run(spec, program)
    except runs_mod.RunError as error:
        return fail(str(error))
    except Exception as error:
        return fail("could not start the run: " + str(error), 500)
    return {"run_id": run_id}


@app.post("/api/runs")
async def create_run(request: Request):
    body, problem = await read_json(request)
    if problem is not None:
        return problem
    spec, problem = find_spec_or_error(body.get("spec_id"))
    if problem is not None:
        return problem
    return await run_in_threadpool(start_run_response, spec, body.get("program"))


def find_run(run_id):
    run = runs_mod.get_run(run_id)
    if run is None:
        return None, fail("unknown run", 404)
    return run, None


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run, problem = find_run(run_id)
    if problem is not None:
        return problem
    return run.state.snapshot()


@app.post("/api/runs/{run_id}/pause")
def pause_run(run_id: str):
    run, problem = find_run(run_id)
    if problem is not None:
        return problem
    run.pause()
    return {"ok": True}


@app.post("/api/runs/{run_id}/resume")
def resume_run(run_id: str):
    run, problem = find_run(run_id)
    if problem is not None:
        return problem
    run.resume()
    return {"ok": True}


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str):
    run, problem = find_run(run_id)
    if problem is not None:
        return problem
    run.stop()
    return {"ok": True}


@app.get("/api/runs/{run_id}/agents/{agent_id}")
def get_agent(run_id: str, agent_id: str):
    run, problem = find_run(run_id)
    if problem is not None:
        return problem
    detail = run.state.agent_detail(agent_id)
    if detail is None:
        return fail("unknown agent", 404)
    return detail


@app.post("/api/runs/{run_id}/agents/{agent_id}/nudge")
async def nudge_agent(run_id: str, agent_id: str, request: Request):
    run, problem = find_run(run_id)
    if problem is not None:
        return problem
    body, problem = await read_json(request)
    if problem is not None:
        return problem
    text = body.get("text")
    if not isinstance(text, str) or text.strip() == "":
        return fail("text is required")
    if not run.state.add_nudge(agent_id, text.strip()):
        return fail("unknown agent", 404)
    return {"ok": True}


def sse_event(snapshot):
    return "event: snapshot\ndata: " + json.dumps(snapshot) + "\n\n"


async def snapshot_stream(run):
    """Send a snapshot on every change (at most every 0.5 s) and a heartbeat every 5 s."""
    state = run.state
    last_version = -1
    last_sent = 0.0
    finished_at = None
    while True:
        version = state.version
        now = time.time()
        if version != last_version or now - last_sent >= STREAM_HEARTBEAT_S:
            yield sse_event(state.snapshot())
            last_version = version
            last_sent = now
        if state.status in ("done", "error"):
            if finished_at is None:
                finished_at = now
            if now - finished_at >= STREAM_LINGER_S:
                if state.version != last_version:
                    yield sse_event(state.snapshot())
                return
        await asyncio.sleep(STREAM_TICK_S)


@app.get("/api/runs/{run_id}/stream")
async def stream_run(run_id: str):
    run, problem = find_run(run_id)
    if problem is not None:
        return problem
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(snapshot_stream(run), media_type="text/event-stream", headers=headers)


# Catalog and program text.

@app.get("/api/catalog")
def get_catalog():
    return catalog_mod.build_palette(settings.backend())


@app.post("/api/program/text")
async def program_text(request: Request):
    body, problem = await read_json(request)
    if problem is not None:
        return problem
    try:
        return {"text": programs.render(body.get("program"))}
    except programs.ProgramError as error:
        return fail(error.message, 400, error.line)


@app.post("/api/program/parse")
async def program_parse(request: Request):
    body, problem = await read_json(request)
    if problem is not None:
        return problem
    text = body.get("text")
    if not isinstance(text, str):
        return fail("text is required")
    try:
        return {"program": programs.parse(text)}
    except programs.ProgramError as error:
        return fail(error.message, 400, error.line)


@app.post("/api/program/validate")
async def program_validate(request: Request):
    body, problem = await read_json(request)
    if problem is not None:
        return problem
    config, catalog = catalog_mod.load_catalog(settings.backend())
    try:
        clean = programs.normalize(body.get("program"))
    except programs.ProgramError as error:
        return {"errors": [error.message]}
    return {"errors": programs.validate(clean, catalog, config)}


# Builder.

def find_session(builder_id):
    session = builder_mod.get_session(builder_id)
    if session is None:
        return None, fail("unknown builder session", 404)
    return session, None


@app.post("/api/builder")
async def create_builder(request: Request):
    body, problem = await read_json(request)
    if problem is not None:
        return problem
    spec, problem = find_spec_or_error(body.get("spec_id"))
    if problem is not None:
        return problem
    session = await run_in_threadpool(builder_mod.create_session, spec)
    return {"builder_id": session.id, "budget": session.budget_dict(), "stub": session.env.is_stub()}


@app.post("/api/builder/{builder_id}/action")
async def builder_action(builder_id: str, request: Request):
    session, problem = find_session(builder_id)
    if problem is not None:
        return problem
    body, problem = await read_json(request)
    if problem is not None:
        return problem
    try:
        job_id = builder_mod.submit_action(session, body.get("program"), body.get("action"))
    except builder_mod.BuilderError as error:
        return fail(str(error))
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = builder_mod.get_job(job_id)
    if job is None:
        return fail("unknown job", 404)
    return job


@app.post("/api/builder/{builder_id}/suggest")
async def builder_suggest(builder_id: str, request: Request):
    session, problem = find_session(builder_id)
    if problem is not None:
        return problem
    body, problem = await read_json(request)
    if problem is not None:
        return problem
    try:
        suggestions = await run_in_threadpool(builder_mod.suggest, session, body.get("program"))
    except builder_mod.BuilderError as error:
        return fail(str(error))
    return {"suggestions": suggestions}


@app.post("/api/builder/{builder_id}/handoff")
async def builder_handoff(builder_id: str, request: Request):
    session, problem = find_session(builder_id)
    if problem is not None:
        return problem
    body, problem = await read_json(request)
    if problem is not None:
        return problem
    if not isinstance(body.get("program"), dict):
        return fail("program is required")
    return await run_in_threadpool(start_run_response, session.spec, body["program"])


# The web app.

@app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "DELETE"])
def unknown_api(rest: str):
    return fail("unknown route: /api/" + rest, 404)


def web_file(path):
    """The file under studio/web/ for a URL path, or None. Blocks paths that leave the folder."""
    root = os.path.realpath(settings.WEB_DIR)
    if path == "" or path.endswith("/"):
        path = path + "index.html"
    full = os.path.realpath(os.path.join(root, path))
    if full != root and not full.startswith(root + os.sep):
        return None
    if not os.path.isfile(full):
        return None
    return full


@app.get("/")
def web_index():
    full = web_file("")
    if full is None:
        return fail("studio/web/index.html is not built yet", 404)
    return FileResponse(full)


@app.get("/{path:path}")
def web_static(path: str):
    full = web_file(path)
    if full is None:
        return fail("not found", 404)
    return FileResponse(full)
