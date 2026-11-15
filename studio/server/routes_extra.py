"""Routes for measured results and the front door link. app.py calls register() once."""

import logging

from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from studio.server import frontdoor
from studio.server import measured
from studio.server import specs as specs_mod

log = logging.getLogger("studio.routes_extra")


def error_body(message, status):
    return JSONResponse({"error": message}, status_code=status)


def register(app):
    """Add the routes to the app. Call this before the catch-all /api route."""

    @app.get("/api/measured")
    def get_measured():
        return measured.all_results()

    @app.get("/api/spec/{spec_id}/request")
    def get_spec_request(spec_id: str):
        spec = specs_mod.get_spec(spec_id)
        if spec is None:
            return error_body("unknown spec", 404)
        try:
            return frontdoor.request_view(spec)
        except Exception as error:
            log.exception("could not build the request for %s", spec_id)
            return error_body("could not turn this spec into a request: " + str(error), 422)

    @app.post("/api/spec/{spec_id}/plan")
    async def plan_spec(spec_id: str):
        spec = specs_mod.get_spec(spec_id)
        if spec is None:
            return error_body("unknown spec", 404)
        try:
            body, status = await run_in_threadpool(frontdoor.plan_for_spec, spec)
        except Exception as error:
            log.exception("plan route failed for %s", spec_id)
            return error_body("the plan failed: " + str(error), 422)
        return JSONResponse(body, status_code=status)
