"""
LIFT Framework — FastAPI web server.
Serves the frontend and streams pipeline progress via SSE.

Run:
    python app.py
Then open http://localhost:8000
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import queue as q_module
import sys
import threading
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent))

# Load .env if present
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
            if _v:
                os.environ.setdefault(_k, _v)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── Startup key validation ────────────────────────────────────────────────────
# All LLM traffic routes through OpenRouter. Fail fast here so the user
# gets a clear message instead of a cryptic 401 mid-run.
if not os.environ.get("OPENROUTER_API_KEY"):
    raise EnvironmentError(
        "OPENROUTER_API_KEY is not set. "
        "Add OPENROUTER_API_KEY=<your-key> to your .env file and restart."
    )
# Guard: never let the OpenAI SDK silently route to api.openai.com.
# Unsetting OPENAI_API_KEY means openai.OpenAI() without an explicit
# base_url will raise AuthenticationError immediately instead of sending
# the OpenRouter key to the wrong endpoint.
os.environ.pop("OPENAI_API_KEY", None)

app = FastAPI(title="LIFT Framework")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# In-memory run store: run_id → {"queue": Queue, "stop_event": Event}
run_store: dict = {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html = (Path(__file__).parent / "static" / "index.html").read_text()
    return HTMLResponse(content=html)


@app.post("/run")
async def start_run(
    file: UploadFile = File(...),
    domain: str = Form(...),
    outcome_col: str = Form(...),
    protected_col: str = Form(...),
    cohort_notes: str = Form(...),
    id_col: str = Form(default=""),
) -> JSONResponse:
    """
    Accepts multipart form with CSV + dataset context.
    Starts the pipeline in a background daemon thread.
    Returns a run_id for the SSE stream.
    """
    run_id = str(uuid.uuid4())[:8]
    q: q_module.Queue = q_module.Queue()
    stop_event = threading.Event()
    run_store[run_id] = {"queue": q, "stop_event": stop_event}

    # Read bytes and filename now — UploadFile is not thread-safe
    content = await file.read()
    fname = file.filename or ""

    def run_pipeline() -> None:
        def progress_fn(event: dict) -> None:
            q.put(event)
            if stop_event.is_set():
                raise InterruptedError("Run stopped by user.")

        try:
            if fname.lower().endswith(".xlsx"):
                df = pd.read_excel(io.BytesIO(content))
            else:
                for _enc in ("utf-8", "utf-8-sig", "latin-1"):
                    try:
                        df = pd.read_csv(io.BytesIO(content), encoding=_enc)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError(
                        "Could not decode the CSV file. Please re-save it as UTF-8 and re-upload."
                    )

            # Drop ID column before anything else so it never counts as a feature
            _id_col = id_col.strip() if id_col else ""
            if _id_col and _id_col in df.columns:
                df = df.drop(columns=[_id_col])

            # Validate required columns exist before entering the pipeline
            missing = [c for c in (outcome_col, protected_col) if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Column(s) {missing} not found in the uploaded file. "
                    f"Available columns: {list(df.columns)}"
                )

            q.put({
                "stage": "upload",
                "status": "done",
                "message": f"Loaded {len(df):,} rows × {len(df.columns)} columns",
            })

            from lift.pipeline import LIFTPipeline
            from lift.schemas import DatasetContext

            pipeline = LIFTPipeline(config_path="lift/config.yaml")

            xi = DatasetContext(
                domain=domain,
                outcome_col=outcome_col,
                protected_col=protected_col,
                cohort_notes=cohort_notes,
                id_col=_id_col or None,
            )

            pipeline.run(df, xi, _progress_fn=progress_fn, _stop_check=stop_event.is_set)

        except InterruptedError:
            q.put({"stage": "error", "status": "error", "message": "Run stopped by user."})
        except Exception as exc:  # noqa: BLE001
            logging.exception("Pipeline error for run %s", run_id)
            q.put({"stage": "error", "status": "error", "message": str(exc)})

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    return JSONResponse({"run_id": run_id})


@app.post("/stop/{run_id}")
async def stop_run(run_id: str) -> JSONResponse:
    """Signals the pipeline thread to stop after the current stage."""
    entry = run_store.get(run_id)
    if not entry:
        return JSONResponse({"ok": False, "error": "Run not found"}, status_code=404)
    entry["stop_event"].set()
    return JSONResponse({"ok": True})


@app.get("/stream/{run_id}")
async def stream(run_id: str) -> StreamingResponse:
    """SSE endpoint — streams progress events until pipeline_done or error."""
    if run_id not in run_store:
        async def not_found():
            yield 'data: {"stage":"error","status":"error","message":"Run not found"}\n\n'
        return StreamingResponse(not_found(), media_type="text/event-stream")

    async def event_gen():
        q = run_store[run_id]["queue"]
        loop = asyncio.get_event_loop()

        while True:
            try:
                # Non-blocking get with 1-second timeout so we can send keepalives
                event = await loop.run_in_executor(
                    None, lambda: q.get(timeout=1)
                )
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("stage") in ("pipeline_done", "error"):
                    break
            except q_module.Empty:
                # Keepalive comment to prevent connection timeout
                yield ": ping\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
