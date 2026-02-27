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

sys.path.insert(0, str(Path(__file__).parent))

# Load .env if present
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            _k, _v = _k.strip(), _v.strip()
            if _v:
                os.environ.setdefault(_k, _v)
                # Mirror OPENROUTER_API_KEY into OPENAI_API_KEY so the openai
                # SDK picks it up automatically when using the OpenRouter backend.
                if _k == "OPENROUTER_API_KEY":
                    os.environ.setdefault("OPENAI_API_KEY", _v)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(title="LIFT Framework")

# In-memory run store: run_id → {"queue": Queue}
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
    llm_model: str = Form(default="gpt-4o"),
) -> JSONResponse:
    """
    Accepts multipart form with CSV + dataset context.
    Starts the pipeline in a background daemon thread.
    Returns a run_id for the SSE stream.
    """
    run_id = str(uuid.uuid4())[:8]
    q: q_module.Queue = q_module.Queue()
    run_store[run_id] = {"queue": q}

    # Read bytes now — UploadFile is not thread-safe
    content = await file.read()

    def run_pipeline() -> None:
        try:
            df = pd.read_csv(io.BytesIO(content))
            q.put({
                "stage": "upload",
                "status": "done",
                "message": f"Loaded {len(df):,} rows × {len(df.columns)} columns",
            })

            from lift.pipeline import LIFTPipeline
            from lift.schemas import DatasetContext

            pipeline = LIFTPipeline(config_path="lift/config.yaml")
            # Override LLM model from user selection
            pipeline.config["llm"]["model"] = llm_model
            # Reinitialise agents with new model
            from lift.schemas import LLMConfig
            llm_cfg = LLMConfig(**pipeline.config["llm"])
            from lift.agents.data_profiler import DataProfiler
            from lift.agents.model_orchestrator import ModelOrchestrator
            from lift.agents.governance_reporter import GovernanceReporter
            pipeline._profiler = DataProfiler(llm_cfg)
            pipeline._orchestrator = ModelOrchestrator(llm_cfg)
            pipeline._reporter = GovernanceReporter(llm_cfg)

            xi = DatasetContext(
                domain=domain,
                outcome_col=outcome_col,
                protected_col=protected_col,
                cohort_notes=cohort_notes,
            )

            pipeline.run(df, xi, _progress_fn=q.put)

        except Exception as exc:  # noqa: BLE001
            logging.exception("Pipeline error for run %s", run_id)
            q.put({"stage": "error", "status": "error", "message": str(exc)})

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    return JSONResponse({"run_id": run_id})


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
