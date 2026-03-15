from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.bootstrap import ApplicationContext, create_application_context
from app.config import Settings, get_settings
from app.errors import InvalidInputError, ServiceBusyError, ServiceNotReadyError
from app.logging import configure_logging
from app.models import DemucsOptions, WhisperXOptions
from app.utils.files import remove_tree

logger = logging.getLogger("app.http")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.app_log_level)
    context = create_application_context()
    app.state.context = context

    if settings.preload_on_startup:
        asyncio.create_task(_startup_preload(context))
    yield


async def _startup_preload(context: ApplicationContext) -> None:
    try:
        logger.info("Application startup preload started")
        await context.preload_all()
        logger.info("Application startup preload finished")
    except Exception:
        logger.exception("Application startup preload failed")
        return


app = FastAPI(
    title="Demucs + WhisperX CUDA Server",
    version="0.1.0",
    lifespan=lifespan,
)


def get_context() -> ApplicationContext:
    return app.state.context


def require_api_key(
    settings: Settings = Depends(get_settings),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key.")


def parse_options(raw_json: str | None, model_type):
    if not raw_json:
        return model_type()
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid options JSON: {exc}") from exc
    try:
        return model_type.model_validate(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid options payload: {exc}") from exc


def cleanup_job(background_tasks: BackgroundTasks, job_dir: Path) -> None:
    logger.info("Scheduling job cleanup | job_dir=%s", job_dir)
    background_tasks.add_task(remove_tree, job_dir)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready(context: ApplicationContext = Depends(get_context)) -> JSONResponse:
    snapshot = context.snapshot()
    status = 200 if snapshot["overall_ready"] else 503
    return JSONResponse(content=snapshot, status_code=status)


@app.get("/status")
async def status(context: ApplicationContext = Depends(get_context)) -> dict:
    return context.snapshot()


@app.get("/status/demucs")
async def demucs_status(context: ApplicationContext = Depends(get_context)) -> dict:
    return context.demucs.snapshot().model_dump(mode="json")


@app.get("/status/whisperx")
async def whisperx_status(context: ApplicationContext = Depends(get_context)) -> dict:
    return context.whisperx.snapshot().model_dump(mode="json")


@app.post("/demucs/process", dependencies=[Depends(require_api_key)])
async def demucs_process(
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(default=None),
    source_url: str | None = Form(default=None),
    source_path: str | None = Form(default=None),
    options_json: str | None = Form(default=None),
    context: ApplicationContext = Depends(get_context),
):
    options = parse_options(options_json, DemucsOptions)
    logger.info(
        "HTTP request accepted | endpoint=/demucs/process | source_kind=%s | response_mode=%s | model=%s",
        "upload" if file is not None else "source_url" if source_url else "source_path" if source_path else "unknown",
        options.response_mode,
        options.model or "default",
    )
    try:
        result, job_dir = await context.demucs.process(
            upload=file,
            source_url=source_url,
            source_path=source_path,
            options=options,
        )
    except ServiceNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ServiceBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    cleanup_job(background_tasks, job_dir)
    if options.response_mode == "manifest":
        logger.info("HTTP response ready | endpoint=/demucs/process | response_mode=manifest")
        return result

    logger.info("HTTP response ready | endpoint=/demucs/process | response_mode=archive | path=%s", result)
    return FileResponse(path=result, filename="demucs-output.zip", media_type="application/zip")


@app.post("/whisperx/process", dependencies=[Depends(require_api_key)])
async def whisperx_process(
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(default=None),
    source_url: str | None = Form(default=None),
    source_path: str | None = Form(default=None),
    options_json: str | None = Form(default=None),
    context: ApplicationContext = Depends(get_context),
):
    options = parse_options(options_json, WhisperXOptions)
    logger.info(
        "HTTP request accepted | endpoint=/whisperx/process | source_kind=%s | response_mode=%s | model=%s | align=%s | diarize=%s",
        "upload" if file is not None else "source_url" if source_url else "source_path" if source_path else "unknown",
        options.response_mode,
        options.model or "default",
        options.align,
        options.diarize,
    )
    try:
        result, job_dir = await context.whisperx.process(
            upload=file,
            source_url=source_url,
            source_path=source_path,
            options=options,
        )
    except ServiceNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ServiceBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    cleanup_job(background_tasks, job_dir)
    if options.response_mode == "json":
        logger.info("HTTP response ready | endpoint=/whisperx/process | response_mode=json")
        return result

    logger.info("HTTP response ready | endpoint=/whisperx/process | response_mode=archive | path=%s", result)
    return FileResponse(path=result, filename="whisperx-output.zip", media_type="application/zip")
