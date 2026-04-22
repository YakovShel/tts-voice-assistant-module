from __future__ import annotations

from time import perf_counter
import shutil
from pathlib import Path
from typing import Iterable

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.assistant_logic import generate_assistant_reply
from app.cache_utils import build_cache_key, get_cached_file_path, has_cached_file
from app.config import MAX_TEXT_LENGTH, TEMPLATES_DIR
from app.history_utils import log_history, read_history
from app.stats_utils import compute_stats
from app.tts_engine import (
    synthesize_to_file,
    stream_kokoro_pcm,
    stream_wav_file_as_pcm,
    get_system_voices,
    get_neural_voices,
    get_engine_capabilities,
)

app = FastAPI(title='TTS Service')
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class SynthesizeRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_LENGTH)
    engine: str = 'system'
    voice: str = 'default'
    rate: float = 1.0


class AssistantRequest(BaseModel):
    text: str = Field('', max_length=MAX_TEXT_LENGTH)


@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request=request, name='index.html')


@app.get('/health')
def health():
    return {'status': 'ok', 'capabilities': get_engine_capabilities()}


@app.get('/engines')
def get_engines():
    caps = get_engine_capabilities()
    return {'engines': ['system', 'neural', 'auto'], 'capabilities': caps}


@app.get('/capabilities')
def capabilities():
    return get_engine_capabilities()


@app.get('/voices')
def get_voices():
    return {'system': get_system_voices(), 'neural': get_neural_voices()}


@app.get('/history')
def get_history(limit: int = Query(default=20, ge=1, le=100)):
    return {'items': read_history(limit=limit)}


@app.get('/stats')
def stats():
    return compute_stats()


@app.post('/assistant/respond')
def assistant_respond(request: AssistantRequest):
    text = normalize_text(request.text, allow_empty=True)
    return {'request_text': text, 'reply_text': generate_assistant_reply(text)}


def normalize_text(text: str, *, allow_empty: bool = False) -> str:
    text = ' '.join((text or '').strip().split())
    if not text:
        if allow_empty:
            return ''
        raise HTTPException(status_code=400, detail='Text must not be empty')
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail=f'Text is too long. Limit: {MAX_TEXT_LENGTH} characters')
    return text


def resolve_engine(engine: str, text: str) -> str:
    caps = get_engine_capabilities()
    selected_engine = engine

    if selected_engine == 'auto':
        preferred = 'system' if len(text) <= 80 else 'neural'
        fallback = 'neural' if preferred == 'system' else 'system'
        if caps.get(preferred, {}).get('available'):
            selected_engine = preferred
        elif caps.get(fallback, {}).get('available'):
            selected_engine = fallback
        else:
            raise HTTPException(status_code=503, detail='No synthesis engine is available')

    if selected_engine not in {'system', 'neural'}:
        raise HTTPException(status_code=400, detail="engine must be 'system', 'neural', or 'auto'")

    if not caps.get(selected_engine, {}).get('available'):
        reason = caps.get(selected_engine, {}).get('reason') or 'engine unavailable'
        raise HTTPException(status_code=503, detail=f'{selected_engine} engine unavailable: {reason}')

    return selected_engine


def validate_voice_for_engine(engine: str, voice: str) -> None:
    if voice == 'default':
        return
    available = get_system_voices() if engine == 'system' else get_neural_voices()
    if available and voice not in available:
        raise HTTPException(status_code=400, detail=f'Unknown {engine} voice: {voice}')


def _safe_unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _log_failure(mode: str, request_engine: str, selected_engine: str | None, voice: str, rate: float, text: str, start_time: float, error: str) -> None:
    elapsed = round(perf_counter() - start_time, 4)
    log_history({
        'mode': mode,
        'text': text,
        'engine_requested': request_engine,
        'engine_used': selected_engine,
        'voice': voice,
        'rate': rate,
        'cache_hit': False,
        'duration_sec': elapsed,
        'file_path': None,
        'completed': False,
        'error': error,
    })


def _empty_stream() -> Iterable[bytes]:
    if False:
        yield b''


@app.post('/synthesize')
def synthesize(request: SynthesizeRequest):
    start_time = perf_counter()
    selected_engine = None
    try:
        text = normalize_text(request.text)
        if request.rate <= 0:
            raise HTTPException(status_code=400, detail='rate must be greater than 0')

        selected_engine = resolve_engine(request.engine, text)
        validate_voice_for_engine(selected_engine, request.voice)

        cache_key = build_cache_key(text=text, engine=selected_engine, voice=request.voice, rate=request.rate)
        cached_path = get_cached_file_path(cache_key, '.wav')
        cache_hit = False

        if has_cached_file(cache_key, '.wav'):
            cache_hit = True
            final_path = cached_path
        else:
            generated_path = synthesize_to_file(text=text, engine=selected_engine, voice=request.voice, rate=request.rate)
            try:
                shutil.move(str(generated_path), str(cached_path))
            finally:
                _safe_unlink(generated_path)
            final_path = cached_path

        elapsed = round(perf_counter() - start_time, 4)
        log_history({
            'mode': 'file',
            'text': text,
            'engine_requested': request.engine,
            'engine_used': selected_engine,
            'voice': request.voice,
            'rate': request.rate,
            'cache_hit': cache_hit,
            'duration_sec': elapsed,
            'file_path': str(final_path),
            'completed': True,
            'error': None,
        })

        return FileResponse(path=final_path, media_type='audio/wav', filename=final_path.name)
    except HTTPException as exc:
        _log_failure('file', request.engine, selected_engine, request.voice, request.rate, (request.text or '')[:MAX_TEXT_LENGTH], start_time, str(exc.detail))
        raise
    except RuntimeError as exc:
        _log_failure('file', request.engine, selected_engine, request.voice, request.rate, (request.text or '')[:MAX_TEXT_LENGTH], start_time, str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        _log_failure('file', request.engine, selected_engine, request.voice, request.rate, (request.text or '')[:MAX_TEXT_LENGTH], start_time, f'Synthesis failed: {exc}')
        raise HTTPException(status_code=500, detail=f'Synthesis failed: {exc}') from exc


@app.post('/synthesize-stream')
def synthesize_stream(request: SynthesizeRequest):
    start_time = perf_counter()
    raw_text = request.text or ''
    text = normalize_text(raw_text, allow_empty=True)
    if not text:
        log_history({
            'mode': 'stream',
            'text': '',
            'engine_requested': request.engine,
            'engine_used': None,
            'voice': request.voice,
            'rate': request.rate,
            'cache_hit': False,
            'duration_sec': round(perf_counter() - start_time, 4),
            'file_path': None,
            'completed': True,
            'error': None,
        })
        return StreamingResponse(_empty_stream(), media_type='application/octet-stream', headers={
            'X-Audio-Format': 'pcm-f32le', 'X-Sample-Rate': '24000', 'X-Channels': '1', 'X-Empty-Stream': '1'
        })

    if request.rate <= 0:
        raise HTTPException(status_code=400, detail='rate must be greater than 0')

    selected_engine = request.engine if request.engine != 'auto' else 'neural'
    selected_engine = resolve_engine(selected_engine, text)
    if selected_engine != 'neural':
        raise HTTPException(status_code=400, detail='Streaming currently supports only neural engine')

    validate_voice_for_engine('neural', request.voice)
    cache_key = build_cache_key(text=text, engine='neural', voice=request.voice, rate=request.rate)
    cached_wav_path = get_cached_file_path(cache_key, '.wav')
    cache_hit = has_cached_file(cache_key, '.wav')

    try:
        raw_stream = stream_wav_file_as_pcm(cached_wav_path) if cache_hit else stream_kokoro_pcm(
            text=text,
            voice=request.voice,
            rate=request.rate,
            cache_wav_path=cached_wav_path,
        )
    except RuntimeError as exc:
        _log_failure('stream', request.engine, 'neural', request.voice, request.rate, text, start_time, str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    def tracked_stream():
        stream_error = None
        try:
            for chunk in raw_stream:
                yield chunk
        except Exception as exc:
            stream_error = exc
            raise
        finally:
            elapsed = round(perf_counter() - start_time, 4)
            final_file_path = str(cached_wav_path) if cached_wav_path.exists() else None
            log_history({
                'mode': 'stream',
                'text': text,
                'engine_requested': request.engine,
                'engine_used': 'neural',
                'voice': request.voice,
                'rate': request.rate,
                'cache_hit': cache_hit,
                'duration_sec': elapsed,
                'file_path': final_file_path,
                'completed': stream_error is None,
                'error': str(stream_error) if stream_error else None,
            })

    return StreamingResponse(tracked_stream(), media_type='application/octet-stream', headers={
        'X-Audio-Format': 'pcm-f32le', 'X-Sample-Rate': '24000', 'X-Channels': '1'
    })
