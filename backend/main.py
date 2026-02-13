from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import aiohttp
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import db
from auth import InitDataError, get_user_id
from banana_client import ApiStatus, BananaCreateRequest, SosanoBananaAPIClient
from image_utils import normalize_collage, prepare_background, remove_background
from models import (
    AssetInfo,
    CreateSceneRequest,
    CreateSceneResponse,
    GenerateResponse,
    JobResponse,
    SendToChatRequest,
    SendToChatResponse,
    UploadObjectResponse,
)
from s3_client import S3Client

logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get('BOT_TOKEN', '').strip()
MEDIA_DIR = Path(os.environ.get('MEDIA_DIR', '/app/data/media'))
MEDIA_PUBLIC_BASE_URL = os.environ.get('MEDIA_PUBLIC_BASE_URL', '').strip().rstrip('/')
MAX_UPLOAD_BYTES = int(os.environ.get('MAX_UPLOAD_MB', '12')) * 1024 * 1024

BANANA_BASE_URL = os.environ.get('BANANA_BASE_URL', '').strip().rstrip('/')
BANANA_BEARER_TOKEN = os.environ.get('BANANA_BEARER_TOKEN', '').strip()
BANANA_DEFAULT_MODEL = os.environ.get('BANANA_MODEL', 'nano-banana-pro-2k').strip()
BANANA_TIMEOUT_SECONDS = float(os.environ.get('BANANA_TIMEOUT_SECONDS', '300'))

S3_OBJECT_PREFIX = os.environ.get('S3_OBJECT_PREFIX', 'tg-mini-app').strip().strip('/')

PROMPTS: dict[str, str] = {
    'photorealistic_v1': (
        '**Type:** Photorealistic Interior Reconstruction\n\n'
        '**Source Inputs:**\n'
        '1. Base empty interior photo = architectural structure (unchanged).\n'
        '2. Collage layout reference = absolute placement and scale guide.\n'
        '3. Individual furniture photos = material and geometry reference.\n\n'
        '**Core Spatial Rule (Strict Enforcement):**\n'
        '- Every furniture item must occupy the exact relative position shown in the collage.\n'
        '- Maintain identical proportional scale relationships between objects as seen in the collage.\n'
        '- Preserve relative distances to walls, corners, windows, and other objects.\n'
        '- Do not reinterpret composition — replicate it precisely.\n\n'
        '**Adaptive Alignment Logic:**\n'
        '- If an item is adjacent to a wall in the collage → place it flush and parallel to the corresponding wall in the base interior.\n'
        '- If centered in the collage → center it proportionally within the room width.\n'
        '- If angled → replicate the exact rotation angle.\n'
        '- Ensure all objects rest physically on the floor plane.\n'
        '- Apply realistic weight distribution and contact shadows.\n\n'
        '**Scale Control:**\n'
        '- Match visual height relative to doors, windows, and wall height.\n'
        '- Preserve object-to-object proportions exactly as in the collage.\n'
        '- No unintended resizing.\n\n'
        '**Lighting & Integration:**\n'
        '- Match original light direction from base interior.\n'
        '- Generate physically accurate shadows aligned with window direction.\n'
        '- Apply global illumination and wall bounce.\n'
        '- Soft natural daylight with ambient occlusion.\n'
        '- Volumetric window light rays.\n\n'
        '**Camera & Technical Parameters:**\n'
        '- Use original camera perspective from base image.\n'
        '- 35mm full-frame lens equivalence.\n'
        '- f/5.6 aperture for architectural sharpness.\n'
        '- ISO 100\n'
        '- High dynamic range\n'
        '- 8K ultra-realistic rendering.\n'
        '- Crisp edges, physically accurate materials.\n\n'
        '**Final Objective:**\n'
        'Produce a seamless photorealistic interior image where furniture placement, scale, and spatial relationships are identical to the collage, while being perfectly integrated into the real architectural environment.'
    )
}

app = FastAPI(title='TG Interior Mini App API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.mount('/media', StaticFiles(directory=str(MEDIA_DIR), check_dir=False), name='media')


@app.on_event('startup')
async def startup() -> None:
    if not BOT_TOKEN:
        raise RuntimeError('BOT_TOKEN is not set')

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    db.init_db()

    app.state.tasks = set()
    app.state.banana_client = None
    app.state.s3_client = None

    if BANANA_BASE_URL and BANANA_BEARER_TOKEN:
        app.state.banana_client = SosanoBananaAPIClient(
            base_url=BANANA_BASE_URL,
            bearer_token=BANANA_BEARER_TOKEN,
        )

    s3_client = S3Client()
    if s3_client.is_configured():
        app.state.s3_client = s3_client


@app.on_event('shutdown')
async def shutdown() -> None:
    tasks = list(getattr(app.state, 'tasks', set()))
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    client: SosanoBananaAPIClient | None = getattr(app.state, 'banana_client', None)
    if client is not None:
        await client.close()


def _user_id_from_init_data(init_data: str) -> str:
    try:
        return get_user_id(init_data, BOT_TOKEN)
    except InitDataError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def _public_url_for_storage_key(request: Request, storage_key: str) -> str:
    if MEDIA_PUBLIC_BASE_URL:
        return f'{MEDIA_PUBLIC_BASE_URL}/media/{storage_key}'

    proto = request.headers.get('x-forwarded-proto') or request.url.scheme
    host = request.headers.get('x-forwarded-host') or request.headers.get('host')
    if host:
        return f'{proto}://{host}/media/{storage_key}'
    return f'/media/{storage_key}'


def _save_bytes(user_id: str, *, data: bytes, suffix: str) -> tuple[str, Path]:
    storage_key = f'{user_id}/{uuid.uuid4().hex}{suffix}'
    path = MEDIA_DIR / storage_key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return storage_key, path


def _asset_info(asset_row: dict[str, Any]) -> AssetInfo:
    return AssetInfo(
        asset_id=asset_row['id'],
        url=asset_row['public_url'],
        width=int(asset_row['width']),
        height=int(asset_row['height']),
        mime_type=asset_row['mime_type'],
        kind=asset_row['kind'],
        object_label=asset_row.get('object_type'),
    )


async def _read_image_file(upload: UploadFile) -> bytes:
    if not upload.content_type or not upload.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='Only image uploads are allowed')

    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail='Uploaded file is empty')
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f'File is too large. Max {MAX_UPLOAD_BYTES // (1024 * 1024)}MB',
        )
    return data


async def _publish_collage_url(request: Request, storage_key: str, local_path: Path) -> str:
    s3_client: S3Client | None = getattr(app.state, 's3_client', None)
    if s3_client is None:
        url = _public_url_for_storage_key(request, storage_key)
        if not url.startswith('http://') and not url.startswith('https://'):
            raise HTTPException(
                status_code=500,
                detail='Cannot publish collage URL without MEDIA_PUBLIC_BASE_URL or public host headers',
            )
        return url

    bucket_key_prefix = f'{S3_OBJECT_PREFIX}/' if S3_OBJECT_PREFIX else ''
    object_key = f'{bucket_key_prefix}{storage_key}'

    try:
        return await s3_client.upload_file(local_path, key=object_key)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _register_asset(
    request: Request,
    *,
    user_id: str,
    kind: str,
    storage_key: str,
    mime_type: str,
    width: int,
    height: int,
    object_type: str | None = None,
    source_asset_id: str | None = None,
) -> AssetInfo:
    public_url = _public_url_for_storage_key(request, storage_key)
    asset_id = db.create_asset(
        user_id,
        kind=kind,
        storage_key=storage_key,
        public_url=public_url,
        mime_type=mime_type,
        width=width,
        height=height,
        object_type=object_type,
        source_asset_id=source_asset_id,
    )
    row = db.get_asset(asset_id, user_id)
    if row is None:
        raise HTTPException(status_code=500, detail='Failed to create asset')
    return _asset_info(row)


def _invalid_image_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=f'Invalid image file: {exc}')


def _is_http_url(value: str) -> bool:
    return value.startswith('http://') or value.startswith('https://')


def _asset_local_path(storage_key: str) -> Path:
    return MEDIA_DIR / storage_key


async def _publish_existing_asset_url(
    request: Request,
    *,
    asset: dict[str, Any],
    required: bool,
    context: str,
) -> str | None:
    storage_key = str(asset.get('storage_key') or '').strip()
    if storage_key:
        local_path = _asset_local_path(storage_key)
        if local_path.exists() and local_path.is_file():
            try:
                return await _publish_collage_url(request, storage_key, local_path)
            except HTTPException:
                if required:
                    raise
                logger.warning('Skipping %s: failed to publish asset %s', context, asset.get('id'))
                return None

    public_url = str(asset.get('public_url') or '').strip()
    if _is_http_url(public_url):
        return public_url

    if storage_key:
        rebuilt_url = _public_url_for_storage_key(request, storage_key)
        if _is_http_url(rebuilt_url):
            return rebuilt_url

    if required:
        raise HTTPException(status_code=500, detail=f'Cannot publish URL for {context}')

    logger.warning('Skipping %s: no public URL for asset %s', context, asset.get('id'))
    return None


async def _build_generation_image_urls(
    request: Request,
    *,
    user_id: str,
    scene: dict[str, Any],
    collage_url: str,
) -> tuple[list[str], list[str]]:
    background_asset_id = str(scene.get('background_asset_id') or '').strip()
    background = db.get_asset(background_asset_id, user_id) if background_asset_id else None
    if background is None:
        raise HTTPException(status_code=404, detail='Background asset not found')
    if background.get('kind') != 'background':
        raise HTTPException(status_code=400, detail='Scene background asset is invalid')

    background_url = await _publish_existing_asset_url(
        request,
        asset=background,
        required=True,
        context='background',
    )
    assert background_url is not None  # required=True guarantees this

    furniture_urls: list[str] = []
    scene_objects = scene.get('objects') or []
    for index, item in enumerate(scene_objects):
        if not isinstance(item, dict):
            logger.warning('Skipping object at index=%s: invalid scene object payload', index)
            continue

        cutout_asset_id = str(item.get('asset_id') or '').strip()
        if not cutout_asset_id:
            logger.warning('Skipping object at index=%s: empty asset_id', index)
            continue

        cutout_asset = db.get_asset(cutout_asset_id, user_id)
        if cutout_asset is None:
            logger.warning('Skipping object at index=%s: cutout asset %s not found', index, cutout_asset_id)
            continue
        if cutout_asset.get('kind') != 'object_cutout':
            logger.warning(
                'Skipping object at index=%s: asset %s has unexpected kind %s',
                index,
                cutout_asset_id,
                cutout_asset.get('kind'),
            )
            continue

        source_asset_id = str(cutout_asset.get('source_asset_id') or '').strip()
        if not source_asset_id:
            logger.warning('Skipping object at index=%s: cutout asset %s has no source_asset_id', index, cutout_asset_id)
            continue

        source_asset = db.get_asset(source_asset_id, user_id)
        if source_asset is None:
            logger.warning(
                'Skipping object at index=%s: source asset %s not found',
                index,
                source_asset_id,
            )
            continue
        if source_asset.get('kind') != 'object_source':
            logger.warning(
                'Skipping object at index=%s: source asset %s has unexpected kind %s',
                index,
                source_asset_id,
                source_asset.get('kind'),
            )
            continue

        source_url = await _publish_existing_asset_url(
            request,
            asset=source_asset,
            required=False,
            context=f'object source #{index + 1}',
        )
        if source_url:
            furniture_urls.append(source_url)

    image_urls = [background_url, collage_url, *furniture_urls]
    image_kinds = ['background', 'collage', *[f'object_source_{i + 1}' for i in range(len(furniture_urls))]]
    return image_urls, image_kinds


def _normalize_object_label(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if len(normalized) > 80:
        raise HTTPException(status_code=400, detail='object_label must be at most 80 characters')
    return normalized


def _schedule_task(task: asyncio.Task[Any]) -> None:
    app.state.tasks.add(task)

    def _cleanup(done_task: asyncio.Task[Any]) -> None:
        app.state.tasks.discard(done_task)

    task.add_done_callback(_cleanup)


def _normalize_generated_result_url(url: str) -> str:
    value = url.strip()
    if not value:
        return value

    parsed = urlparse(value)
    if parsed.scheme.lower() != 'http' or not BANANA_BASE_URL:
        return value

    banana_base = urlparse(BANANA_BASE_URL)
    if (
        banana_base.scheme.lower() == 'https'
        and parsed.hostname
        and banana_base.hostname
        and parsed.hostname.lower() == banana_base.hostname.lower()
    ):
        return urlunparse(parsed._replace(scheme='https'))

    return value


async def _fetch_remote_image(url: str) -> tuple[bytes, str]:
    timeout = aiohttp.ClientTimeout(total=45)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, allow_redirects=True) as response:
            if response.status >= 400:
                raise HTTPException(status_code=502, detail='Failed to load generated image')
            data = await response.read()
            if not data:
                raise HTTPException(status_code=502, detail='Generated image is empty')
            content_type = response.headers.get('content-type', 'application/octet-stream').split(';')[0].strip()
            return data, content_type


def _read_local_media_image(media_url_path: str) -> tuple[bytes, str]:
    storage_key = media_url_path[len('/media/'):]
    candidate = (MEDIA_DIR / storage_key).resolve()
    media_root = MEDIA_DIR.resolve()
    if not str(candidate).startswith(str(media_root)):
        raise HTTPException(status_code=400, detail='Invalid media path')
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail='Generated image file not found')

    data = candidate.read_bytes()
    if not data:
        raise HTTPException(status_code=404, detail='Generated image file is empty')
    media_type = mimetypes.guess_type(str(candidate))[0] or 'application/octet-stream'
    return data, media_type


async def _run_generation_job(job_id: str, banana_request: BananaCreateRequest) -> None:
    client: SosanoBananaAPIClient | None = app.state.banana_client
    if client is None:
        db.set_job_status(job_id, status='failed', error='BANANA API is not configured')
        return

    try:
        create_response = await client.create_task_async(banana_request)
        db.set_job_running(job_id, banana_uid=create_response.uid)

        final_task = await client.poll_task(
            create_response.uid,
            interval=2.0,
            timeout=BANANA_TIMEOUT_SECONDS,
        )

        if final_task.status == ApiStatus.COMPLETED:
            result_url = (final_task.result_file_url or '').strip()
            if not result_url:
                db.set_job_status(
                    job_id,
                    status='failed',
                    error='Generation completed but result URL is missing',
                )
                return

            normalized_url = _normalize_generated_result_url(result_url)
            db.set_job_status(job_id, status='done', result_url=normalized_url)
            return

        if final_task.status == ApiStatus.MODERATED:
            db.set_job_status(job_id, status='moderated', error=final_task.error or 'Request was moderated')
            return

        db.set_job_status(job_id, status='failed', error=final_task.error or 'Generation failed')
    except TimeoutError:
        db.set_job_status(job_id, status='failed', error='Generation timed out')
    except Exception as exc:  # pragma: no cover - defensive production guard
        logger.exception('Generation job failed for %s', job_id)
        db.set_job_status(job_id, status='failed', error=str(exc))


@app.post('/api/upload/background', response_model=AssetInfo)
async def upload_background(
    request: Request,
    init_data: str = Form(...),
    file: UploadFile = File(...),
) -> AssetInfo:
    user_id = _user_id_from_init_data(init_data)
    raw_bytes = await _read_image_file(file)
    try:
        image_bytes, width, height = prepare_background(raw_bytes)
    except Exception as exc:
        raise _invalid_image_error(exc) from exc

    storage_key, _ = _save_bytes(user_id, data=image_bytes, suffix='.png')
    return _register_asset(
        request,
        user_id=user_id,
        kind='background',
        storage_key=storage_key,
        mime_type='image/png',
        width=width,
        height=height,
    )


@app.post('/api/upload/object', response_model=UploadObjectResponse)
async def upload_object(
    request: Request,
    init_data: str = Form(...),
    file: UploadFile = File(...),
    object_label: str | None = Form(default=None),
) -> UploadObjectResponse:
    user_id = _user_id_from_init_data(init_data)
    raw_bytes = await _read_image_file(file)
    normalized_label = _normalize_object_label(object_label)

    try:
        source_bytes, source_w, source_h = prepare_background(raw_bytes)
        cutout_bytes, cutout_w, cutout_h = remove_background(raw_bytes)
    except Exception as exc:
        raise _invalid_image_error(exc) from exc

    source_storage_key, _ = _save_bytes(user_id, data=source_bytes, suffix='.png')
    source_asset = _register_asset(
        request,
        user_id=user_id,
        kind='object_source',
        storage_key=source_storage_key,
        mime_type='image/png',
        width=source_w,
        height=source_h,
        object_type=normalized_label,
    )

    cutout_storage_key, _ = _save_bytes(user_id, data=cutout_bytes, suffix='.png')
    cutout_asset = _register_asset(
        request,
        user_id=user_id,
        kind='object_cutout',
        storage_key=cutout_storage_key,
        mime_type='image/png',
        width=cutout_w,
        height=cutout_h,
        object_type=normalized_label,
        source_asset_id=source_asset.asset_id,
    )

    return UploadObjectResponse(source=source_asset, cutout=cutout_asset)


@app.post('/api/scene', response_model=CreateSceneResponse)
def create_scene(payload: CreateSceneRequest) -> CreateSceneResponse:
    user_id = _user_id_from_init_data(payload.init_data)

    background = db.get_asset(payload.background_asset_id, user_id)
    if background is None or background.get('kind') != 'background':
        raise HTTPException(status_code=404, detail='Background asset not found')

    for obj in payload.objects:
        asset = db.get_asset(obj.asset_id, user_id)
        if asset is None:
            raise HTTPException(status_code=404, detail=f'Asset {obj.asset_id} not found')
        if asset.get('kind') != 'object_cutout':
            raise HTTPException(status_code=400, detail=f'Asset {obj.asset_id} is not an object cutout')

    scene_id = db.create_scene(
        user_id,
        background_asset_id=payload.background_asset_id,
        objects=[item.model_dump() for item in payload.objects],
    )
    return CreateSceneResponse(scene_id=scene_id)


@app.post('/api/generate', response_model=GenerateResponse)
async def generate(
    request: Request,
    init_data: str = Form(...),
    scene_id: str = Form(...),
    collage: UploadFile = File(...),
    prompt_mode: str = Form('photorealistic_v1'),
    model: str | None = Form(None),
    p: int | None = Form(None),
) -> GenerateResponse:
    user_id = _user_id_from_init_data(init_data)

    scene = db.get_scene(scene_id, user_id)
    if scene is None:
        raise HTTPException(status_code=404, detail='Scene not found')

    collage_bytes = await _read_image_file(collage)
    try:
        normalized_collage, collage_w, collage_h = normalize_collage(collage_bytes)
    except Exception as exc:
        raise _invalid_image_error(exc) from exc
    collage_key, collage_path = _save_bytes(user_id, data=normalized_collage, suffix='.png')

    collage_asset = _register_asset(
        request,
        user_id=user_id,
        kind='collage',
        storage_key=collage_key,
        mime_type='image/png',
        width=collage_w,
        height=collage_h,
    )

    collage_url = await _publish_collage_url(request, collage_key, collage_path)
    image_urls, image_kinds = await _build_generation_image_urls(
        request,
        user_id=user_id,
        scene=scene,
        collage_url=collage_url,
    )
    logger.info(
        'Prepared generation image order for scene_id=%s: total=%s kinds=%s',
        scene_id,
        len(image_urls),
        ','.join(f'{idx + 1}:{kind}' for idx, kind in enumerate(image_kinds)),
    )

    prompt = PROMPTS.get(prompt_mode, PROMPTS['photorealistic_v1'])

    banana_request = BananaCreateRequest(
        prompt=prompt,
        image_urls=image_urls,
        model=(model or BANANA_DEFAULT_MODEL),
        aspect_ratio='auto',
        p=p,
    )

    job_id = db.create_job(user_id, scene_id=scene_id, collage_asset_id=collage_asset.asset_id)
    task = asyncio.create_task(_run_generation_job(job_id, banana_request))
    _schedule_task(task)

    return GenerateResponse(job_id=job_id, status='queued')


@app.get('/api/jobs/{job_id}', response_model=JobResponse)
def get_job(job_id: str, init_data: str = Query(...)) -> JobResponse:
    user_id = _user_id_from_init_data(init_data)
    job = db.get_job(job_id, user_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job not found')

    return JobResponse(
        job_id=job['id'],
        status=job['status'],
        banana_uid=job['banana_uid'],
        result_url=job['result_url'],
        error=job['error'],
    )


@app.get('/api/jobs/{job_id}/result')
async def get_job_result(job_id: str, init_data: str = Query(...)) -> Response:
    user_id = _user_id_from_init_data(init_data)
    job = db.get_job(job_id, user_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job not found')

    if job.get('status') != 'done' or not job.get('result_url'):
        raise HTTPException(status_code=400, detail='Job result is not ready')

    result_url = str(job['result_url']).strip()
    if not result_url:
        raise HTTPException(status_code=400, detail='Job result URL is empty')

    if result_url.startswith('/media/'):
        data, content_type = _read_local_media_image(result_url)
    else:
        parsed = urlparse(result_url)
        if parsed.scheme not in ('http', 'https'):
            raise HTTPException(status_code=400, detail='Unsupported result URL format')
        data, content_type = await _fetch_remote_image(result_url)

    return Response(
        content=data,
        media_type=content_type,
        headers={
            'Cache-Control': 'private, max-age=300',
        },
    )


@app.post('/api/send-to-chat', response_model=SendToChatResponse)
async def send_to_chat(payload: SendToChatRequest) -> SendToChatResponse:
    user_id = _user_id_from_init_data(payload.init_data)
    job = db.get_job(payload.job_id, user_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job not found')

    if job.get('status') != 'done' or not job.get('result_url'):
        raise HTTPException(status_code=400, detail='Job is not completed yet')

    telegram_url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto'
    payload_json = {
        'chat_id': user_id,
        'photo': job['result_url'],
        'caption': 'Your generated interior is ready.',
    }

    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(telegram_url, json=payload_json) as response:
            data = await response.json()
            if response.status >= 400 or not data.get('ok'):
                raise HTTPException(status_code=502, detail='Failed to send photo to Telegram chat')

    return SendToChatResponse(ok=True)
