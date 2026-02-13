from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Literal

import aiohttp
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ApiStatus(str, Enum):
    PROCESSING = 'PROCESSING'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'
    MODERATED = 'MODERATED'


class BananaCreateRequest(BaseModel):
    prompt: str
    image_urls: list[str] | None = None
    webhook_url: str | None = None
    model: Literal[
        'nano-banana',
        'nano-banana-pro-1k',
        'nano-banana-pro-2k',
        'nano-banana-pro-4k',
    ] = 'nano-banana-pro-2k'
    aspect_ratio: Literal[
        '1:1',
        '2:3',
        '3:2',
        '3:4',
        '4:3',
        '4:5',
        '5:4',
        '9:16',
        '16:9',
        '21:9',
        'auto',
    ] = 'auto'
    p: int | None = None


class BananaTaskResponse(BaseModel):
    uid: str
    status: ApiStatus
    created_at: datetime
    prompt: str
    optimized_prompt: str | None = None
    webhook_url: str | None = None
    result_file_url: str | None = Field(None, alias='result_file_url')
    error: str | None = None
    elapsed: float | None = None


class UserInfo(BaseModel):
    id: int
    name: str
    balance: float


class SosanoBananaAPIClient:
    def __init__(self, *, base_url: str, bearer_token: str) -> None:
        self.base_url = base_url.rstrip('/')
        self.bearer_token = bearer_token
        self.session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            headers = {'Authorization': f'Bearer {self.bearer_token}'}
            timeout = aiohttp.ClientTimeout(total=120)
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self.session

    async def close(self) -> None:
        if self.session and not self.session.closed:
            await self.session.close()

    async def create_task_async(self, request: BananaCreateRequest) -> BananaTaskResponse:
        session = await self._get_session()
        url = f'{self.base_url}/api/banana/create-async'
        payload = request.model_dump(exclude_none=True)
        logger.info('Banana create-async payload: %s', payload)

        async with session.post(url, json=payload) as response:
            data = await response.json()
            logger.info('Banana create-async response [%s]: %s', response.status, data)
            response.raise_for_status()
            return BananaTaskResponse.model_validate(data)

    async def get_task(self, uid: str) -> BananaTaskResponse:
        session = await self._get_session()
        url = f'{self.base_url}/api/banana/{uid}'
        async with session.get(url) as response:
            data = await response.json()
            if data.get('status') != ApiStatus.PROCESSING.value:
                logger.info('Banana get_task [%s]: %s', uid, data)
            response.raise_for_status()
            return BananaTaskResponse.model_validate(data)

    async def get_user_info(self) -> UserInfo:
        session = await self._get_session()
        url = f'{self.base_url}/api/user/me'

        async with session.get(url) as response:
            response.raise_for_status()
            data = await response.json()
            return UserInfo.model_validate(data)

    async def poll_task(
        self,
        uid: str,
        *,
        interval: float = 2.0,
        timeout: float = 300.0,
    ) -> BananaTaskResponse:
        deadline = asyncio.get_event_loop().time() + timeout

        while True:
            task = await self.get_task(uid)
            if task.status in (ApiStatus.COMPLETED, ApiStatus.FAILED, ApiStatus.MODERATED):
                return task

            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(f'Task {uid} did not complete within {timeout}s')

            await asyncio.sleep(interval)
