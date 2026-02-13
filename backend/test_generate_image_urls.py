from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

import main


class _DummyRequest:
    def __init__(self) -> None:
        self.headers = {'x-forwarded-proto': 'https', 'x-forwarded-host': 'example.com'}
        self.url = SimpleNamespace(scheme='https')


class BuildGenerationImageUrlsTests(unittest.IsolatedAsyncioTestCase):
    async def test_builds_expected_order_background_collage_sources(self) -> None:
        request = _DummyRequest()
        scene = {
            'background_asset_id': 'bg1',
            'objects': [
                {'asset_id': 'cut1'},
                {'asset_id': 'cut2'},
            ],
        }
        assets = {
            'bg1': {'id': 'bg1', 'kind': 'background', 'storage_key': 'u/bg1.png'},
            'cut1': {'id': 'cut1', 'kind': 'object_cutout', 'source_asset_id': 'src1'},
            'cut2': {'id': 'cut2', 'kind': 'object_cutout', 'source_asset_id': 'src2'},
            'src1': {'id': 'src1', 'kind': 'object_source', 'storage_key': 'u/src1.png'},
            'src2': {'id': 'src2', 'kind': 'object_source', 'storage_key': 'u/src2.png'},
        }
        published_urls = {
            'bg1': 'https://cdn.example.com/bg1.png',
            'src1': 'https://cdn.example.com/src1.png',
            'src2': 'https://cdn.example.com/src2.png',
        }

        with patch('main.db.get_asset', side_effect=lambda asset_id, _user_id: assets.get(asset_id)):
            with patch(
                'main._publish_existing_asset_url',
                new=AsyncMock(side_effect=lambda *_args, **kwargs: published_urls[kwargs['asset']['id']]),
            ):
                image_urls, image_kinds = await main._build_generation_image_urls(
                    request,
                    user_id='u1',
                    scene=scene,
                    collage_url='https://cdn.example.com/collage.png',
                )

        self.assertEqual(
            image_urls,
            [
                'https://cdn.example.com/bg1.png',
                'https://cdn.example.com/collage.png',
                'https://cdn.example.com/src1.png',
                'https://cdn.example.com/src2.png',
            ],
        )
        self.assertEqual(image_kinds, ['background', 'collage', 'object_source_1', 'object_source_2'])

    async def test_skips_invalid_object_and_keeps_generation(self) -> None:
        request = _DummyRequest()
        scene = {
            'background_asset_id': 'bg1',
            'objects': [
                {'asset_id': 'cut_missing_source'},
                {'asset_id': 'cut_ok'},
            ],
        }
        assets = {
            'bg1': {'id': 'bg1', 'kind': 'background', 'storage_key': 'u/bg1.png'},
            'cut_missing_source': {'id': 'cut_missing_source', 'kind': 'object_cutout', 'source_asset_id': None},
            'cut_ok': {'id': 'cut_ok', 'kind': 'object_cutout', 'source_asset_id': 'src_ok'},
            'src_ok': {'id': 'src_ok', 'kind': 'object_source', 'storage_key': 'u/src_ok.png'},
        }
        published_urls = {
            'bg1': 'https://cdn.example.com/bg1.png',
            'src_ok': 'https://cdn.example.com/src_ok.png',
        }

        with patch('main.db.get_asset', side_effect=lambda asset_id, _user_id: assets.get(asset_id)):
            with patch(
                'main._publish_existing_asset_url',
                new=AsyncMock(side_effect=lambda *_args, **kwargs: published_urls.get(kwargs['asset']['id'])),
            ):
                image_urls, image_kinds = await main._build_generation_image_urls(
                    request,
                    user_id='u1',
                    scene=scene,
                    collage_url='https://cdn.example.com/collage.png',
                )

        self.assertEqual(
            image_urls,
            [
                'https://cdn.example.com/bg1.png',
                'https://cdn.example.com/collage.png',
                'https://cdn.example.com/src_ok.png',
            ],
        )
        self.assertEqual(image_kinds, ['background', 'collage', 'object_source_1'])

    async def test_raises_when_background_missing(self) -> None:
        request = _DummyRequest()
        scene = {'background_asset_id': 'missing_bg', 'objects': [{'asset_id': 'cut1'}]}

        with patch('main.db.get_asset', return_value=None):
            with self.assertRaises(HTTPException) as exc_ctx:
                await main._build_generation_image_urls(
                    request,
                    user_id='u1',
                    scene=scene,
                    collage_url='https://cdn.example.com/collage.png',
                )

        self.assertEqual(exc_ctx.exception.status_code, 404)
        self.assertEqual(exc_ctx.exception.detail, 'Background asset not found')


if __name__ == '__main__':
    unittest.main()
