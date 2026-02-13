from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import aioboto3


class S3Client:
    """Client for S3-compatible storage operations."""

    def __init__(self) -> None:
        """Initialize the S3 client from environment variables."""
        self.endpoint_url = os.environ.get('S3_ENDPOINT_URL', '').strip() or None
        self.access_key_id = os.environ.get('S3_ACCESS_KEY_ID', '').strip()
        self.secret_access_key = os.environ.get('S3_SECRET_ACCESS_KEY', '').strip()
        self.bucket_name = os.environ.get('S3_BUCKET', '').strip()
        self.bucket_uuid = os.environ.get('S3_BUCKET_UUID', '').strip()
        self.region = os.environ.get('S3_REGION', '').strip() or None
        self.session = aioboto3.Session()

    def is_configured(self) -> bool:
        return bool(
            self.endpoint_url
            and self.access_key_id
            and self.secret_access_key
            and self.bucket_name
        )

    async def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Get a presigned URL for an S3 object."""
        async with self.session.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region,
        ) as s3:
            return await s3.generate_presigned_url(
                ClientMethod='get_object',
                Params={'Bucket': self.bucket_name, 'Key': key},
                ExpiresIn=expires_in,
            )

    async def upload_file(
        self,
        file_path: str | Path,
        key: str | None = None,
    ) -> str:
        """Upload a file to S3 and return a presigned URL."""
        if key is None:
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            file_name = Path(file_path).name
            key = f'uploads/{timestamp}_{file_name}'

        try:
            async with self.session.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                region_name=self.region,
            ) as s3:
                await s3.upload_file(str(file_path), self.bucket_name, key)
                return await s3.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': self.bucket_name, 'Key': key},
                    ExpiresIn=3600,
                )
        except Exception as exc:  # pragma: no cover - external service failure
            raise RuntimeError(f'Failed to upload file to S3: {exc}') from exc

    async def upload_bytes(
        self,
        data: bytes,
        key: str,
        content_type: str = 'application/octet-stream',
    ) -> str:
        """Upload bytes to S3 and return a presigned URL."""
        try:
            async with self.session.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                region_name=self.region,
            ) as s3:
                await s3.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=data,
                    ContentType=content_type,
                )
                return await s3.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': self.bucket_name, 'Key': key},
                    ExpiresIn=3600,
                )
        except Exception as exc:  # pragma: no cover - external service failure
            raise RuntimeError(f'Failed to upload bytes to S3: {exc}') from exc
