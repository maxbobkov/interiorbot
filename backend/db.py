from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

DB_PATH = os.environ.get('DB_PATH', '/app/data/app.db')

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
    return _conn


def init_db() -> None:
    conn = _get_conn()
    with _lock:
        conn.execute(
            'CREATE TABLE IF NOT EXISTS assets ('
            'id TEXT PRIMARY KEY, '
            'user_id TEXT NOT NULL, '
            'kind TEXT NOT NULL, '
            'object_type TEXT, '
            'storage_key TEXT NOT NULL, '
            'public_url TEXT NOT NULL, '
            'mime_type TEXT NOT NULL, '
            'width INTEGER NOT NULL, '
            'height INTEGER NOT NULL, '
            'source_asset_id TEXT, '
            'created_at TEXT NOT NULL'
            ')'
        )
        conn.execute(
            'CREATE TABLE IF NOT EXISTS scenes ('
            'id TEXT PRIMARY KEY, '
            'user_id TEXT NOT NULL, '
            'background_asset_id TEXT NOT NULL, '
            'objects_json TEXT NOT NULL, '
            'created_at TEXT NOT NULL'
            ')'
        )
        conn.execute(
            'CREATE TABLE IF NOT EXISTS jobs ('
            'id TEXT PRIMARY KEY, '
            'user_id TEXT NOT NULL, '
            'scene_id TEXT NOT NULL, '
            'status TEXT NOT NULL, '
            'banana_uid TEXT, '
            'collage_asset_id TEXT NOT NULL, '
            'result_url TEXT, '
            'error TEXT, '
            'created_at TEXT NOT NULL, '
            'updated_at TEXT NOT NULL'
            ')'
        )
        conn.commit()


def create_asset(
    user_id: str,
    *,
    kind: str,
    storage_key: str,
    public_url: str,
    mime_type: str,
    width: int,
    height: int,
    object_type: str | None = None,
    source_asset_id: str | None = None,
) -> str:
    conn = _get_conn()
    asset_id = uuid.uuid4().hex
    created_at = _utcnow_iso()
    with _lock:
        conn.execute(
            'INSERT INTO assets ('
            'id, user_id, kind, object_type, storage_key, public_url, mime_type, width, height, source_asset_id, created_at'
            ') VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                asset_id,
                user_id,
                kind,
                object_type,
                storage_key,
                public_url,
                mime_type,
                width,
                height,
                source_asset_id,
                created_at,
            ),
        )
        conn.commit()
    return asset_id


def get_asset(asset_id: str, user_id: str) -> dict[str, Any] | None:
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            'SELECT * FROM assets WHERE id = ? AND user_id = ?',
            (asset_id, user_id),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return dict(row)


def create_scene(user_id: str, background_asset_id: str, objects: list[dict[str, Any]]) -> str:
    conn = _get_conn()
    scene_id = uuid.uuid4().hex
    created_at = _utcnow_iso()
    objects_json = json.dumps(objects, separators=(',', ':'))
    with _lock:
        conn.execute(
            'INSERT INTO scenes (id, user_id, background_asset_id, objects_json, created_at) VALUES (?, ?, ?, ?, ?)',
            (scene_id, user_id, background_asset_id, objects_json, created_at),
        )
        conn.commit()
    return scene_id


def get_scene(scene_id: str, user_id: str) -> dict[str, Any] | None:
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            'SELECT * FROM scenes WHERE id = ? AND user_id = ?',
            (scene_id, user_id),
        )
        row = cur.fetchone()
    if row is None:
        return None
    data = dict(row)
    data['objects'] = json.loads(data.pop('objects_json'))
    return data


def create_job(user_id: str, scene_id: str, collage_asset_id: str) -> str:
    conn = _get_conn()
    job_id = uuid.uuid4().hex
    now = _utcnow_iso()
    with _lock:
        conn.execute(
            'INSERT INTO jobs ('
            'id, user_id, scene_id, status, banana_uid, collage_asset_id, result_url, error, created_at, updated_at'
            ') VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (job_id, user_id, scene_id, 'queued', None, collage_asset_id, None, None, now, now),
        )
        conn.commit()
    return job_id


def set_job_running(job_id: str, banana_uid: str | None = None) -> None:
    conn = _get_conn()
    now = _utcnow_iso()
    with _lock:
        conn.execute(
            'UPDATE jobs SET status = ?, banana_uid = COALESCE(?, banana_uid), updated_at = ? WHERE id = ?',
            ('running', banana_uid, now, job_id),
        )
        conn.commit()


def set_job_status(
    job_id: str,
    *,
    status: str,
    error: str | None = None,
    result_url: str | None = None,
) -> None:
    conn = _get_conn()
    now = _utcnow_iso()
    with _lock:
        conn.execute(
            'UPDATE jobs SET status = ?, error = ?, result_url = ?, updated_at = ? WHERE id = ?',
            (status, error, result_url, now, job_id),
        )
        conn.commit()


def get_job(job_id: str, user_id: str) -> dict[str, Any] | None:
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            'SELECT * FROM jobs WHERE id = ? AND user_id = ?',
            (job_id, user_id),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return dict(row)
