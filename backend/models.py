from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


JobStatus = Literal['queued', 'running', 'done', 'failed', 'moderated']
MAX_SCENE_OBJECTS = 10


class AssetInfo(BaseModel):
    asset_id: str
    url: str
    width: int
    height: int
    mime_type: str
    kind: str
    object_label: str | None = None


class UploadObjectResponse(BaseModel):
    source: AssetInfo
    cutout: AssetInfo


class SceneObject(BaseModel):
    asset_id: str
    label: str | None = Field(default=None, max_length=80)
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    scale: float = Field(gt=0.01, le=1.2)
    rotation_deg: float = Field(ge=-180.0, le=180.0)
    z_index: int = Field(ge=0, le=100)


class CreateSceneRequest(BaseModel):
    init_data: str
    background_asset_id: str
    objects: list[SceneObject]

    @model_validator(mode='after')
    def validate_scene_objects(self) -> 'CreateSceneRequest':
        if len(self.objects) < 1:
            raise ValueError('scene must contain at least one object')
        if len(self.objects) > MAX_SCENE_OBJECTS:
            raise ValueError(f'scene supports at most {MAX_SCENE_OBJECTS} objects')

        asset_ids = [item.asset_id for item in self.objects]
        if len(set(asset_ids)) != len(asset_ids):
            raise ValueError('scene objects must reference unique asset_id values')
        return self


class CreateSceneResponse(BaseModel):
    scene_id: str


class GenerateResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    banana_uid: str | None = None
    result_url: str | None = None
    error: str | None = None


class SendToChatRequest(BaseModel):
    init_data: str
    job_id: str


class SendToChatResponse(BaseModel):
    ok: bool
