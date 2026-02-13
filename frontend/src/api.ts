export interface AssetInfo {
  asset_id: string;
  url: string;
  width: number;
  height: number;
  mime_type: string;
  kind: string;
  object_label?: string;
}

export interface UploadObjectResponse {
  source: AssetInfo;
  cutout: AssetInfo;
}

export interface SceneObject {
  asset_id: string;
  label?: string;
  x: number;
  y: number;
  scale: number;
  rotation_deg: number;
  z_index: number;
}

export interface JobResponse {
  job_id: string;
  status: 'queued' | 'running' | 'done' | 'failed' | 'moderated';
  banana_uid?: string | null;
  result_url?: string | null;
  error?: string | null;
}

async function parseError(response: Response): Promise<string> {
  try {
    const data = await response.json();
    if (data && typeof data.detail === 'string') {
      return data.detail;
    }
  } catch {
    // ignore json parse failures
  }
  return `Request failed with status ${response.status}`;
}

export async function uploadBackground(initData: string, file: File): Promise<AssetInfo> {
  const formData = new FormData();
  formData.set('init_data', initData);
  formData.set('file', file);

  const response = await fetch('/api/upload/background', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  return (await response.json()) as AssetInfo;
}

export async function uploadObject(
  initData: string,
  file: File,
  objectLabel?: string
): Promise<UploadObjectResponse> {
  const formData = new FormData();
  formData.set('init_data', initData);
  formData.set('file', file);
  if (objectLabel && objectLabel.trim().length > 0) {
    formData.set('object_label', objectLabel.trim());
  }

  const response = await fetch('/api/upload/object', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  return (await response.json()) as UploadObjectResponse;
}

export async function createScene(
  initData: string,
  backgroundAssetId: string,
  objects: SceneObject[]
): Promise<string> {
  const response = await fetch('/api/scene', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      init_data: initData,
      background_asset_id: backgroundAssetId,
      objects
    })
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  const data = (await response.json()) as { scene_id: string };
  return data.scene_id;
}

export async function generateInterior(
  initData: string,
  sceneId: string,
  collage: Blob,
  promptMode = 'photorealistic_v1'
): Promise<{ job_id: string; status: string }> {
  const formData = new FormData();
  formData.set('init_data', initData);
  formData.set('scene_id', sceneId);
  formData.set('prompt_mode', promptMode);
  formData.set('collage', collage, 'collage.png');

  const response = await fetch('/api/generate', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  return (await response.json()) as { job_id: string; status: string };
}

export async function getJob(initData: string, jobId: string): Promise<JobResponse> {
  const response = await fetch(
    `/api/jobs/${encodeURIComponent(jobId)}?init_data=${encodeURIComponent(initData)}`
  );

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  return (await response.json()) as JobResponse;
}

export async function sendToChat(initData: string, jobId: string): Promise<void> {
  const response = await fetch('/api/send-to-chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      init_data: initData,
      job_id: jobId
    })
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }
}
