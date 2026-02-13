import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent
} from 'react';
import {
  AssetInfo,
  JobResponse,
  SceneObject,
  UploadObjectResponse,
  createScene,
  generateInterior,
  getJob,
  sendToChat,
  uploadBackground,
  uploadObject
} from './api';

const MAX_OBJECTS = 10;

type AppMode = 'upload' | 'compose' | 'generating' | 'result';

interface ObjectSlot {
  slotId: number;
  label: string;
  upload: UploadObjectResponse | null;
}

interface Placement {
  slotId: number;
  label: string;
  assetId: string;
  url: string;
  aspect: number;
  x: number;
  y: number;
  scale: number;
  rotationDeg: number;
  zIndex: number;
}

interface DragState {
  slotId: number;
  dx: number;
  dy: number;
}

const getInitData = () => window.Telegram?.WebApp?.initData || '';

const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

const toAbsoluteUrl = (url: string): string => new URL(url, window.location.origin).toString();

function normalizeJobStatus(status: unknown): string {
  if (typeof status !== 'string') {
    return '';
  }
  return status.trim().toLowerCase();
}

function extractJobResultUrl(job: Record<string, unknown>): string | null {
  const candidates = [
    job.result_url,
    job.result_file_url,
    job.resultUrl,
    job.resultFileUrl,
  ];

  for (const value of candidates) {
    if (typeof value === 'string' && value.trim().length > 0) {
      return value.trim();
    }
  }

  return null;
}

function buildJobResultProxyUrl(jobId: string, initData: string): string {
  const encodedJobId = encodeURIComponent(jobId);
  const encodedInitData = encodeURIComponent(initData);
  return `/api/jobs/${encodedJobId}/result?init_data=${encodedInitData}`;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function createObjectSlot(slotId: number): ObjectSlot {
  return {
    slotId,
    label: '',
    upload: null
  };
}

function getDefaultPlacement(index: number, total: number): { x: number; y: number; scale: number } {
  const columns = Math.min(5, Math.max(1, total));
  const row = Math.floor(index / columns);
  const col = index % columns;
  const rows = Math.max(1, Math.ceil(total / columns));

  const x = (col + 0.5) / columns;
  const rowProgress = rows > 1 ? row / (rows - 1) : 0;
  const y = clamp(0.62 + rowProgress * 0.22, 0.36, 0.9);

  const scale =
    total <= 2 ? 0.34 :
    total <= 4 ? 0.28 :
    total <= 7 ? 0.24 : 0.2;

  return { x, y, scale };
}

function objectDisplayName(label: string, index: number): string {
  const trimmed = label.trim();
  return trimmed.length > 0 ? trimmed : `Предмет ${index + 1}`;
}

function suggestLabelFromFileName(fileName: string): string {
  const noExt = fileName.replace(/\.[^/.]+$/, '');
  const normalized = noExt.replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim();
  return normalized.slice(0, 80);
}

export default function App() {
  const [initData, setInitData] = useState('');
  const [mode, setMode] = useState<AppMode>('upload');
  const [error, setError] = useState('');
  const [info, setInfo] = useState('');

  const [background, setBackground] = useState<AssetInfo | null>(null);
  const [objectSlots, setObjectSlots] = useState<ObjectSlot[]>([createObjectSlot(1)]);
  const [nextSlotId, setNextSlotId] = useState(2);
  const [placements, setPlacements] = useState<Placement[]>([]);
  const [selectedSlotId, setSelectedSlotId] = useState<number | null>(null);

  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatusText, setJobStatusText] = useState('');
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [sendingToChat, setSendingToChat] = useState(false);
  const [savingToDevice, setSavingToDevice] = useState(false);

  const [viewport, setViewport] = useState({
    width: typeof window !== 'undefined' ? window.innerWidth : 390,
    height: typeof window !== 'undefined' ? window.innerHeight : 844
  });

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const imagePromiseCacheRef = useRef(new Map<string, Promise<HTMLImageElement>>());

  useEffect(() => {
    const init = getInitData();
    if (!init) {
      setError('Open this mini app inside Telegram.');
      return;
    }

    setInitData(init);
    window.Telegram?.WebApp?.ready?.();
    window.Telegram?.WebApp?.expand?.();
  }, []);

  useEffect(() => {
    const onResize = () => {
      setViewport({ width: window.innerWidth, height: window.innerHeight });
    };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  const uploadedSlots = useMemo(
    () => objectSlots.filter((slot) => slot.upload !== null),
    [objectSlots]
  );

  const canOpenComposer = useMemo(
    () => Boolean(background) && uploadedSlots.length > 0,
    [background, uploadedSlots.length]
  );

  const selectedPlacement = useMemo(
    () => placements.find((item) => item.slotId === selectedSlotId) ?? null,
    [placements, selectedSlotId]
  );

  const canvasSize = useMemo(() => {
    if (!background) {
      return { width: 320, height: 220 };
    }

    const maxWidth = Math.max(260, Math.min(560, viewport.width - 24));
    const maxHeight = Math.max(220, Math.min(540, Math.round(viewport.height * 0.54)));
    const scale = Math.min(maxWidth / background.width, maxHeight / background.height);

    return {
      width: Math.max(220, Math.round(background.width * scale)),
      height: Math.max(180, Math.round(background.height * scale))
    };
  }, [background, viewport.height, viewport.width]);

  const loadImage = useCallback((url: string): Promise<HTMLImageElement> => {
    const absoluteUrl = toAbsoluteUrl(url);
    const cached = imagePromiseCacheRef.current.get(absoluteUrl);
    if (cached) {
      return cached;
    }

    const promise = new Promise<HTMLImageElement>((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`Failed to load image: ${absoluteUrl}`));
      img.src = absoluteUrl;
    });

    imagePromiseCacheRef.current.set(absoluteUrl, promise);
    return promise;
  }, []);

  const drawPreview = useCallback(async () => {
    if (!background || !canvasRef.current) {
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }

    const bgImage = await loadImage(background.url);
    const placementEntries = await Promise.all(
      placements.map(async (item) => ({
        placement: item,
        image: await loadImage(item.url)
      }))
    );

    canvas.width = canvasSize.width;
    canvas.height = canvasSize.height;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(bgImage, 0, 0, canvas.width, canvas.height);

    const sorted = [...placementEntries].sort((a, b) => a.placement.zIndex - b.placement.zIndex);
    for (const item of sorted) {
      const p = item.placement;
      const width = canvas.width * p.scale;
      const height = width * p.aspect;
      const centerX = p.x * canvas.width;
      const centerY = p.y * canvas.height;

      ctx.save();
      ctx.translate(centerX, centerY);
      ctx.rotate((p.rotationDeg * Math.PI) / 180);
      ctx.drawImage(item.image, -width / 2, -height / 2, width, height);
      ctx.restore();

      if (selectedSlotId === p.slotId) {
        ctx.save();
        ctx.strokeStyle = '#fff2a8';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 4]);
        ctx.strokeRect(centerX - width / 2, centerY - height / 2, width, height);
        ctx.restore();
      }
    }
  }, [background, canvasSize.height, canvasSize.width, loadImage, placements, selectedSlotId]);

  useEffect(() => {
    drawPreview().catch((err: unknown) => {
      setError(err instanceof Error ? err.message : 'Failed to render preview');
    });
  }, [drawPreview]);

  const initializeComposer = useCallback(() => {
    if (!background || uploadedSlots.length < 1) {
      return;
    }

    const nextPlacements: Placement[] = uploadedSlots.map((slot, index) => {
      const cutout = slot.upload?.cutout;
      if (!cutout) {
        throw new Error(`Object slot ${slot.slotId} is missing upload`);
      }

      const defaults = getDefaultPlacement(index, uploadedSlots.length);

      return {
        slotId: slot.slotId,
        label: slot.label,
        assetId: cutout.asset_id,
        url: cutout.url,
        aspect: cutout.height / cutout.width,
        x: defaults.x,
        y: defaults.y,
        scale: defaults.scale,
        rotationDeg: 0,
        zIndex: index + 1
      };
    });

    setPlacements(nextPlacements);
    setSelectedSlotId(nextPlacements[0]?.slotId ?? null);
    setMode('compose');
    setError('');
    setInfo('Перетащите предметы на сцене и настройте масштаб/поворот.');
  }, [background, uploadedSlots]);

  const updatePlacement = useCallback((slotId: number, updater: (current: Placement) => Placement) => {
    setPlacements((current) => current.map((item) => (item.slotId === slotId ? updater(item) : item)));
  }, []);

  const handleBackgroundUpload = async (file: File | null) => {
    if (!file || !initData) {
      return;
    }

    setError('');
    setInfo('Загружаем фото интерьера...');

    try {
      const asset = await uploadBackground(initData, file);
      setBackground(asset);
      setMode('upload');
      setInfo('Фон загружен. Добавьте до 10 предметов интерьера.');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload background');
    }
  };

  const addObjectSlot = () => {
    if (objectSlots.length >= MAX_OBJECTS) {
      return;
    }
    setObjectSlots((current) => [...current, createObjectSlot(nextSlotId)]);
    setNextSlotId((current) => current + 1);
  };

  const removeObjectSlot = (slotId: number) => {
    if (objectSlots.length <= 1) {
      return;
    }

    setObjectSlots((current) => current.filter((slot) => slot.slotId !== slotId));
    setPlacements((current) => current.filter((item) => item.slotId !== slotId));

    if (selectedSlotId === slotId) {
      setSelectedSlotId(null);
    }
  };

  const updateObjectLabel = (slotId: number, label: string) => {
    setObjectSlots((current) =>
      current.map((slot) => (slot.slotId === slotId ? { ...slot, label } : slot))
    );
    setPlacements((current) =>
      current.map((placement) =>
        placement.slotId === slotId ? { ...placement, label } : placement
      )
    );
  };

  const handleObjectUpload = async (slotId: number, file: File | null) => {
    if (!file || !initData) {
      return;
    }

    const slot = objectSlots.find((item) => item.slotId === slotId);
    const label = slot?.label?.trim() ?? '';

    setError('');
    setInfo(`Загружаем ${label || 'предмет'}...`);

    try {
      const payload = await uploadObject(initData, file, label || undefined);
      setObjectSlots((current) =>
        current.map((item) =>
          item.slotId === slotId
            ? {
                ...item,
                upload: payload
              }
            : item
        )
      );
      setMode('upload');
      setInfo(`${label || 'Предмет'} готов.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload object');
    }
  };

  const handleBulkObjectUpload = async (filesList: FileList | null) => {
    if (!filesList || !initData) {
      return;
    }

    const files = Array.from(filesList).filter((file) => file.type.startsWith('image/'));
    if (files.length === 0) {
      setError('Выберите один или несколько файлов изображений.');
      return;
    }

    setError('');

    const draftSlots: ObjectSlot[] = objectSlots.map((slot) => ({ ...slot }));
    let draftNextSlotId = nextSlotId;
    const emptySlotQueue = draftSlots
      .filter((slot) => slot.upload === null)
      .map((slot) => slot.slotId);

    const assignments: Array<{ slotId: number; file: File; label: string }> = [];
    for (const file of files) {
      let slotId = emptySlotQueue.shift();

      if (slotId === undefined) {
        if (draftSlots.length >= MAX_OBJECTS) {
          break;
        }
        slotId = draftNextSlotId;
        draftNextSlotId += 1;
        draftSlots.push(createObjectSlot(slotId));
      }

      const slot = draftSlots.find((item) => item.slotId === slotId);
      if (!slot) {
        continue;
      }

      const slotLabel = slot.label.trim();
      const label = slotLabel || suggestLabelFromFileName(file.name);
      if (!slotLabel && label) {
        slot.label = label;
      }

      assignments.push({ slotId, file, label });
    }

    const skipped = files.length - assignments.length;
    if (assignments.length === 0) {
      setError(`Достигнут лимит в ${MAX_OBJECTS} предметов.`);
      return;
    }

    setObjectSlots(draftSlots);
    setNextSlotId(draftNextSlotId);

    let success = 0;
    let failed = 0;
    let lastError = '';

    for (let index = 0; index < assignments.length; index += 1) {
      const assignment = assignments[index];
      const displayName = assignment.label || `предмет ${index + 1}`;
      setInfo(`Загружаем ${index + 1}/${assignments.length}: ${displayName}...`);

      try {
        const payload = await uploadObject(initData, assignment.file, assignment.label || undefined);
        success += 1;
        setObjectSlots((current) =>
          current.map((slot) =>
            slot.slotId === assignment.slotId
              ? {
                  ...slot,
                  label: slot.label.trim() || assignment.label,
                  upload: payload
                }
              : slot
          )
        );
      } catch (err) {
        failed += 1;
        lastError = err instanceof Error ? err.message : 'Failed to upload object';
      }
    }

    setMode('upload');
    const skippedMessage = skipped > 0 ? `, пропущено: ${skipped}` : '';
    setInfo(`Загружено: ${success}, ошибок: ${failed}${skippedMessage}.`);

    if (failed > 0) {
      setError(lastError || `Не удалось загрузить ${failed} файл(ов).`);
    }
  };

  const pointerToCanvas = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return { x: 0, y: 0 };
    }

    const rect = canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((event.clientY - rect.top) / rect.height) * canvas.height;
    return { x, y };
  };

  const hitTestPlacement = (x: number, y: number): Placement | null => {
    const sorted = [...placements].sort((a, b) => b.zIndex - a.zIndex);
    for (const item of sorted) {
      const width = canvasSize.width * item.scale;
      const height = width * item.aspect;
      const centerX = item.x * canvasSize.width;
      const centerY = item.y * canvasSize.height;

      if (
        x >= centerX - width / 2 &&
        x <= centerX + width / 2 &&
        y >= centerY - height / 2 &&
        y <= centerY + height / 2
      ) {
        return item;
      }
    }

    return null;
  };

  const handlePointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (mode !== 'compose') {
      return;
    }

    const point = pointerToCanvas(event);
    const hit = hitTestPlacement(point.x, point.y);
    if (!hit) {
      setSelectedSlotId(null);
      return;
    }

    const centerX = hit.x * canvasSize.width;
    const centerY = hit.y * canvasSize.height;

    setSelectedSlotId(hit.slotId);
    dragRef.current = {
      slotId: hit.slotId,
      dx: point.x - centerX,
      dy: point.y - centerY
    };

    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const handlePointerMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!dragRef.current || mode !== 'compose') {
      return;
    }

    const point = pointerToCanvas(event);
    const drag = dragRef.current;

    updatePlacement(drag.slotId, (current) => ({
      ...current,
      x: clamp((point.x - drag.dx) / canvasSize.width, 0.04, 0.96),
      y: clamp((point.y - drag.dy) / canvasSize.height, 0.04, 0.96)
    }));
  };

  const handlePointerUp = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const renderCollageBlob = useCallback(async (): Promise<Blob> => {
    if (!background) {
      throw new Error('Background is missing');
    }

    const bgImage = await loadImage(background.url);
    const entries = await Promise.all(
      placements.map(async (item) => ({
        placement: item,
        image: await loadImage(item.url)
      }))
    );

    const longestSide = 1536;
    const scale = longestSide / Math.max(background.width, background.height);
    const targetWidth = Math.max(1, Math.round(background.width * scale));
    const targetHeight = Math.max(1, Math.round(background.height * scale));

    const canvas = document.createElement('canvas');
    canvas.width = targetWidth;
    canvas.height = targetHeight;

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Canvas context is unavailable');
    }

    ctx.drawImage(bgImage, 0, 0, targetWidth, targetHeight);
    const sorted = [...entries].sort((a, b) => a.placement.zIndex - b.placement.zIndex);

    for (const item of sorted) {
      const p = item.placement;
      const width = targetWidth * p.scale;
      const height = width * p.aspect;
      const centerX = p.x * targetWidth;
      const centerY = p.y * targetHeight;

      ctx.save();
      ctx.translate(centerX, centerY);
      ctx.rotate((p.rotationDeg * Math.PI) / 180);
      ctx.drawImage(item.image, -width / 2, -height / 2, width, height);
      ctx.restore();
    }

    const blob = await new Promise<Blob | null>((resolve) => {
      canvas.toBlob(resolve, 'image/png');
    });

    if (!blob) {
      throw new Error('Failed to render collage image');
    }

    return blob;
  }, [background, loadImage, placements]);

  const startGeneration = useCallback(async () => {
    if (!initData || !background || placements.length < 1 || placements.length > MAX_OBJECTS) {
      return;
    }

    setError('');
    setJobId(null);
    setJobStatusText('');
    setResultUrl(null);
    setInfo('Сохраняем сцену...');

    const sceneObjects: SceneObject[] = placements.map((item) => {
      const label = item.label.trim();
      return {
        asset_id: item.assetId,
        label: label.length > 0 ? label : undefined,
        x: item.x,
        y: item.y,
        scale: item.scale,
        rotation_deg: item.rotationDeg,
        z_index: item.zIndex
      };
    });

    try {
      const sceneId = await createScene(initData, background.asset_id, sceneObjects);
      setInfo('Собираем коллаж...');
      const collageBlob = await renderCollageBlob();

      setMode('generating');
      setJobStatusText('Запускаем генерацию...');
      const generated = await generateInterior(initData, sceneId, collageBlob);
      setJobId(generated.job_id);
    } catch (err) {
      setMode('compose');
      setError(err instanceof Error ? err.message : 'Failed to start generation');
    }
  }, [background, initData, placements, renderCollageBlob]);

  useEffect(() => {
    if (mode !== 'generating' || !jobId || !initData) {
      return;
    }

    let cancelled = false;

    const loop = async () => {
      while (!cancelled) {
        try {
          const job = (await getJob(initData, jobId)) as JobResponse & Record<string, unknown>;
          const status = normalizeJobStatus(job.status);
          const resolvedResultUrl = extractJobResultUrl(job);

          setJobStatusText(`Статус: ${status || 'unknown'}`);

          if ((status === 'done' || status === 'completed') && resolvedResultUrl) {
            setResultUrl(buildJobResultProxyUrl(jobId, initData));
            setMode('result');
            setInfo('Готово. Можно отправить результат в чат.');
            return;
          }

          if (status === 'done' || status === 'completed') {
            setMode('compose');
            setError('Генерация завершилась, но ссылка на результат отсутствует.');
            return;
          }

          if (status === 'failed' || status === 'moderated') {
            setMode('compose');
            setError(job.error || 'Generation failed');
            return;
          }
        } catch (err) {
          setMode('compose');
          setError(err instanceof Error ? err.message : 'Polling failed');
          return;
        }

        await sleep(2000);
      }
    };

    loop().catch((err: unknown) => {
      setMode('compose');
      setError(err instanceof Error ? err.message : 'Unexpected polling error');
    });

    return () => {
      cancelled = true;
    };
  }, [initData, jobId, mode]);

  const handleSendToChat = async () => {
    if (!initData || !jobId) {
      return;
    }

    setSendingToChat(true);
    setError('');

    try {
      await sendToChat(initData, jobId);
      setInfo('Результат отправлен в чат Telegram.');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send to chat');
    } finally {
      setSendingToChat(false);
    }
  };

  const handleSaveToDevice = async () => {
    if (!resultUrl) {
      return;
    }

    setSavingToDevice(true);
    setError('');

    const absoluteResultUrl = toAbsoluteUrl(resultUrl);
    const fallbackOpenImage = () => {
      const tgOpenLink = window.Telegram?.WebApp?.openLink;
      if (typeof tgOpenLink === 'function') {
        tgOpenLink(absoluteResultUrl);
      } else {
        window.open(absoluteResultUrl, '_blank', 'noopener,noreferrer');
      }
      setInfo('Открыли изображение. Если файл не скачался автоматически, сохраните его через меню браузера.');
    };

    try {
      const response = await fetch(absoluteResultUrl, { cache: 'no-store' });
      if (!response.ok) {
        throw new Error('Failed to load generated image');
      }

      const blob = await response.blob();
      if (blob.size < 1) {
        throw new Error('Generated image is empty');
      }

      const objectUrl = URL.createObjectURL(blob);
      try {
        const link = document.createElement('a');
        const stamp = new Date().toISOString().replace(/[:.]/g, '-');
        link.href = objectUrl;
        link.download = `interior-result-${stamp}.png`;
        link.rel = 'noopener noreferrer';
        link.style.display = 'none';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        setInfo('Сохранение запущено. Проверьте загрузки телефона.');
      } finally {
        URL.revokeObjectURL(objectUrl);
      }
    } catch (_err) {
      fallbackOpenImage();
    } finally {
      setSavingToDevice(false);
    }
  };

  const setScale = (nextScale: number) => {
    if (selectedSlotId === null) return;
    updatePlacement(selectedSlotId, (current) => ({
      ...current,
      scale: clamp(nextScale, 0.08, 0.9)
    }));
  };

  const setRotation = (nextDeg: number) => {
    if (selectedSlotId === null) return;
    updatePlacement(selectedSlotId, (current) => ({
      ...current,
      rotationDeg: clamp(nextDeg, -180, 180)
    }));
  };

  const bringForward = () => {
    if (selectedSlotId === null) return;
    const maxZ = placements.reduce((acc, item) => Math.max(acc, item.zIndex), 0);
    updatePlacement(selectedSlotId, (current) => ({ ...current, zIndex: maxZ + 1 }));
  };

  const sendBackward = () => {
    if (selectedSlotId === null) return;
    updatePlacement(selectedSlotId, (current) => ({ ...current, zIndex: Math.max(0, current.zIndex - 1) }));
  };

  return (
    <div className="app-shell">
      <header className="page-header">
        <h1>Interior Collage Studio</h1>
        <p>Загрузите интерьер и до 10 любых предметов, расставьте их и запустите AI-генерацию.</p>
      </header>

      {error ? <div className="alert error">{error}</div> : null}
      {info ? <div className="alert info">{info}</div> : null}

      <section className="panel">
        <h2>1. Загрузка</h2>
        <label className="upload-row">
          <span>Фото интерьера</span>
          <input
            type="file"
            accept="image/*"
            onChange={(event) => {
              const file = event.target.files?.[0] ?? null;
              handleBackgroundUpload(file);
              event.currentTarget.value = '';
            }}
          />
        </label>

        {background ? <p className="hint">Фон: загружен ({background.width}x{background.height})</p> : null}
        <p className="hint">Добавьте от 1 до {MAX_OBJECTS} предметов. Это могут быть любые объекты интерьера.</p>
        <label className="upload-row">
          <span>Несколько фото сразу</span>
          <input
            type="file"
            accept="image/*"
            multiple
            onChange={(event) => {
              handleBulkObjectUpload(event.target.files);
              event.currentTarget.value = '';
            }}
          />
        </label>
        <p className="hint">Можно выбрать сразу несколько фото из галереи телефона.</p>

        <div className="object-list">
          {objectSlots.map((slot, index) => (
            <div key={slot.slotId} className="object-card">
              <div className="object-card-head">
                <strong>Предмет {index + 1}</strong>
                <button
                  type="button"
                  className="chip"
                  disabled={objectSlots.length <= 1}
                  onClick={() => removeObjectSlot(slot.slotId)}
                >
                  Удалить
                </button>
              </div>

              <label className="upload-row compact">
                <span>Название (необязательно)</span>
                <input
                  type="text"
                  maxLength={80}
                  value={slot.label}
                  placeholder="Например: кресло, комод, тумба"
                  onChange={(event) => updateObjectLabel(slot.slotId, event.target.value)}
                />
              </label>

              <label className="upload-row compact">
                <span>Фото предмета</span>
                <input
                  type="file"
                  accept="image/*"
                  onChange={(event) => {
                    const file = event.target.files?.[0] ?? null;
                    handleObjectUpload(slot.slotId, file);
                    event.currentTarget.value = '';
                  }}
                />
                <small>{slot.upload ? 'Готово' : 'Не загружено'}</small>
              </label>
            </div>
          ))}
        </div>

        <div className="actions-row">
          <button type="button" disabled={objectSlots.length >= MAX_OBJECTS} onClick={addObjectSlot}>
            Добавить предмет
          </button>
          <button type="button" disabled={!canOpenComposer} onClick={initializeComposer}>
            Перейти к редактору
          </button>
        </div>
      </section>

      {(mode === 'compose' || mode === 'generating' || mode === 'result') && background ? (
        <section className="panel">
          <h2>2. Расстановка и генерация</h2>
          <div className="editor-wrap">
            <canvas
              ref={canvasRef}
              className="scene-canvas"
              width={canvasSize.width}
              height={canvasSize.height}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerCancel={handlePointerUp}
            />

            <div className="controls">
              <p className="hint">Тапните по предмету и двигайте пальцем по сцене.</p>

              <div className="chip-row">
                {placements.map((item, index) => (
                  <button
                    type="button"
                    key={item.slotId}
                    className={selectedSlotId === item.slotId ? 'chip active' : 'chip'}
                    onClick={() => setSelectedSlotId(item.slotId)}
                  >
                    {objectDisplayName(item.label, index)}
                  </button>
                ))}
              </div>

              {selectedPlacement ? (
                <div className="sliders">
                  <label>
                    <span>Масштаб: {selectedPlacement.scale.toFixed(2)}</span>
                    <input
                      type="range"
                      min={0.08}
                      max={0.9}
                      step={0.01}
                      value={selectedPlacement.scale}
                      onChange={(event) => setScale(Number(event.target.value))}
                    />
                  </label>

                  <label>
                    <span>Поворот: {Math.round(selectedPlacement.rotationDeg)}°</span>
                    <input
                      type="range"
                      min={-180}
                      max={180}
                      step={1}
                      value={selectedPlacement.rotationDeg}
                      onChange={(event) => setRotation(Number(event.target.value))}
                    />
                  </label>

                  <div className="actions-row">
                    <button type="button" onClick={bringForward}>
                      Вперед
                    </button>
                    <button type="button" onClick={sendBackward}>
                      Назад
                    </button>
                  </div>
                </div>
              ) : null}

              <div className="actions-row">
                <button type="button" disabled={mode === 'generating'} onClick={startGeneration}>
                  {mode === 'generating' ? 'Генерация...' : 'Generate'}
                </button>
              </div>

              {mode === 'generating' ? <p className="hint">{jobStatusText || 'Ожидаем ответ модели...'}</p> : null}
            </div>
          </div>
        </section>
      ) : null}

      {mode === 'result' && resultUrl ? (
        <section className="panel">
          <h2>3. Результат</h2>
          <img
            src={toAbsoluteUrl(resultUrl)}
            className="result-image"
            alt="Generated interior"
            referrerPolicy="no-referrer"
            onError={() => {
              setError('Не удалось загрузить изображение результата. Попробуйте Regenerate.');
            }}
          />
          <div className="actions-row">
            <button type="button" onClick={startGeneration}>
              Regenerate
            </button>
            <button type="button" onClick={handleSaveToDevice} disabled={savingToDevice}>
              {savingToDevice ? 'Сохраняем...' : 'Сохранить на телефон'}
            </button>
            <button type="button" onClick={handleSendToChat} disabled={sendingToChat}>
              {sendingToChat ? 'Отправляем...' : 'Send to chat'}
            </button>
          </div>
        </section>
      ) : null}
    </div>
  );
}
