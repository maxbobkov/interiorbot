from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageOps


def _image_from_bytes(data: bytes) -> Image.Image:
    image = Image.open(BytesIO(data))
    image = ImageOps.exif_transpose(image)
    image.load()
    return image


def _to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format='PNG', optimize=True)
    return buffer.getvalue()


def prepare_background(data: bytes, *, max_side: int = 2048) -> tuple[bytes, int, int]:
    image = _image_from_bytes(data).convert('RGB')
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    out = _to_png_bytes(image)
    return out, image.width, image.height


def remove_background(data: bytes, *, max_side: int = 2048) -> tuple[bytes, int, int]:
    image = _image_from_bytes(data).convert('RGBA')
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    pixels = image.load()
    width, height = image.size

    # Corner color sampling works for most user photos with plain walls/backgrounds.
    samples: list[tuple[int, int, int]] = []
    max_offset_x = min(3, width - 1)
    max_offset_y = min(3, height - 1)
    for ox in range(max_offset_x + 1):
        for oy in range(max_offset_y + 1):
            samples.append(pixels[ox, oy][:3])
            samples.append(pixels[width - 1 - ox, oy][:3])
            samples.append(pixels[ox, height - 1 - oy][:3])
            samples.append(pixels[width - 1 - ox, height - 1 - oy][:3])

    avg_r = sum(v[0] for v in samples) / len(samples)
    avg_g = sum(v[1] for v in samples) / len(samples)
    avg_b = sum(v[2] for v in samples) / len(samples)

    threshold = 58
    soft_threshold = 84
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            dist = abs(r - avg_r) + abs(g - avg_g) + abs(b - avg_b)
            if dist <= threshold:
                pixels[x, y] = (r, g, b, 0)
            elif dist < soft_threshold:
                # Soft edge to reduce visible halo in the final render.
                alpha = int(((dist - threshold) / (soft_threshold - threshold)) * a)
                pixels[x, y] = (r, g, b, max(0, min(255, alpha)))

    bbox = image.getbbox()
    if bbox:
        pad = 12
        left = max(0, bbox[0] - pad)
        upper = max(0, bbox[1] - pad)
        right = min(width, bbox[2] + pad)
        lower = min(height, bbox[3] + pad)
        image = image.crop((left, upper, right, lower))

    out = _to_png_bytes(image)
    return out, image.width, image.height


def normalize_collage(data: bytes, *, longest_side: int = 1536) -> tuple[bytes, int, int]:
    image = _image_from_bytes(data).convert('RGB')
    w, h = image.size
    max_side = max(w, h)
    if max_side > longest_side:
        scale = longest_side / max_side
        resized = (
            max(1, int(round(w * scale))),
            max(1, int(round(h * scale))),
        )
        image = image.resize(resized, Image.Resampling.LANCZOS)

    out = _to_png_bytes(image)
    return out, image.width, image.height
