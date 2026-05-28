"""
Image model sandbox — compare OpenRouter image generation models for Time Machine.

Tests two scenarios:
  A) text-to-image  — generate a historical scene from a prompt
  B) img2img        — transform a reference photo to a historical period

Results saved to sandbox_results/<model_slug>_{a,b}.jpg

Usage:
  poetry run python sandbox.py               # all models
  poetry run python sandbox.py flux          # models matching "flux"
  poetry run python sandbox.py gemini gpt    # multiple filters

OpenRouter docs: images live in message.images[], NOT in message.content parts.
All image models use /v1/chat/completions — no separate /v1/images/generations.
"""

import asyncio
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

REFERENCE_IMAGE = Path("samples/PXL_20240509_171740643.jpg")

PROMPT_TEXT_TO_IMAGE = (
    "Frankfurt Römerberg square in the year 1600. "
    "Medieval half-timbered houses, cobblestone square, people in period clothing. "
    "Golden hour light. Photorealistic, ultra-detailed, no anachronisms."
)

PROMPT_IMG2IMG = (
    "Transform this modern street photo into the same location as it looked in the year 1600. "
    "Keep the camera angle and composition identical. "
    "Replace modern elements with medieval half-timbered houses, cobblestones, period clothing. "
    "Photorealistic, historically accurate, no anachronisms."
)

OUTPUT_DIR = Path("sandbox_results")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ── Model registry ────────────────────────────────────────────────────────────
# (display_name, model_id, text+image or image-only)

MODELS = [
    # Gemini (currently in production)
    ("Gemini 2.5 Flash",          "google/gemini-2.5-flash-image",           "text+image"),
    ("Gemini 3.1 Flash",          "google/gemini-3.1-flash-image-preview",   "text+image"),
    ("Gemini 3 Pro",              "google/gemini-3-pro-image-preview",       "text+image"),
    # OpenAI GPT Image
    ("GPT-5 Image Mini",          "openai/gpt-5-image-mini",                 "text+image"),
    ("GPT-5 Image",               "openai/gpt-5-image",                      "text+image"),
    ("GPT-5.4 Image 2",           "openai/gpt-5.4-image-2",                  "text+image"),
    # FLUX — image-only
    ("FLUX.2 Klein 4B",           "black-forest-labs/flux.2-klein-4b",       "image"),
    ("FLUX.2 Pro",                "black-forest-labs/flux.2-pro",            "image"),
    ("FLUX.2 Max",                "black-forest-labs/flux.2-max",            "image"),
    # Others
    ("Seedream 4.5",              "bytedance-seed/seedream-4.5",             "image"),
    ("Grok Imagine",              "x-ai/grok-imagine-image-quality",         "image"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def _slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")


def _load_reference() -> str:
    return base64.b64encode(REFERENCE_IMAGE.read_bytes()).decode()


def _save(name: str, b64: str, mime: str) -> Path:
    ext = mime.split("/")[-1].replace("jpeg", "jpg")
    path = OUTPUT_DIR / f"{name}.{ext}"
    path.write_bytes(base64.b64decode(b64))
    return path


def _extract_image(resp_json: dict) -> tuple[str | None, str]:
    """
    OpenRouter image response format:
      choices[0].message.images[0].image_url.url  → "data:<mime>;base64,<data>"
    Fallback: also check content parts (older/alt format).
    """
    try:
        msg = resp_json["choices"][0]["message"]

        # Primary path: message.images[]
        images = msg.get("images") or []
        for img in images:
            url = (img.get("image_url") or {}).get("url", "")
            if url.startswith("data:"):
                m = re.match(r"data:([^;]+);base64,(.+)", url, re.DOTALL)
                if m:
                    return m.group(2).strip(), m.group(1)

        # Fallback: content as list of parts (some models)
        content = msg.get("content")
        parts = content if isinstance(content, list) else []
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if url.startswith("data:"):
                    m = re.match(r"data:([^;]+);base64,(.+)", url, re.DOTALL)
                    if m:
                        return m.group(2).strip(), m.group(1)

        # Log what we actually got
        text = msg.get("content") if isinstance(msg.get("content"), str) else ""
        if text:
            print(f"    ⚠  text-only: {text[:120]}")
        else:
            print(f"    ⚠  no image found. keys in message: {list(msg.keys())}")
            print(f"       raw snippet: {json.dumps(resp_json)[:300]}")

    except Exception as e:
        print(f"    ✗  extraction error: {e}")

    return None, ""


# ── API call ──────────────────────────────────────────────────────────────────


async def call_model(
    client: httpx.AsyncClient,
    model_id: str,
    modality_mode: str,  # "text+image" | "image"
    prompt: str,
    ref_b64: str | None,
) -> tuple[str | None, str]:
    modalities = ["image", "text"] if modality_mode == "text+image" else ["image"]

    if ref_b64:
        content: list | str = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ref_b64}"}},
        ]
    else:
        content = prompt

    payload = {
        "model": model_id,
        "modalities": modalities,
        "messages": [{"role": "user", "content": content}],
    }
    resp = await client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json=payload,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    return _extract_image(resp.json())


# ── Runner ────────────────────────────────────────────────────────────────────


async def test_one(
    client: httpx.AsyncClient,
    model_id: str,
    modality_mode: str,
    ref_b64: str,
    label: str,
    prompt: str,
    is_img2img: bool,
    slug: str,
) -> None:
    suffix = "b_img2img" if is_img2img else "a_txt2img"
    t0 = time.perf_counter()
    try:
        b64, mime = await call_model(
            client, model_id, modality_mode, prompt,
            ref_b64 if is_img2img else None,
        )
        elapsed = time.perf_counter() - t0
        if b64:
            path = _save(f"{slug}__{suffix}", b64, mime)
            print(f"  ✓  {label:12s}  {elapsed:.1f}s  → {path.name}")
        else:
            print(f"  ✗  {label:12s}  {elapsed:.1f}s  no image returned")
    except httpx.HTTPStatusError as e:
        elapsed = time.perf_counter() - t0
        print(f"  ✗  {label:12s}  {elapsed:.1f}s  HTTP {e.response.status_code}: {e.response.text[:150]}")
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  ✗  {label:12s}  {elapsed:.1f}s  {e}")


async def main(filters: list[str]) -> None:
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    ref_b64 = _load_reference()

    models = MODELS
    if filters:
        models = [
            m for m in MODELS
            if any(f.lower() in m[1].lower() or f.lower() in m[0].lower() for f in filters)
        ]

    print(f"\nTesting {len(models)} model(s)  |  output → {OUTPUT_DIR}/\n")
    print(f"Reference image: {REFERENCE_IMAGE} ({REFERENCE_IMAGE.stat().st_size // 1024} KB)\n")

    async with httpx.AsyncClient() as client:
        for name, model_id, modality_mode in models:
            print(f"── {name}  ({model_id})")
            slug = _slug(model_id)
            await test_one(client, model_id, modality_mode, ref_b64, "txt2img", PROMPT_TEXT_TO_IMAGE, False, slug)
            await test_one(client, model_id, modality_mode, ref_b64, "img2img", PROMPT_IMG2IMG, True, slug)
            print()

    results = list(OUTPUT_DIR.glob("*.jp*")) + list(OUTPUT_DIR.glob("*.png"))
    print(f"Saved {len(results)} image(s) to {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
