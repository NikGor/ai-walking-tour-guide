import asyncio
import logging
from dataclasses import dataclass, field

import aiohttp

logger = logging.getLogger(__name__)

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_HEADERS = {"User-Agent": "SolarisPliny/0.1 (walking-tour-guide)"}


@dataclass
class LocationContext:
    """Rich location data assembled from OSM (Nominatim + Overpass) and Wikipedia."""

    name: str
    wikipedia: str | None = None  # e.g. "ru:Медный всадник"
    wikidata: str | None = None  # e.g. "Q175111"
    architect: str | None = None
    start_date: str | None = None
    historic: str | None = None  # monument / castle / memorial / church …
    nearby: list[dict] = field(default_factory=list)  # [{name, type, wikipedia, …}]
    wikipedia_summary: str | None = None  # first paragraph from Wikipedia REST API


async def get_location_context(lat: float, lon: float) -> LocationContext:
    """
    Orchestrate Nominatim + Overpass in parallel, then fetch Wikipedia if a tag is found.
    Falls back gracefully: a failed sub-request produces empty data, not an error.
    """
    nom_task = asyncio.create_task(_fetch_nominatim(lat, lon))
    ovp_task = asyncio.create_task(_fetch_overpass(lat, lon))

    nom_result, ovp_result = await asyncio.gather(nom_task, ovp_task, return_exceptions=True)

    if isinstance(nom_result, Exception):
        logger.warning("geo_nom: failed — %s", nom_result)
        nom_result = {}
    if isinstance(ovp_result, Exception):
        logger.warning("geo_ovp: failed — %s", ovp_result)
        ovp_result = []

    ctx = _build_context(lat, lon, nom_result, ovp_result)  # type: ignore[arg-type]

    if ctx.wikipedia:
        try:
            ctx.wikipedia_summary = await _fetch_wikipedia(ctx.wikipedia)
        except Exception as e:
            logger.warning("geo_wiki: failed — %s", e)

    return ctx


async def reverse_geocode(lat: float, lon: float) -> str:
    """Thin backward-compat wrapper — returns just the place name string."""
    ctx = await get_location_context(lat, lon)
    return ctx.name


# ── Internal fetchers ──────────────────────────────────────────────────────────


async def _fetch_nominatim(lat: float, lon: float) -> dict:
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        "zoom": 17,
        "addressdetails": 1,
        "extratags": 1,
        "namedetails": 1,
    }
    async with aiohttp.ClientSession(headers=_HEADERS) as session:
        async with session.get(
            _NOMINATIM_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            resp.raise_for_status()
            return await resp.json()


async def _fetch_overpass(lat: float, lon: float, radius: int = 150) -> list[dict]:
    query = f"""
[out:json][timeout:6];
(
  node["historic"](around:{radius},{lat},{lon});
  way["historic"](around:{radius},{lat},{lon});
  node["tourism"~"^(museum|attraction|artwork|monument|gallery)$"](around:{radius},{lat},{lon});
  way["tourism"~"^(museum|attraction|artwork|monument|gallery)$"](around:{radius},{lat},{lon});
);
out tags center 15;
"""
    async with aiohttp.ClientSession(headers=_HEADERS) as session:
        async with session.post(
            _OVERPASS_URL,
            data={"data": query},
            timeout=aiohttp.ClientTimeout(total=8),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

    pois = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name") or tags.get("name:en") or tags.get("name:ru")
        if not name:
            continue
        poi: dict = {"name": name}
        for key in ("historic", "tourism", "wikipedia", "wikidata", "architect", "start_date", "description"):
            val = tags.get(key)
            if val:
                poi[key] = val
        pois.append(poi)

    return pois


async def _fetch_wikipedia(tag: str) -> str | None:
    """
    Fetch the intro paragraph for a Wikipedia tag like 'ru:Медный всадник'.
    Uses the Wikipedia REST summary endpoint — returns a short extract, not the full article.
    """
    if ":" not in tag:
        return None
    lang, title = tag.split(":", 1)
    title_encoded = title.replace(" ", "_")
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title_encoded}"
    async with aiohttp.ClientSession(headers=_HEADERS) as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
    return data.get("extract")


# ── Context builder ────────────────────────────────────────────────────────────


def _build_context(lat: float, lon: float, nom: dict, nearby: list[dict]) -> LocationContext:
    address = nom.get("address", {})
    extratags = nom.get("extratags", {})

    parts: list[str] = []
    for key in (
        "tourism",
        "amenity",
        "building",
        "road",
        "pedestrian",
        "suburb",
        "city_district",
        "city",
        "town",
        "village",
        "country",
    ):
        val = address.get(key)
        if val and val not in parts:
            parts.append(val)
        if len(parts) >= 4:
            break

    display = nom.get("display_name", "")
    name = ", ".join(parts) if parts else (display.split(",")[0] if display else f"{lat:.4f}, {lon:.4f}")

    logger.info("geo_001: \033[32m%s\033[0m  (%.4f, %.4f)", name, lat, lon)
    if nearby:
        logger.info("geo_ovp: \033[32m%d POIs\033[0m nearby", len(nearby))

    return LocationContext(
        name=name,
        wikipedia=extratags.get("wikipedia"),
        wikidata=extratags.get("wikidata"),
        architect=extratags.get("architect"),
        start_date=extratags.get("start_date"),
        historic=extratags.get("historic"),
        nearby=nearby,
    )
