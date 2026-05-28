import asyncio
import logging
import re
import urllib.parse
from dataclasses import dataclass, field

import aiohttp

logger = logging.getLogger(__name__)

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_HEADERS = {"User-Agent": "SolarisPliny/1.0 (github.com/NikGor/ai-walking-tour-guide)"}


@dataclass
class LocationContext:
    """Rich location data assembled from OSM (Nominatim + Overpass) and Wikipedia."""

    name: str
    wikipedia: str | None = None  # e.g. "ru:Медный всадник"
    wikidata: str | None = None  # e.g. "Q175111"
    architect: str | None = None
    start_date: str | None = None
    historic: str | None = None  # monument / castle / memorial / church …
    description: str | None = None  # OSM description tag
    nearby: list[dict] = field(default_factory=list)  # [{name, type, wikipedia, …}]
    wikipedia_summary: str | None = None  # first paragraph from Wikipedia REST API
    wikipedia_image_url: str | None = None  # thumbnail from Wikipedia REST API
    commons_image_url: str | None = None  # archival photo from Wikimedia Commons


async def get_location_context(lat: float, lon: float) -> LocationContext:
    """
    Orchestrate Nominatim + Overpass in parallel, then fetch Wikipedia if a tag is found.
    Commons archival photo is fetched in parallel with Wikipedia.
    Falls back gracefully: a failed sub-request produces empty data, not an error.
    """
    nom_task = asyncio.create_task(_fetch_nominatim(lat, lon))
    ovp_task = asyncio.create_task(_fetch_overpass(lat, lon))
    commons_task = asyncio.create_task(_fetch_commons_image(lat, lon))

    nom_result, ovp_result = await asyncio.gather(nom_task, ovp_task, return_exceptions=True)

    if isinstance(nom_result, Exception):
        logger.warning("geo_nom: failed — %s", nom_result)
        nom_result = {}
    if isinstance(ovp_result, Exception):
        logger.warning("geo_ovp: failed — %s", ovp_result)
        ovp_result = []

    ctx = _build_context(lat, lon, nom_result, ovp_result)  # type: ignore[arg-type]

    # Wikipedia tag: prefer main object, fall back to first nearby POI with one
    wiki_tag = ctx.wikipedia or next((p["wikipedia"] for p in ctx.nearby if p.get("wikipedia")), None)
    # Wikidata ID: prefer main object, fall back to nearby POIs
    wikidata_id = ctx.wikidata or next(
        (p["wikidata"] for p in ctx.nearby if p.get("wikidata") and not p.get("wikipedia")), None
    )

    if wiki_tag:
        try:
            ctx.wikipedia_summary, ctx.wikipedia_image_url = await _fetch_wikipedia(wiki_tag)
            if not ctx.wikipedia:
                ctx.wikipedia = wiki_tag
        except Exception as e:
            logger.warning("geo_wiki: failed — %s", e)
    elif wikidata_id:
        try:
            wiki_tag_from_wd = await _wikidata_to_wikipedia_tag(wikidata_id)
            if wiki_tag_from_wd:
                ctx.wikipedia_summary, ctx.wikipedia_image_url = await _fetch_wikipedia(wiki_tag_from_wd)
                ctx.wikipedia = wiki_tag_from_wd
        except Exception as e:
            logger.warning("geo_wikidata: failed for %s — %s", wikidata_id, e)

    try:
        ctx.commons_image_url = await commons_task
    except Exception as e:
        logger.warning("geo_commons: task failed — %s", e)

    return ctx


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
[out:json][timeout:8];
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


async def _wikidata_to_wikipedia_tag(qid: str) -> str | None:
    """Resolve a Wikidata Q-ID to a 'lang:Title' Wikipedia tag (en preferred, then de)."""
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    async with aiohttp.ClientSession(headers=_HEADERS) as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
    entity = next(iter(data.get("entities", {}).values()), {})
    sitelinks = entity.get("sitelinks", {})
    for lang in ("en", "de", "ru", "fr"):
        site = sitelinks.get(f"{lang}wiki")
        if site:
            return f"{lang}:{site['title']}"
    return None


async def _fetch_wikipedia(tag: str) -> tuple[str | None, str | None]:
    """Fetch intro paragraph and thumbnail URL for a Wikipedia tag like 'ru:Медный всадник'."""
    if ":" not in tag:
        return None, None
    lang, title = tag.split(":", 1)
    title_encoded = urllib.parse.quote(title.replace(" ", "_"), safe="/:_-.()")
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title_encoded}"
    async with aiohttp.ClientSession(headers=_HEADERS) as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=6)) as resp:
            if resp.status != 200:
                logger.debug("geo_wiki: %s → HTTP %d", url, resp.status)
                return None, None
            data = await resp.json()
    extract = data.get("extract")
    image_url = (data.get("originalimage") or data.get("thumbnail") or {}).get("source")
    if image_url:
        logger.info("geo_wiki: \033[32m%s\033[0m → image found", tag)
    return extract, image_url


async def _fetch_commons_image(lat: float, lon: float) -> str | None:
    """Search Wikimedia Commons for an archival (pre-1960) photo near these coordinates."""
    params = {
        "action": "query",
        "generator": "geosearch",
        "ggscoord": f"{lat}|{lon}",
        "ggsradius": "500",
        "ggslimit": "20",
        "prop": "imageinfo",
        "iiprop": "url|thumburl|extmetadata",
        "iiurlwidth": "800",
        "iiextmetadatafilter": "DateTimeOriginal|DateTime|Categories",
        "format": "json",
        "formatversion": "2",
    }
    url = "https://commons.wikimedia.org/w/api.php"
    try:
        async with aiohttp.ClientSession(headers=_HEADERS) as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
    except Exception as e:
        logger.warning("geo_commons: request failed — %s", e)
        return None

    pages = (data.get("query") or {}).get("pages") or []
    for page in pages:
        title = page.get("title", "")
        if not re.search(r"\.(jpe?g|png)$", title, re.IGNORECASE):
            continue
        imageinfo_list = page.get("imageinfo") or []
        if not imageinfo_list:
            continue
        info = imageinfo_list[0]
        ext = info.get("extmetadata") or {}

        orig_date = (ext.get("DateTimeOriginal") or ext.get("DateTime") or {}).get("value", "")
        if orig_date:
            m = re.match(r"(\d{4})", orig_date.strip())
            if m and int(m.group(1)) < 1960:
                thumb = info.get("thumburl") or info.get("url")
                if thumb:
                    logger.info("geo_commons: \033[32marchival photo\033[0m  year=%s", m.group(1))
                    return thumb

        year_in_title = re.search(r"\b(18\d\d|19[0-5]\d)\b", title)
        if year_in_title:
            thumb = info.get("thumburl") or info.get("url")
            if thumb:
                logger.info(
                    "geo_commons: \033[32marchival photo\033[0m  year=%s (title)", year_in_title.group(1)
                )
                return thumb

    return None


# ── Context builder ────────────────────────────────────────────────────────────


def _build_context(lat: float, lon: float, nom: dict, nearby: list[dict]) -> LocationContext:
    address = nom.get("address") or {}
    extratags = nom.get("extratags") or {}

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
        description=extratags.get("description"),
        nearby=nearby,
    )
