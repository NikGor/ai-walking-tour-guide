import logging

import aiohttp

logger = logging.getLogger(__name__)

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
_HEADERS = {"User-Agent": "SolarisPliny/0.1 (walking-tour-guide)"}


async def reverse_geocode(lat: float, lon: float) -> str:
    """Return a human-readable location name for given coordinates via Nominatim."""
    params = {"lat": lat, "lon": lon, "format": "json", "zoom": 17, "addressdetails": 1}
    try:
        async with aiohttp.ClientSession(headers=_HEADERS) as session:
            async with session.get(
                _NOMINATIM_URL, params=params, timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:  # noqa: E501
                if resp.status != 200:
                    logger.warning("geo_001: Nominatim returned %d", resp.status)
                    return f"{lat:.4f}, {lon:.4f}"
                data = await resp.json()

        display = data.get("display_name", "")
        address = data.get("address", {})

        # Build a concise name: POI / road, district, city, country
        parts = []
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
        ):  # noqa: E501
            val = address.get(key)
            if val and val not in parts:
                parts.append(val)
            if len(parts) >= 4:
                break

        name = ", ".join(parts) if parts else display.split(",")[0]
        logger.info("geo_001: \033[32m%s\033[0m  (%.4f, %.4f)", name, lat, lon)
        return name

    except Exception as e:
        logger.warning("geo_001: geocoding failed — %s", e)
        return f"{lat:.4f}, {lon:.4f}"
