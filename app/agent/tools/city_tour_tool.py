"""City walking tour planner.

Pipeline:
  1. Geocode each POI via Nominatim (1 req/s rate limit respected)
  2. Build pairwise walking-distance matrix (haversine)
  3. TSP: nearest-neighbour heuristic + 2-opt improvement
  4. Walking route polyline via OSRM public foot instance
  5. Static map PNG (staticmap lib + OSM tiles)
"""

import asyncio
import io
import logging
import math
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_OSRM_FOOT_URL = "https://routing.openstreetmap.de/routed-foot/route/v1/foot/{coords}"
_WALKING_SPEED_KMH = 4.0
_VISIT_MINUTES = 45  # default time at each stop


# ── Geocoding ──────────────────────────────────────────────────────────────────


# Viewbox half-size around city centre (degrees).
# ±0.25° lon ≈ ±18 km, ±0.18° lat ≈ ±20 km — covers even large metro areas
# while preventing cross-city false matches.
_VBOX_LON = 0.25
_VBOX_LAT = 0.18


async def _nominatim_search(
    client: httpx.AsyncClient,
    query: str,
    city_center: tuple[float, float] | None = None,
    bounded: bool = False,
) -> dict | None:
    """Single Nominatim search, returns first hit or None.

    If city_center is given the viewbox biases results to that area.
    bounded=True additionally restricts results to the viewbox only.
    """
    params: dict = {"q": query, "format": "json", "limit": 1, "addressdetails": 0}
    if city_center:
        clat, clon = city_center
        params["viewbox"] = f"{clon - _VBOX_LON},{clat - _VBOX_LAT},{clon + _VBOX_LON},{clat + _VBOX_LAT}"
        if bounded:
            params["bounded"] = "1"
    try:
        resp = await client.get(
            _NOMINATIM_URL,
            params=params,
            headers={"User-Agent": "SolarisPliny/1.0 (walking-tour-guide)"},
            timeout=10.0,
        )
        data = resp.json()
        if data:
            return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"])}
    except Exception as e:
        logger.warning("tour_geo_warn: query=%r  error=%s", query, e)
    return None


async def _geocode_one(
    client: httpx.AsyncClient,
    name: str,
    city: str,
    city_center: tuple[float, float] | None,
) -> dict | None:
    """Two-stage geocoding, both stages bounded to the city area.

    Stage 1: '{name}, {city}'           — precise compound query
    Stage 2: '{name}' + viewbox bounded — name-only but geographically constrained
    """
    # Stage 1: compound query, hard-bounded to city viewbox
    hit = await _nominatim_search(client, f"{name}, {city}", city_center=city_center, bounded=True)
    if hit:
        return {"name": name, **hit}

    # Stage 2: name-only, also hard-bounded (different tokenisation may help)
    await asyncio.sleep(1.1)
    hit = await _nominatim_search(client, name, city_center=city_center, bounded=True)
    if hit:
        logger.info("tour_geo_fallback: %r found via name-only bounded search", name)
        return {"name": name, **hit}

    return None


async def _geocode_pois(city: str, poi_names: list[str]) -> list[dict]:
    """Geocode POIs sequentially — Nominatim policy: ≤1 req/s.

    First resolves the city centre, then uses that as a viewbox anchor
    for all POI searches to prevent false cross-country matches.
    """
    results: list[dict] = []
    async with httpx.AsyncClient() as client:
        # Resolve city centre first — used as viewbox anchor for all POI searches
        city_center: tuple[float, float] | None = None
        city_hit = await _nominatim_search(client, city)
        if city_hit:
            city_center = (city_hit["lat"], city_hit["lon"])
            logger.info("tour_city_center: %s → %.4f, %.4f", city, city_center[0], city_center[1])
        else:
            logger.warning("tour_city_center: could not resolve %r — searches will be unbound", city)
        await asyncio.sleep(1.1)

        for i, name in enumerate(poi_names):
            if i > 0:
                await asyncio.sleep(1.1)
            result = await _geocode_one(client, name, city, city_center)
            if result:
                results.append(result)
                logger.info(
                    "tour_geo_%02d: ✓ %-40s → %.4f, %.4f",
                    i + 1,
                    name,
                    result["lat"],
                    result["lon"],
                )
            else:
                logger.warning("tour_geo_%02d: ✗ %s — skipped", i + 1, name)
    return results


# ── Distance & TSP ─────────────────────────────────────────────────────────────


_EARTH_RADIUS_KM = 6371.0


def _haversine_km(a: dict, b: dict) -> float:
    lat1, lon1 = math.radians(a["lat"]), math.radians(a["lon"])
    lat2, lon2 = math.radians(b["lat"]), math.radians(b["lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def _distance_matrix(stops: list[dict]) -> list[list[float]]:
    n = len(stops)
    return [[_haversine_km(stops[i], stops[j]) for j in range(n)] for i in range(n)]


def _tour_cost(tour: list[int], matrix: list[list[float]]) -> float:
    return sum(matrix[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))


def _nn_tour(matrix: list[list[float]]) -> list[int]:
    """Nearest-neighbour TSP heuristic starting from node 0."""
    n = len(matrix)
    visited = [False] * n
    tour = [0]
    visited[0] = True
    for _ in range(n - 1):
        last = tour[-1]
        nearest = min((j for j in range(n) if not visited[j]), key=lambda j: matrix[last][j])
        tour.append(nearest)
        visited[nearest] = True
    return tour


def _two_opt(tour: list[int], matrix: list[list[float]]) -> list[int]:
    """Improve tour with 2-opt swaps until no improvement found."""
    best = tour[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                if _tour_cost(candidate, matrix) < _tour_cost(best, matrix) - 1e-9:
                    best = candidate
                    improved = True
    return best


# ── OSRM routing ───────────────────────────────────────────────────────────────


async def _osrm_route(stops: list[dict]) -> list[tuple[float, float]] | None:
    """Fetch walking polyline from OSRM public foot instance. Returns [(lon, lat), …]."""
    coord_str = ";".join(f"{s['lon']},{s['lat']}" for s in stops)
    url = _OSRM_FOOT_URL.format(coords=coord_str)
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params={"overview": "full", "geometries": "geojson"}, timeout=15.0)
            data = resp.json()
            if data.get("code") == "Ok" and data.get("routes"):
                return [(c[0], c[1]) for c in data["routes"][0]["geometry"]["coordinates"]]
    except Exception as e:
        logger.warning("tour_osrm_warn: routing failed: %s", e)
    return None


# ── Map generation ─────────────────────────────────────────────────────────────


def _generate_map(stops: list[dict], route: list[tuple[float, float]] | None) -> bytes | None:
    """Render static map PNG: OSM tiles + route line + numbered stop markers."""
    try:
        from staticmap import CircleMarker, Line, StaticMap

        m = StaticMap(800, 550, url_template="https://tile.openstreetmap.org/{z}/{x}/{y}.png")

        if route:
            m.add_line(Line(route, "#4A90D9", 4))

        for stop in stops:
            coord = (stop["lon"], stop["lat"])
            m.add_marker(CircleMarker(coord, "white", 20))
            m.add_marker(CircleMarker(coord, "#E74C3C", 13))

        image = m.render()

        # Draw stop numbers on top using PIL
        try:
            from PIL import ImageDraw, ImageFont

            draw = ImageDraw.Draw(image)
            # Recompute pixel positions from the rendered map
            zoom = m._zoom  # type: ignore[attr-defined]

            def _ll_to_px(lat: float, lon: float) -> tuple[int, int]:
                # Slippy-map tile math
                lat_r = math.radians(lat)
                n = 2**zoom
                x_tile = (lon + 180) / 360 * n
                y_tile = (1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n
                # Map origin in tile coords
                x_center = (m._x_center - m.width / 2) / 256  # type: ignore[attr-defined]
                y_center = (m._y_center - m.height / 2) / 256  # type: ignore[attr-defined]
                px = int((x_tile - x_center) * 256)
                py = int((y_tile - y_center) * 256)
                return px, py

            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
            except Exception:
                font = ImageFont.load_default()

            for i, stop in enumerate(stops, start=1):
                px, py = _ll_to_px(stop["lat"], stop["lon"])
                label = str(i)
                bbox = draw.textbbox((0, 0), label, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                draw.text((px - tw // 2, py - th // 2), label, fill="white", font=font)
        except Exception as e:
            logger.debug("tour_map_numbers_skip: %s", e)

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        logger.info("tour_map_ok: %d bytes", buf.tell())
        return buf.getvalue()

    except Exception as e:
        logger.error("tour_map_error: %s", e, exc_info=True)
        return None


# ── Itinerary text ─────────────────────────────────────────────────────────────


def _build_itinerary(stops: list[dict], leg_km: list[float], start_time: str) -> str:
    h, m = map(int, start_time.split(":"))
    minutes = h * 60 + m
    lines: list[str] = []

    for i, stop in enumerate(stops):
        ts = f"{minutes // 60:02d}:{minutes % 60:02d}"
        lines.append(f"{i + 1}. {stop['name']} — {ts}")
        minutes += _VISIT_MINUTES
        if i < len(leg_km):
            walk_min = max(1, int(leg_km[i] / _WALKING_SPEED_KMH * 60))
            lines.append(f"   🚶 {leg_km[i]:.1f} km · {walk_min} мин")
            minutes += walk_min

    end_ts = f"{minutes // 60:02d}:{minutes % 60:02d}"
    total_km = sum(leg_km)
    lines.append(f"\n📊 {len(stops)} остановок · {total_km:.1f} km · финиш ~{end_ts}")
    return "\n".join(lines)


# ── Public entry point ─────────────────────────────────────────────────────────


async def city_tour_tool(
    city: str,
    poi_names: list[str],
    start_time: str = "10:00",
) -> dict[str, Any]:
    logger.info("\033[36mTOUR ›\033[0m city=%r  pois=%d  start=%s", city, len(poi_names), start_time)

    # 1. Geocode
    stops = await _geocode_pois(city, poi_names)
    if len(stops) < 2:
        return {
            "success": False,
            "message": f"Geocoded only {len(stops)}/{len(poi_names)} POIs — need at least 2. "
            "Try more specific names or add the country.",
        }

    skipped = [n for n in poi_names if not any(s["name"] == n for s in stops)]
    logger.info("\033[36mTOUR ›\033[0m geocoded %d stops  skipped=%s", len(stops), skipped)

    # 2. Distance matrix
    matrix = _distance_matrix(stops)

    # 3. TSP
    order = _nn_tour(matrix)
    if len(stops) <= 12:
        order = _two_opt(order, matrix)
    ordered = [stops[i] for i in order]

    # 4. OSRM walking route
    route = await _osrm_route(ordered)
    logger.info("\033[36mTOUR ›\033[0m OSRM route: %s", f"{len(route)} pts" if route else "FAILED")

    # 5. Static map
    map_png = _generate_map(ordered, route)

    # 6. Leg distances & itinerary
    leg_km = [_haversine_km(ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1)]
    itinerary = _build_itinerary(ordered, leg_km, start_time)

    result: dict[str, Any] = {
        "success": True,
        "city": city,
        "num_stops": len(ordered),
        "total_km": round(sum(leg_km), 1),
        "stops": [{"name": s["name"], "lat": round(s["lat"], 5), "lon": round(s["lon"], 5)} for s in ordered],
        "itinerary": itinerary,
    }
    if skipped:
        result["skipped"] = skipped
    if map_png:
        result["_map_png"] = map_png  # bytes — stripped before passing to LLM context

    return result
