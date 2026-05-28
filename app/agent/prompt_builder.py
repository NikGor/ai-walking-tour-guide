from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader

if TYPE_CHECKING:
    from app.utils.geocoder_utils import LocationContext

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class PromptBuilder:
    def __init__(self):
        self.env = Environment(
            loader=FileSystemLoader(str(_PROMPTS_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def build_system_prompt(self, persona: str) -> str:
        system = self.env.get_template("system.j2").render()
        persona_voice = self.env.get_template(f"{persona}.j2").render()
        return f"{system}\n\n## Your persona\n\n{persona_voice}"

    _FORMAT_HINTS: dict[str, str] = {
        "html": (
            "Format your response using basic Telegram HTML: "
            "<b>bold</b> for names and titles, <i>italic</i> for terms or asides, "
            "blank lines between paragraphs. No <p> tags. No headers."
        ),
        "markdown": (
            "Format your response using Markdown: "
            "**bold** for names and titles, *italic* for terms or asides, "
            "blank lines between paragraphs."
        ),
    }

    def build_user_message(
        self,
        latitude: float | None,
        longitude: float | None,
        location_ctx: LocationContext | None = None,
        location_name: str | None = None,  # backward-compat for tests
        message: str | None = None,
        language: str | None = None,
        response_format: str = "plain",
    ) -> str:
        _lang_names = {"ru": "Russian", "en": "English", "de": "German"}
        lines: list[str] = []

        if latitude is not None and longitude is not None:
            lines.append(f"Coordinates: {latitude}, {longitude}")

        # Resolved place name — prefer ctx, fall back to bare string
        name = location_ctx.name if location_ctx else location_name
        if name:
            lines.append(f"Location: {name}")

        # OSM tags from extratags (only when present — no invented fields)
        if location_ctx:
            if location_ctx.historic:
                lines.append(f"Historic type: {location_ctx.historic}")
            if location_ctx.architect:
                lines.append(f"Architect: {location_ctx.architect}")
            if location_ctx.start_date:
                lines.append(f"Built: {location_ctx.start_date}")
            if location_ctx.description:
                lines.append(f"Description: {location_ctx.description}")
            if location_ctx.wikipedia:
                lines.append(f"Wikipedia tag: {location_ctx.wikipedia}")

            # Nearby POIs from Overpass
            if location_ctx.nearby:
                lines.append("")
                lines.append("Nearby landmarks (from OpenStreetMap):")
                for poi in location_ctx.nearby[:8]:
                    poi_parts = [f"• {poi['name']}"]
                    tags: list[str] = []
                    if poi.get("historic"):
                        tags.append(poi["historic"])
                    elif poi.get("tourism"):
                        tags.append(poi["tourism"])
                    if poi.get("start_date"):
                        tags.append(f"est. {poi['start_date']}")
                    if poi.get("architect"):
                        tags.append(f"arch. {poi['architect']}")
                    if tags:
                        poi_parts.append(f"[{', '.join(tags)}]")
                    if poi.get("wikipedia"):
                        poi_parts.append(f"— {poi['wikipedia']}")
                    lines.append(" ".join(poi_parts))
                    if poi.get("description"):
                        lines.append(f"  {poi['description']}")

            # Wikipedia summary — grounding context for the LLM
            if location_ctx.wikipedia_summary:
                lines.append("")
                lines.append("Retrieved context (Wikipedia):")
                lines.append(location_ctx.wikipedia_summary)

        if language and language != "auto":
            lines.append("")
            lines.append(f"Language: {_lang_names.get(language, language)}")

        if message:
            lines.append("")
            lines.append(f"User message: {message}")

        hint = self._FORMAT_HINTS.get(response_format)
        if hint:
            lines.append("")
            lines.append(f"Formatting: {hint}")

        return "\n".join(lines)
