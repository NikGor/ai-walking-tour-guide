from pathlib import Path

from jinja2 import Environment, FileSystemLoader

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

    def build_user_message(
        self,
        latitude: float | None,
        longitude: float | None,
        location_name: str | None,
        message: str | None = None,
        language: str | None = None,
    ) -> str:
        _LANG_NAMES = {"ru": "Russian", "en": "English", "de": "German"}
        lines = []
        if latitude is not None and longitude is not None:
            lines.append(f"Coordinates: {latitude}, {longitude}")
        if location_name:
            lines.append(f"Location name: {location_name}")
        if language and language != "auto":
            lines.append(f"Language: {_LANG_NAMES.get(language, language)}")
        if message:
            if lines:
                lines.append("")
            lines.append(f"User message: {message}")
        return "\n".join(lines)
