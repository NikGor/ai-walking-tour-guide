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
        latitude: float,
        longitude: float,
        retrieved_context: str,
        detected_objects: str | None = None,
        message: str | None = None,
    ) -> str:
        lines = [
            f"Coordinates: {latitude}, {longitude}",
            f"Location name: (not yet resolved)",
        ]
        if detected_objects:
            lines.append(f"Detected in photo: {detected_objects}")
        if message:
            lines += ["", f"User message: {message}"]
        lines += [
            "",
            "Retrieved context:",
            retrieved_context,
        ]
        return "\n".join(lines)
