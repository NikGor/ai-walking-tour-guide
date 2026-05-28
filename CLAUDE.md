# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Solaris Pliny** — AI walking tour guide bot, fronted by a Telegram bot. User sends GPS coordinates (and optionally a photo) and receives a historically grounded narrative about that exact spot, in the voice of a chosen persona. Named after Pliny the Elder.

LLM, text-to-speech, and on-demand image generation all run through **OpenRouter** (using the OpenAI SDK as the HTTP client). Google Places and Gemini grounded-search are separate provider calls used by tools.

## Commands

```bash
make run                          # uvicorn dev server on :8000 with --reload
poetry install                    # install dependencies
poetry add <package>              # add dependency
poetry run pytest                 # run all tests
poetry run pytest tests/unit/test_prompt_builder.py::test_name  # single test
poetry run ruff check app         # lint
```

Manual endpoint test:
```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"latitude": 41.8902, "longitude": 12.4922, "persona": "roman_empire"}'
```

## Architecture

Request flow: `POST /chat` → `endpoints.py` (router) → `api_controller.handle_chat` (persists messages, loads history) → `AgentFactory.run` → structured `ChatResponse`.

- `main.py` — FastAPI app, Alembic auto-migrate on startup, launches Telegram polling if `TELEGRAM_BOT_TOKEN` is set
- `app/endpoints.py` — thin HTTP layer, one function per route
- `app/api_controller.py` — orchestration: conversation persistence + agent invocation
- `app/agent/agent_factory.py` — geocode/enrich → build messages → run agentic loop
- `app/agent/function_runner.py` — the agentic loop; round 1 sends tools **and** `response_format=ChatResponse`, so when no tool is called the structured answer comes back in one LLM call
- `app/agent/prompt_builder.py` — assembles system + persona prompt and the user message from `LocationContext`
- `app/agent/tools/` — one file per tool (`google_search`, `google_places_search`, `plan_city_tour`, `generate_image`); `utils/registry_utils.py` defines schemas, `utils/dispatcher_utils.py` routes calls
- `app/utils/geocoder_utils.py` — retrieval layer: Nominatim reverse-geocode + Overpass nearby POIs + Wikipedia summary/image + Wikimedia Commons archival photo, assembled into `LocationContext`
- `app/backend/openrouter_client.py` — OpenRouter chat client
- `app/db/` — SQLAlchemy async ORM, repository, session (SQLite local, Postgres on Railway)
- `app/telegram/` — `bot.py` (polling), `handlers.py` (commands/callbacks/dispatch), `tts.py` (persona-mapped voices), `ui_strings.json` (ru/en/de i18n)
- `app/time_travel/` — standalone feature: a Telegram Mini App + router that renders a location in a chosen historical era via image generation
- `app/prompts/` — `system.j2` (shared rules) + one `<persona>.j2` per persona

### Personas

`historian`, `architecture_expert`, `roman_empire`, `storyteller`, `medieval_resident`, `military_expert`, `deep_time`. Each is a `.j2` file in `app/prompts/` and an enum member in `Persona` (`agent/models/chat_models.py`). Adding a persona = add the enum value + the prompt file + UI labels in `ui_strings.json`.

## AI Design Intent

Responses must be **grounded** (retrieval-first, no hallucination). Flow: location resolve → enrich from OSM/Wikipedia/Commons → optional photo → LLM persona narrator generating from retrieved context. When facts are thin the narrator says so rather than inventing.

`ChatResponse` (the structured LLM output):
```json
{
  "text": "<narrative, 150–300 words>",
  "suggestions": ["<nearby place button>", "..."],
  "recommended_personas": ["<slug>", "..."]
}
```

## Environment

Copy `.env.example` → `.env`. Keys: `OPENROUTER_API_KEY` (required), `GOOGLE_API_KEY` + `GEMINI_API_KEY` (tools), `TELEGRAM_BOT_TOKEN` (bot), `APP_BASE_URL` (Time Travel Mini App), `SOLARIS_DB_URL` (defaults to local SQLite). See `.env.example` for `DEBUG_MODE` and `SOLARIS_RESPONSE_FORMAT`.
