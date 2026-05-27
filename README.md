# Solaris Pliny

**AI walking tour guide** named after Pliny the Elder — the Roman historian who spent his life documenting the world around him.

Send a GPS location (and optionally a photo) and get a historically grounded narrative about the exact building, square, or street in front of you. No hallucination, no general city history — the story of *this* specific place.

---

## What it does

### 📍 Location stories
Share your GPS coordinates and receive a narrative tied to that exact spot: who stood there, what happened, what it looked like. The bot geocodes the location, fetches Wikipedia and OpenStreetMap context, then uses GPT-4.1 to write the story grounded in retrieved facts.

### 📷 Photo identification
Send a photo of a building or monument — the bot identifies it and makes it the centre of the story.

### 🎭 Seven narrative lenses (personas)
Switch between specialists, each reframing the same place through a different lens:

| Persona | Focus |
|---|---|
| 📜 Историк | Academic historian — dates, events, documented facts |
| 🏛 Архитектор | Architectural analysis — materials, style, construction, restoration |
| ⚔️ Римская империя | Everything through the lens of Rome: did the legions pass here? |
| 🎭 Сказитель | One specific human story — cinematic, conspiratorial, the detail the guidebooks left out |
| 🏚 Средневековый житель | Life at street level in the Middle Ages — smells, sounds, daily survival |
| 🗡 Военный историк | Battles, sieges, fortifications, the military geography of the place |
| 🧊 Геолог / ледниковый период | Deep time — ice ages, geology, what was here before humans |

The bot also **recommends which persona to try next** based on the location — e.g. a Roman amphitheatre gets `["deep_time", "medieval_resident"]` as suggestions.

### 🕰 Машина времени (Time Travel Lens)
A **Telegram Mini App** that generates an AI image of any location in any historical era.

- Pick a year (slider from −3000 to 2200, or type any value including millions for prehistoric)
- Toggle BCE / CE
- Choose from preset eras: 🦕 Динозавры · 🏺 Пирамиды · ⚔️ Средневековье · ⚓ Титаник · 🚀 Луна-69 · 🤖 Киберпанк 2150
- Select art style: 📷 Реалистично · 🤳 Селфи · 🎨 Арт эпохи
- **"Моя улица" mode**: upload your own street photo — the bot redraws it in the target era (img2img), preserving the composition

Pipeline: GPT-4.1 generates the historical scene description + image prompt → Gemini 2.0 Flash Preview Image Generation renders the image (Imagen 3 fallback).

Opened via the `🕰 Машина времени` button in the persistent keyboard (requires `APP_BASE_URL` env var — see below).

### 💬 Conversation awareness
- The bot never repeats facts it already told you in this session
- After covering the founding of a building it goes deeper: the people inside, a specific decade, a structural detail
- Nearby place suggestions are tracked — shown buttons don't appear again
- `/continue` continues the narrative with the active persona
- Persona switch mid-conversation: `/continue` picks up where the story left off, now through the new lens

---

## Commands

| Command | Action |
|---|---|
| `/start` | Introduction + show keyboard |
| `/whereami` | Story of the last known location |
| `/continue` | Continue the narrative (same persona, or new one after `/modes`) |
| `/modes` | Switch narrative lens |
| `/lang` | Set response language (auto / RU / EN / DE) |
| `/fmt` | Text format (HTML / Markdown / plain) |
| `/new` | Reset conversation, keep settings |
| `/history` | Token usage + cost for this session |
| `/settings` | Show all current settings |
| `/help` | This reference |

---

## Architecture

```
POST /chat
  ├── geocode (Nominatim + Overpass + Wikipedia)
  ├── build prompt (system.j2 + persona .j2 + conversation history)
  ├── agentic loop — GPT-4.1 via OpenRouter
  │     ├── tool: google_search          — fill specific factual gaps
  │     └── tool: google_places_search   — venue queries (NEVER invented)
  ├── parse structured response (text + suggestions + recommended_personas)
  └── save to DB (conversation · messages · suggestion history)

GET  /time-travel              → Telegram Mini App HTML
POST /time-travel/generate
  ├── geocode
  ├── GPT-4.1 → historical narrative + image prompt
  └── Gemini 2.0 Flash / Imagen 3 → image (base64)

GET  /conversations/{id}       → conversation history (JSON)
```

**Stack:** FastAPI · SQLAlchemy async · Alembic · SQLite (local) / PostgreSQL (Railway) · aiogram v3 · OpenRouter → GPT-4.1 · Google Gemini 2.0 Flash / Imagen 3 · google-genai · OpenStreetMap Nominatim + Overpass · Wikipedia REST API

---

## Running locally

```bash
cp .env.example .env
# fill in OPENROUTER_API_KEY, GOOGLE_API_KEY, GEMINI_API_KEY, TELEGRAM_BOT_TOKEN

poetry install
make run          # FastAPI on :8000 with --reload + Telegram polling
```

Manual endpoint test:
```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"latitude": 50.1104, "longitude": 8.6821, "persona": "historian"}' \
  | python3 -m json.tool
```

---

## Docker

```bash
docker-compose up --build
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | ✅ | LLM access (GPT-4.1 via OpenRouter) |
| `GOOGLE_API_KEY` | ✅ | Google Places API (venue search) |
| `GEMINI_API_KEY` | ✅ | Gemini image generation (Time Travel Lens) |
| `TELEGRAM_BOT_TOKEN` | for bot | Telegram bot token |
| `APP_BASE_URL` | for Mini App | Public HTTPS URL of this server (e.g. Railway URL). Enables the 🕰 Машина времени keyboard button. |
| `SOLARIS_DB_URL` | optional | Database URL. Default: `sqlite+aiosqlite:///./data/solaris.db` |
| `SOLARIS_RESPONSE_FORMAT` | optional | `html` (default) · `markdown` · `plain` |

---

## Hallucination prevention

Hard rules that override everything else:

- **No invented facts** — if sources are thin, one honest sentence beats a plausible-sounding story
- **No invented local color** — no past residents, craftsmen, or shopkeepers without a retrieved source; a convincing invented story is worse than silence because it will be believed
- **No invented venues** — `google_places_search` is always called for restaurant / café questions; venue names are never generated from model knowledge
- Storyteller persona ends with: *"One honest sentence is better than a charming lie"*

---

## Tests

```bash
poetry run pytest              # all tests
poetry run pytest tests/unit/  # unit only (no API keys needed)
```

Integration tests (`@pytest.mark.integration`) require `OPENROUTER_API_KEY`.
