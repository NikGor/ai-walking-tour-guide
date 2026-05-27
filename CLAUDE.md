# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Solaris Pliny** — AI walking tour guide bot. User sends GPS coordinates + optional photo and receives a historically grounded narrative about that location. Named after Pliny the Elder. All AI logic goes through the OpenAI SDK.

## Commands

```bash
make run                        # start dev server with --reload
poetry install                  # install dependencies
poetry add <package>            # add dependency
poetry run pytest               # run tests
poetry run pytest tests/test_x.py::test_name  # run single test
```

Manual endpoint test:
```bash
curl -X POST http://localhost:8000/chat
```

## Architecture

Request flow: `POST /chat` → `endpoints.py` (router) → `api_controller.py` (facade) → agent layer (not yet implemented).

- `main.py` — FastAPI app instantiation and router registration only
- `app/endpoints.py` — thin HTTP layer, one function per route
- `app/api_controller.py` — business logic facade; controllers call into the agent layer
- `app/agent/agent_factory.py` — will construct the agent pipeline (retrieval + LLM)
- `app/agent/prompt_builder.py` — will assemble system/user prompts per AI persona

## AI Design Intent

Responses must be **grounded** (retrieval-first, no hallucination). The flow is: location resolve → object detection → search aggregation → LLM historian. The model generates from retrieved context only.

Response shape target:
```json
{
  "title": "...",
  "summary": "...",
  "history": "...",
  "facts": [],
  "timeline": [],
  "related_people": [],
  "sources": [],
  "confidence": 0.91
}
```

## Environment

Copy `.env.example` → `.env`. Key vars: `OPENAI_API_KEY`, `GOOGLE_API_KEY`.
