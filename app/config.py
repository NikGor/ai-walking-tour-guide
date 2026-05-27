"""App-level configuration read from environment variables."""

import os

# Response markup format sent to clients.
# Telegram handler uses this to set parse_mode and tell the LLM how to format.
# Supported: "plain" | "markdown" | "html"
RESPONSE_FORMAT: str = os.getenv("SOLARIS_RESPONSE_FORMAT", "html")
