"""Async retry utility for sync callables."""

import asyncio
import logging
from collections.abc import Callable
from typing import Any


logger = logging.getLogger(__name__)


async def call_with_retry(
    func: Callable,
    retryable_exceptions: tuple = (Exception,),
    max_retries: int = 3,
    base_delay: float = 1.0,
    context: str = "retry",
) -> Any:
    """
    Run a sync callable with automatic retry on specified exceptions.

    Executes ``func`` in a thread-pool executor so it doesn't block the event
    loop. Retries up to ``max_retries`` times with exponential back-off when
    the call raises one of ``retryable_exceptions``.

    Args:
        func: Sync callable to invoke (e.g. ``lambda: client.create(...)``).
        retryable_exceptions: Exception types that trigger a retry.
        max_retries: Maximum number of attempts (including the first call).
        base_delay: Seconds to wait before the first retry; doubles each time.
        context: Prefix for log messages (usually the caller module name).

    Returns:
        Whatever ``func`` returns on success.
    """
    loop = asyncio.get_event_loop()
    for attempt in range(max_retries):
        try:
            return await loop.run_in_executor(None, func)
        except retryable_exceptions as e:
            if attempt == max_retries - 1:
                logger.error(
                    f"{context}_retry_exhausted: all {max_retries} attempts failed: "
                    f"\033[31m{e!s}\033[0m"
                )
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(
                f"{context}_retry_{attempt + 1}: \033[33m{e!s}\033[0m "
                f"— retrying in {delay:.1f}s"
            )
            await asyncio.sleep(delay)
